from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
import joblib
import os
import json
import time
from datetime import datetime, timezone

app = FastAPI()


def get_realtime_price(symbol):
    """Anlık (canlı) fiyatı çeker - Google'daki fiyatla eşleşir."""
    yf_symbol = resolve_symbol(symbol)
    try:
        ticker = yf.Ticker(yf_symbol)
        price = ticker.fast_info.get("lastPrice", None)
        if price is None:
            price = ticker.fast_info.get("last_price", None)
        return price
    except Exception as e:
        print(f"[REALTIME] Could not fetch realtime price: {e}")
        return None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# AYARLAR
# ============================================================
TIME_STEP = 60
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
CACHE_TTL_SECONDS = 3600  # 1 saat

os.makedirs(CACHE_DIR, exist_ok=True)

memory_cache = {}


def resolve_symbol(symbol):
    if symbol in ["BTC", "ETH", "SOL"]:
        return f"{symbol}-USD"
    elif symbol == "BFREN":
        return "BFREN.IS"
    return symbol


def get_cache_paths(symbol):
    safe_name = symbol.replace("-", "_").replace(".", "_")
    return {
        "model": os.path.join(CACHE_DIR, f"{safe_name}_model.keras"),
        "scaler": os.path.join(CACHE_DIR, f"{safe_name}_scaler.pkl"),
        "data": os.path.join(CACHE_DIR, f"{safe_name}_data.npz"),
        "meta": os.path.join(CACHE_DIR, f"{safe_name}_meta.json"),
    }


def is_cache_valid(symbol):
    paths = get_cache_paths(symbol)
    for path in paths.values():
        if not os.path.exists(path):
            return False
    try:
        with open(paths["meta"], "r") as f:
            meta = json.load(f)
        age = time.time() - meta["trained_at"]
        if age > CACHE_TTL_SECONDS:
            print(f"[CACHE EXPIRED] {symbol} - {age:.0f}s ago, will retrain")
            return False
        return True
    except Exception:
        return False


def load_from_disk(symbol):
    paths = get_cache_paths(symbol)
    print(f"[DISK CACHE] Loading {symbol} model from disk...")
    model = load_model(paths["model"])
    scaler = joblib.load(paths["scaler"])
    npz = np.load(paths["data"], allow_pickle=True)
    scaled_data = npz["scaled_data"]
    raw_data = npz["raw_data"]
    dates = pd.DatetimeIndex(npz["dates"])
    entry = {
        "model": model,
        "scaler": scaler,
        "scaled_data": scaled_data,
        "raw_data": raw_data,
        "dates": dates,
    }
    memory_cache[symbol] = entry
    print(f"[DISK CACHE] {symbol} model loaded [OK]")
    return entry


def save_to_disk(symbol, model, scaler, scaled_data, raw_data, dates):
    paths = get_cache_paths(symbol)
    model.save(paths["model"])
    joblib.dump(scaler, paths["scaler"])
    np.savez(
        paths["data"],
        scaled_data=scaled_data,
        raw_data=raw_data,
        dates=dates.values
    )
    with open(paths["meta"], "w") as f:
        json.dump({"trained_at": time.time(), "symbol": symbol}, f)
    print(f"[DISK SAVE] {symbol} model saved to disk [OK]")


def train_model_for_symbol(symbol):
    yf_symbol = resolve_symbol(symbol)
    print(f"\n{'='*60}")
    print(f"[TRAIN] Training model for {symbol} ({yf_symbol})...")
    print(f"{'='*60}")

    df = yf.download(yf_symbol, period="2y", progress=False)
    if df.empty:
        raise ValueError(f"No data found for {yf_symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if 'Close' in df.columns:
        data = df['Close'].values.reshape(-1, 1)
    else:
        data = df.iloc[:, 0].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) <= TIME_STEP:
        raise ValueError("Not enough data")

    X_train, y_train = [], []
    for i in range(TIME_STEP, len(scaled_data)):
        X_train.append(scaled_data[i - TIME_STEP:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    print(f"[TRAIN] Fitting model (3 epochs)...")
    model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1)
    print(f"[TRAIN] {symbol} training complete [OK]")

    save_to_disk(symbol, model, scaler, scaled_data, data, df.index)

    entry = {
        "model": model,
        "scaler": scaler,
        "scaled_data": scaled_data,
        "raw_data": data,
        "dates": df.index,
    }
    memory_cache[symbol] = entry
    print(f"[CACHE] {symbol} model cached in memory [OK]")
    return entry


def get_model(symbol):
    if symbol in memory_cache:
        print(f"[MEMORY HIT] {symbol} - using cached model")
        return memory_cache[symbol]

    if is_cache_valid(symbol):
        return load_from_disk(symbol)

    return train_model_for_symbol(symbol)


@app.get("/api/forecast")
def get_forecast(symbol: str = "BTC", days: int = 30):
    try:
        entry = get_model(symbol)

        model = entry["model"]
        scaler = entry["scaler"]
        scaled_data = entry["scaled_data"]
        data = entry["raw_data"]
        dates = entry["dates"]

        # --- Anlık fiyatı çek ---
        realtime_price = get_realtime_price(symbol)
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        last_date_str = dates[-1].strftime('%Y-%m-%d')

        historical_data = data.flatten().tolist()
        historical_labels = dates.strftime('%Y-%m-%d').tolist()

        # Bugünün canlı fiyatını tarihsel veriye ekle
        today_is_new = False
        if realtime_price is not None:
            if last_date_str == today_str:
                historical_data[-1] = float(realtime_price)
                print(f"[REALTIME] Updated today's price: ${realtime_price:,.2f}")
            else:
                historical_data.append(float(realtime_price))
                historical_labels.append(today_str)
                today_is_new = True
                print(f"[REALTIME] Added today's price: ${realtime_price:,.2f}")

        # --- Tahmin: ÖNCEKİ veriden (dünkü kapanıştan) başla ---
        # Böylece bugünün tahmini de dahil olur
        last_input = scaled_data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
        
        # Bugün dahil toplam kaç gün tahmin edelim
        # Eğer bugün yeni eklendiyse: bugün + days gün = days+1 tahmin
        # Eğer bugün zaten verideyse: days tahmin (yarından itibaren)
        total_forecast_days = days + 1 if today_is_new else days

        forecast_scaled = []
        current_input = last_input

        for _ in range(total_forecast_days):
            pred = model.predict(current_input, verbose=0)
            forecast_scaled.append(pred[0, 0])
            current_input = np.append(current_input[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        forecast_prices = scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).flatten().tolist()

        # Tahmin tarihleri: dünkü kapanışın ertesi gününden başla
        original_last_date = dates[-1]
        future_dates = pd.date_range(
            start=original_last_date + pd.Timedelta(days=1),
            periods=total_forecast_days
        )
        future_labels = future_dates.strftime('%Y-%m-%d').tolist()

        # Bugün hem tarihsel hem tahmin verisinde olacak
        # Tarihsel: canlı fiyat | Tahmin: modelin tahmin ettiği kapanış
        all_labels = historical_labels + [l for l in future_labels if l not in historical_labels]
        
        # Tahmin verisini doğru pozisyona yerleştir
        forecast_data_padded = [None] * len(all_labels)
        upper_data_padded = [None] * len(all_labels)
        lower_data_padded = [None] * len(all_labels)
        
        for i, fl in enumerate(future_labels):
            idx = all_labels.index(fl)
            forecast_data_padded[idx] = forecast_prices[i]
            upper_data_padded[idx] = forecast_prices[i] * 1.05
            lower_data_padded[idx] = forecast_prices[i] * 0.95
        
        # Tarihsel veriyi doğru boyuta getir
        historical_data_padded = [None] * len(all_labels)
        for i, val in enumerate(historical_data):
            historical_data_padded[i] = val

        # Bugünün tahmin edilen kapanışını ayrıca gönder
        today_forecast = None
        if today_is_new and len(forecast_prices) > 0:
            today_forecast = forecast_prices[0]  # İlk tahmin = bugünün tahmini

        return {
            "labels": all_labels,
            "historicalData": historical_data_padded,
            "forecastData": forecast_data_padded,
            "upperData": upper_data_padded,
            "lowerData": lower_data_padded,
            "realtimePrice": realtime_price,
            "todayForecast": today_forecast,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/retrain")
def retrain(symbol: str = "BTC"):
    try:
        if symbol in memory_cache:
            del memory_cache[symbol]
        paths = get_cache_paths(symbol)
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)
        train_model_for_symbol(symbol)
        return {"status": "ok", "message": f"{symbol} model retrained"}
    except Exception as e:
        return {"error": str(e)}