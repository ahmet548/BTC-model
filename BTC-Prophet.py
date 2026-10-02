import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. VERİ İNDİRME VE HAZIRLAMA
# ==========================================
print("1. Veri indiriliyor...")
df = yf.download('BTC-USD', start='2020-01-01')

# Kolon yapısı güvenliği
if 'Close' in df.columns:
    data = df['Close'].values.reshape(-1, 1)
else:
    data = df.iloc[:, 0].values.reshape(-1, 1)

print("2. Veri 0-1 arasına ölçeklendiriliyor (Scaling)...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

training_data_len = int(np.ceil(len(data) * 0.8))
train_data = scaled_data[0:int(training_data_len), :]

print("3. Zaman pencereleri (Time Steps) oluşturuluyor...")
time_step = 60 
X_train, y_train = [], []

for i in range(time_step, len(train_data)):
    X_train.append(train_data[i-time_step:i, 0]) 
    y_train.append(train_data[i, 0])             

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(f"X_train Boyutu: {X_train.shape} | y_train Boyutu: {y_train.shape}")

# ==========================================
# 2. LSTM MODELİNİN KURULUMU VE EĞİTİMİ
# ==========================================
print("4. Sinir Ağı (LSTM) Mimarisi Kuruluyor...")
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=100, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# İYİLEŞTİRME: Modelin ezberlemesini önlemek için Early Stopping eklendi
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("5. Model Eğitimi Başlıyor (Early Stopping Aktif)...")
model.fit(
    X_train, y_train, 
    batch_size=32, 
    epochs=50, 
    validation_split=0.1, # Verinin %10'unu kendini test etmek için ayırır
    callbacks=[early_stop]
)
print("\nModel Eğitimi Başarıyla Tamamlandı!")

# ==========================================
# 3. TEST VE TAHMİN (PREDICTION) AŞAMASI
# ==========================================
print("6. Test Verisi Hazırlanıyor...")
test_data = scaled_data[training_data_len - time_step: , :]
X_test = []
y_test = data[training_data_len:, :] 

for i in range(time_step, len(test_data)):
    X_test.append(test_data[i-time_step:i, 0])

X_test = np.reshape(np.array(X_test), (X_test.shape[0], X_test.shape[1], 1))

print("7. Tahminler Yapılıyor ve Hata Hesaplanıyor...")
predictions = scaler.inverse_transform(model.predict(X_test))

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"Modelin Hata Payı (RMSE): {rmse:,.2f}")

# ==========================================
# 4. YARININ FİYATINI TAHMİN ETME
# ==========================================
print("\n8. Yarının Fiyatı Tahmin Ediliyor...")
last_60_days = scaled_data[-time_step:]
X_tomorrow = np.reshape(np.array([last_60_days]), (1, time_step, 1))

predicted_price = scaler.inverse_transform(model.predict(X_tomorrow))
print(f"\n>>> BİR SONRAKİ GÜN İÇİN TAHMİN EDİLEN BTC KAPANIŞ FİYATI: ${predicted_price[0][0]:,.2f} <<<\n")

# ==========================================
# 5. GÖRSELLEŞTİRME
# ==========================================
print("9. Sonuçlar Grafiğe Dökülüyor...")
train = df.iloc[:training_data_len].copy()
valid = df.iloc[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('LSTM Model Başarısı: Gerçek vs Tahmin')
plt.xlabel('Tarih', fontsize=18)
plt.ylabel('Kapanış Fiyatı (USD)', fontsize=18)
close_col = 'Close' if 'Close' in train.columns else train.columns[0]

plt.plot(train.index, train[close_col], color='blue', label='Eğitim (Train) Verisi')
plt.plot(valid.index, valid[close_col], color='orange', label='Gerçek (Test) Fiyatlar')
plt.plot(valid.index, valid['Predictions'], color='green', label='Model Tahminleri (Predictions)')
plt.legend(loc='lower right')
plt.show()