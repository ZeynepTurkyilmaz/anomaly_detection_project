# Gerekli kütüphaneleri içe aktar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Rastgelelik için sabit seed değeri
np.random.seed(42)

# 📌 1. Normal sıcaklık ve basınç verilerini üret
temperature = np.random.normal(loc=25, scale=2, size=950)
pressure = np.random.normal(loc=1013, scale=5, size=950)

# 📌 2. Anormal sıcaklık ve basınç verileri
temperature_anomaly = np.random.normal(loc=40, scale=1, size=50)
pressure_anomaly = np.random.normal(loc=900, scale=5, size=50)

# 📌 3. Verileri birleştir
temp_all = np.concatenate([temperature, temperature_anomaly])
pres_all = np.concatenate([pressure, pressure_anomaly])
labels = np.concatenate([np.zeros(950), np.ones(50)])  # 0: normal, 1: anormal

# 📌 4. DataFrame oluştur
df = pd.DataFrame({
    'temperature': temp_all,
    'pressure': pres_all,
    'anomaly': labels
})

# 📌 5. Zaman sütunu ekle (1 saat aralıkla başlayarak)
df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')

# 📌 6. CSV dosyasına kaydet
df.to_csv("data/simulated_sensor_data.csv", index=False)
