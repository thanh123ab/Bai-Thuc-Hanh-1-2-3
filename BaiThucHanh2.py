import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# 1. Mô phỏng dữ liệu
np.random.seed(42)
time_range = pd.date_range(start='2021-01-01', periods=1000, freq='h')  # Sửa 'H' thành 'h'
num_sensors = 10
data = []
for sensor_id in range(num_sensors):
    x = np.random.uniform(0, 100)
    y = np.random.uniform(0, 100)
    pm25 = np.random.normal(loc=35, scale=10, size=len(time_range))
    temperature = np.random.normal(loc=25, scale=5, size=len(time_range))
    humidity = np.random.uniform(30, 90, size=len(time_range))
    wind_speed = np.random.uniform(0, 10, size=len(time_range))
    for i in range(len(time_range)):
        data.append([time_range[i], sensor_id, x, y, pm25[i], temperature[i], humidity[i], wind_speed[i]])

df = pd.DataFrame(data, columns=['timestamp', 'sensor_id', 'x', 'y', 'pm25', 'temperature', 'humidity', 'wind_speed'])

# 2. DBSCAN: xác định điểm nóng ô nhiễm
snapshot = df[df['timestamp'] == df['timestamp'].iloc[0]]
features = snapshot[['x', 'y', 'pm25']]
features_scaled = StandardScaler().fit_transform(features)
dbscan = DBSCAN(eps=0.5, min_samples=2)
snapshot['cluster'] = dbscan.fit_predict(features_scaled)
sns.scatterplot(data=snapshot, x='x', y='y', hue='cluster', palette='tab10')
plt.title('Clusters of Pollution Hotspots')
plt.show()

# 3. Chỉ số rủi ro ô nhiễm
risk = df.groupby('sensor_id').apply(lambda x: pd.Series({
    'pm25_mean': x['pm25'].mean(),
    'exceed_freq': (x['pm25'] > 50).mean()
})).reset_index()
risk['risk_index'] = (risk['pm25_mean'] / risk['pm25_mean'].max() + risk['exceed_freq']) / 2
print("\nChỉ số rủi ro:\n", risk)

# 4. Chỉ số thời tiết bất lợi
df['weather_adversity'] = df['humidity'] * (1 / (df['wind_speed'] + 1))

# 5. Xu hướng ô nhiễm
df = df.sort_values(['sensor_id', 'timestamp'])
df['pm25_trend'] = df.groupby('sensor_id')['pm25'].transform(
    lambda x: x.rolling(window=24).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=False)
)

# 6. LSTM dự đoán PM2.5
def create_sequences(data, seq_length=24, forecast_horizon=6):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(X), np.array(y)

sensor_data = df[df['sensor_id'] == 0].sort_values('timestamp')
pm25_values = sensor_data['pm25'].values
X, y = create_sequences(pm25_values)
X = X[..., np.newaxis]  # reshape for LSTM

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(y.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, verbose=1)

# 7. Đánh giá và hiển thị kết quả
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
print(f"\nRMSE: {rmse:.2f}")

plt.plot(y_test[:, 0], label='Thực tế')
plt.plot(y_pred[:, 0], label='Dự đoán')
plt.legend()
plt.title('Dự đoán PM2.5 - Bước đầu tiên')
plt.xlabel('Thời điểm')
plt.ylabel('PM2.5')
plt.show()