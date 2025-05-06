import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Sinh Dữ Liệu Giả Lập
np.random.seed(42)
random.seed(42)

# Tạo thời gian (5000 bản ghi)
start_time = datetime(2023, 1, 1)
end_time = datetime(2023, 12, 31)
time_range = [start_time + timedelta(minutes=i) for i in range(0, 5000)]

# Tạo vị trí (tọa độ giả lập)
locations = [(random.uniform(10.0, 21.0), random.uniform(105.0, 110.0)) for _ in range(5000)]  # (vĩ độ, kinh độ)

# Loại phương tiện
vehicle_types = ['Xe máy', 'Ô tô', 'Xe buýt']

# Tốc độ trung bình và mật độ giao thông
speeds = [random.uniform(10, 80) for _ in range(5000)]  # Giả lập tốc độ từ 10 đến 80 km/h
traffic_density = ['Thấp', 'Trung bình', 'Cao']
densities = [random.choice(traffic_density) for _ in range(5000)]

# Dữ liệu vào DataFrame
data = pd.DataFrame({
    'timestamp': time_range,
    'location': locations,
    'vehicle_type': [random.choice(vehicle_types) for _ in range(5000)],
    'avg_speed': speeds,
    'traffic_density': densities
})

# 2. Phân Cụm K-Means Để Xác Định Điểm Tắc Nghẽn
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute
data['traffic_density_num'] = data['traffic_density'].map({'Thấp': 0, 'Trung bình': 1, 'Cao': 2})

# Tiền xử lý dữ liệu
scaler = StandardScaler()
features = data[['hour', 'minute', 'traffic_density_num'] + ['avg_speed']]
scaled_features = scaler.fit_transform(features)

# Phân cụm K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Hiển thị các điểm tắc nghẽn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['location'].apply(lambda x: x[0]), y=data['location'].apply(lambda x: x[1]), hue=data['cluster'], palette='tab10')
plt.title("Clusters of Traffic Congestion Points")
plt.show()

# 3. Tính Toán Chỉ Số Mức Độ Nghiêm Trọng Của Tắc Nghẽn
def congestion_severity(row):
    if row['traffic_density'] == 'Cao' and row['avg_speed'] < 30:
        return 2  # Mức độ nghiêm trọng cao
    elif row['traffic_density'] == 'Trung bình' and row['avg_speed'] < 50:
        return 1  # Mức độ nghiêm trọng trung bình
    else:
        return 0  # Mức độ nghiêm trọng thấp

data['congestion_severity'] = data.apply(congestion_severity, axis=1)

# 4. Tạo Đặc Trưng "Giờ Cao Điểm"
def is_peak_hour(hour):
    return 1 if (7 <= hour < 9) or (17 <= hour < 19) else 0

data['peak_hour'] = data['hour'].apply(is_peak_hour)

# 5. Tạo Đặc Trưng "Tỷ Lệ Xe Lớn"
def vehicle_ratio(row):
    total_vehicles = 0
    large_vehicles = 0
    if row['vehicle_type'] in ['Ô tô', 'Xe buýt']:
        large_vehicles += 1
    total_vehicles += 1
    return large_vehicles / total_vehicles if total_vehicles > 0 else 0

data['large_vehicle_ratio'] = data.apply(vehicle_ratio, axis=1)

# 6. Xây Dựng Mô Hình Gradient Boosting Để Dự Đoán Mật Độ Giao Thông
# Tiền xử lý dữ liệu cho mô hình
X = data[['hour', 'minute', 'avg_speed', 'large_vehicle_ratio', 'peak_hour']]
y = data['traffic_density_num']  # Chuyển mật độ giao thông thành số

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Hiển Thị Kết Quả Dự Đoán
# Vẽ đồ thị dự đoán mật độ giao thông trên dữ liệu test
plt.figure(figsize=(10, 6))
sns.countplot(x=y_pred, palette='viridis')
plt.title('Distribution of Predicted Traffic Density')
plt.xlabel('Traffic Density')
plt.ylabel('Frequency')
plt.show()