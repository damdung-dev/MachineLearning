import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# Đọc file CSV
file_path = 'df_train.csv'  # Thay đường dẫn tới file CSV
df = pd.read_csv(file_path)

# Giả sử CSV có các cột: 'date', 'value1', 'value2'
df['date'] = pd.to_datetime(df['date'])
df['date_num'] = df['date'].map(datetime.toordinal)  # Chuyển 'date' thành số

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Tách dữ liệu thành X (date_num, value1) và y (value2)
X = df[['date_num', 'month']]
y = df['price']

# Huấn luyện mô hình hồi quy tuyến tính
model.fit(X, y)

# Dự đoán y dựa trên mô hình hồi quy
y_pred = model.predict(X)

# Tạo biểu đồ không gian 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Vẽ các điểm dữ liệu thực tế
ax.scatter(df['date_num'], df['month'], y, color='blue', label='Dữ liệu thực tế')

# Vẽ mặt phẳng hồi quy tuyến tính (dự đoán)
# Tạo lưới giá trị của date_num và value1 để vẽ mặt phẳng
date_num_range = np.linspace(df['date_num'].min(), df['date_num'].max(), 100)
value1_range = np.linspace(df['month'].min(), df['month'].max(), 100)
date_num_grid, value1_grid = np.meshgrid(date_num_range, value1_range)

# Dự đoán giá trị y (value2) từ mặt phẳng hồi quy
y_grid = model.predict(np.c_[date_num_grid.ravel(), value1_grid.ravel()])
y_grid = y_grid.reshape(date_num_grid.shape)

# Vẽ mặt phẳng hồi quy tuyến tính
ax.plot_surface(date_num_grid, value1_grid, y_grid, color='red', alpha=0.5, rstride=100, cstride=100, label='Mặt phẳng hồi quy')

# Nhãn trục và tiêu đề
ax.set_xlabel('Date (num)')
ax.set_ylabel('Month')
ax.set_zlabel('price')
ax.set_title('Biểu đồ không gian 3D và hồi quy tuyến tính')

# Hiển thị biểu đồ
plt.legend()
plt.show()
