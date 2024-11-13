import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Đọc file CSV
file_path = 'df_train.csv'  # Thay đường dẫn tới file CSV
df = pd.read_csv(file_path)

# Đổi cột 'date' sang định dạng số để sử dụng trong hồi quy
df['date'] = pd.to_datetime(df['date'])
df['date_num'] = df['date'].map(datetime.toordinal)

# Tách dữ liệu thành X (ngày dạng số) và y (giá trị để dự đoán)
X = df[['date_num']]
y = df['price']  # Thay 'value' bằng tên cột giá trị cần dự đoán

# Khởi tạo mô hình hồi quy tuyến tính và huấn luyện mô hình
model = LinearRegression()
model.fit(X, y)

# Phương trình hồi quy tuyến tính
intercept = model.intercept_
slope = model.coef_[0]
#date_num=2030-26-3
#y1 = slope *date_num + intercept

#----------------------------
# Dự đoán giá trị y dựa trên phương trình hồi quy
y_pred = slope * X['date_num'] + intercept

# Vẽ biểu đồ với các điểm dữ liệu và đường hồi quy
plt.plot(df['date'], y_pred, color='red', label='Đường hồi quy tuyến tính')
#----------------------------

# Vẽ biểu đồ với các điểm dữ liệu
plt.scatter(df['date'], y, color='blue', label='Dữ liệu thực tế')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Biểu đồ dữ liệu')

# Hiển thị biểu đồ
plt.legend()
plt.show()
