import pandas as pd

train_data = pd.read_csv('../UIT-VSFC_train_oversampling_20.csv')
dev_data = pd.read_csv('../uit-vsfc-cleaned-v4/UIT-VSFC_dev_cleaned.csv')
test_data = pd.read_csv('../uit-vsfc-cleaned-v4/UIT-VSFC_test_cleaned.csv')

train_data = train_data.dropna()
test_data = test_data.dropna()
dev_data = dev_data.dropna()

train_data.info()
test_data.info()
dev_data.info()

import numpy as np

# Kết hợp dữ liệu
# data_full = pd.concat([train_data, dev_data, test_data], ignore_index=True)

# Tự động chuyển giá trị không hợp lệ thành NaN
train_data.iloc[:, 1] = pd.to_numeric(train_data.iloc[:, 1], errors='coerce')

# Loại bỏ các hàng chứa NaN trong cả data_full
train_data = train_data.dropna(subset=[train_data.columns[1]])

# Chuyển cột cuối cùng thành kiểu số nguyên và mảng 1 chiều
label_idx_train = train_data.iloc[:, 1].astype(int).to_numpy()

print(label_idx_train)

# Chuyển đổi nhãn thành định dạng one-hot encoding
label_tf_train = tf_keras.utils.to_categorical(label_idx_train, num_classes=3, dtype='float32')