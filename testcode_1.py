import pandas as pd

train_data_aug = pd.read_csv('UIT-VSFC_train_augmented.csv')
test_data_aug = pd.read_csv('UIT-VSFC_test_augmented.csv')
dev_data_aug = pd.read_csv('UIT-VSFC_dev_augmented.csv')

# Xóa các hàng mà cột 'sents' bị thiếu dữ liệu (NaN)
train_data_aug.dropna(subset=['sents'], inplace=True)
test_data_aug.dropna(subset=['sents'], inplace=True)
dev_data_aug.dropna(subset=['sents'], inplace=True)

train_data_aug.to_excel('UIT-VSFC_train_augmented.xlsx')
test_data_aug.to_excel('UIT-VSFC_test_augmented.xlsx')
dev_data_aug.to_excel('UIT-VSFC_dev_augmented.xlsx')

train_data_aug.info()
test_data_aug.info()
dev_data_aug.info()
