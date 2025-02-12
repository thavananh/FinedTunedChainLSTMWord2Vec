from os import read
import pandas as pd
from sklearn.model_selection import train_test_split

# train_data_aug = pd.read_csv('UIT-VSFC_train_augmented.csv')
# test_data_aug = pd.read_csv('UIT-VSFC_test_augmented.csv')
# dev_data_aug = pd.read_csv('UIT-VSFC_dev_augmented.csv')

# # Xóa các hàng mà cột 'sents' bị thiếu dữ liệu (NaN)
# train_data_aug.dropna(subset=['sents'], inplace=True)
# test_data_aug.dropna(subset=['sents'], inplace=True)
# dev_data_aug.dropna(subset=['sents'], inplace=True)

# train_data_aug.to_excel('UIT-VSFC_train_augmented.xlsx')
# test_data_aug.to_excel('UIT-VSFC_test_augmented.xlsx')
# dev_data_aug.to_excel('UIT-VSFC_dev_augmented.xlsx')

# train_data_aug.info()
# test_data_aug.info()
# dev_data_aug.info()


df = pd.read_excel('data_20k.xlsx')
df.info()

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

train_df.to_csv('train_aug_merge_old_data.csv', index=False)
test_df.to_csv('test_aug_merge_old_data.csv', index=False)
val_df.to_csv('val_aug_merge_old_data.csv', index=False)

train_df.info()
test_df.info()
val_df.info()