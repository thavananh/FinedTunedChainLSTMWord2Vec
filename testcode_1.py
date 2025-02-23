from os import read
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


df = pd.read_excel('data_16k.xlsx')
df.info()

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=2004)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=2004)

train_df.to_csv('new_train_16k.csv', index=False)
test_df.to_csv('new_test_16k.csv', index=False)
val_df.to_csv('new_dev_16k.csv', index=False)

train_df.info()
test_df.info()
val_df.info()

# Separate features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

# Apply Random Over-Sampling
ros = RandomOverSampler(random_state=2004)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Convert back to DataFrame
train_df_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
train_df_resampled["label"] = y_resampled  # Add the label column back

# Save the new training set
train_df_resampled.to_csv('balanced_train_16k.csv', index=False)

train_df_resampled.info()