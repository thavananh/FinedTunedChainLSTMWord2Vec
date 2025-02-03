import pandas as pd
import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, train_path, dev_path, test_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.label_tf_train = None
        self.label_tf_dev = None
        self.label_tf_test = None
        self.train_data_text = None
        self.dev_data_text = None
        self.test_data_text = None
        self.unprocessed_train_data = None
        self.unprocessed_dev_data = None
        self.unprocessed_test_data = None
        self.unprocessed_label_train = None
        self.unprocessed_label_dev = None
        self.unprocessed_label_test = None

    def load_data(self):
        # Load data from CSV files
        self.train_data = pd.read_csv(self.train_path)
        self.dev_data = pd.read_csv(self.dev_path)
        self.test_data = pd.read_csv(self.test_path)

        # Drop rows with NaN values
        self.train_data = self.train_data.dropna()
        self.dev_data = self.dev_data.dropna()
        self.test_data = self.test_data.dropna()

        self.train_data.info()
        self.dev_data.info()
        self.test_data.info()

        self.extract_text_data()

        self.unprocessed_train_data = self.train_data.copy()
        self.unprocessed_dev_data = self.dev_data.copy()
        self.unprocessed_test_data = self.test_data.copy()

        # Display info

        self.preprocess_labels()

    def preprocess_labels(self):
        # Convert labels to numeric and handle errors
        self.train_data.iloc[:, 1] = pd.to_numeric(
            self.train_data.iloc[:, 1], errors="coerce"
        )
        self.train_data = self.train_data.dropna(subset=[self.train_data.columns[1]])
        label_idx_train = self.train_data.iloc[:, 1].astype(int).to_numpy()
        self.unprocessed_label_train = label_idx_train
        self.label_tf_train = tf.keras.utils.to_categorical(
            label_idx_train, num_classes=3
        )

        self.test_data.iloc[:, 1] = pd.to_numeric(
            self.test_data.iloc[:, 1], errors="coerce"
        )
        self.test_data = self.test_data.dropna(subset=[self.test_data.columns[1]])
        label_idx_test = self.test_data.iloc[:, 1].astype(int).to_numpy()
        self.unprocessed_label_test = label_idx_test
        self.label_tf_test = tf.keras.utils.to_categorical(
            label_idx_test, num_classes=3
        )

        self.dev_data.iloc[:, 1] = pd.to_numeric(
            self.dev_data.iloc[:, 1], errors="coerce"
        )
        self.dev_data = self.dev_data.dropna(subset=[self.dev_data.columns[1]])
        label_idx_dev = self.dev_data.iloc[:, 1].astype(int).to_numpy()
        self.unprocessed_label_dev = label_idx_dev
        self.label_tf_dev = tf.keras.utils.to_categorical(label_idx_dev, num_classes=3)

    def extract_text_data(self):
        # Extract text data
        self.train_data_text = self.train_data.iloc[:, 0].values.tolist()
        self.test_data_text = self.test_data.iloc[:, 0].values.tolist()
        self.dev_data_text = self.dev_data.iloc[:, 0].values.tolist()

    def get_processed_data(self):
        # Return processed data

        return {
            "train_text": self.train_data_text,
            "train_labels": self.label_tf_train,
            "dev_text": self.dev_data_text,
            "dev_labels": self.label_tf_dev,
            "test_text": self.test_data_text,
            "test_labels": self.label_tf_test,
            "unprocessed_train_data": self.unprocessed_train_data,
            "unprocessed_dev_data": self.unprocessed_dev_data,
            "unprocessed_test_data": self.unprocessed_test_data,
            "unprocessed_label_train": self.unprocessed_label_train,
            "unprocessed_label_dev": self.unprocessed_label_dev,
            "unprocessed_label_test": self.unprocessed_label_test,
        }
