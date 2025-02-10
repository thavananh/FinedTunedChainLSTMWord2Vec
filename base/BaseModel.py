# base/base_model.py
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

from utils.Attribute import WarmUp

class BaseModel:
    def __init__(self, input_length, dropout_combine, dense_attributes_3, log_dir="logs"):
        self.input_length = input_length
        self.model = None
        self.history = None
        self.dropout_combine = dropout_combine
        self.dense_attributes_3 = dense_attributes_3
        self.tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model method")

    def compile_model(self, learning_rate=1e-4, weight_decay=0.0):
        if self.model is None:
            self.build_model()  # Call build_model if it hasn't been called yet
        lr_schedule = WarmUp(initial_lr=learning_rate, warmup_steps=500, decay_steps=10000)
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=500):
        early_stop = EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        )
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0)

    def generate_classification_report(self, y_true, y_pred, labels=["Negative", "Neutral", "Positive"]):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        print(classification_report(y_true_labels, y_pred_labels, target_names=labels, zero_division=0, digits=3))

    def plot_confusion_matrix(self, y_true, y_pred, labels=["Negative", "Neutral", "Positive"], is_print_terminal=False):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2])

        if is_print_terminal:
            print("\nConfusion Matrix:\n")
            # ... (Phần in ma trận)
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap="Blues")
            plt.grid(False)
            plt.title("Confusion Matrix")
            plt.show()