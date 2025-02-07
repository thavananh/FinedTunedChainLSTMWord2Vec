#Kim LSTM maybe better than CZhou Model because it have stack embedding
#https://arxiv.org/pdf/1408.5882.pdf

# https://arxiv.org/pdf/1511.08630
# CZhou have conv2d
from calendar import c
from regex import B
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    Conv2D,
    MaxPool1D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    GlobalMaxPooling1D,
    GlobalMaxPooling2D,
    Dense,
    Bidirectional,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
    Concatenate,
    Reshape,
)

from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from torch import dropout
from Attribute import *
from Block import *

log_dir = "logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
from Block import *


class KimCNN:
    def __init__(
        self,
        data_vocab_size,
        embedding_matrix,
        input_length=110,
        cnn_2d_attribute_1 = Cnn2DAttribute(filters=32, kernel_size=(3, 3)),
        cnn_2d_attribute_2 = Cnn2DAttribute(filters=32, kernel_size=(3, 3)),
        dense_attributes_3=DenseAttribute(3, activation="softmax"),
        dropout_features=0.0,
        dropout_combine=0.0,
    ):
        self.data_vocab_size = data_vocab_size
        self.embedding_matrix = embedding_matrix
        self.input_length = input_length
        self.model = None
        self.history = None

        # Model hyperparameters
        self.embedding_output_dim = embedding_matrix.shape[1]
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.dropout_features = dropout_features
        self.cnn_2d_attribute_1 = cnn_2d_attribute_1
        self.cnn_2d_attribute_2 = cnn_2d_attribute_2
        self.dropout_combine = dropout_combine

        self.dense_attributes_3 = dense_attributes_3

    def build_model(self):
        input_layer = Input(shape=(self.input_length,))

        # Embedding layer
        x = Embedding(
            input_dim=self.data_vocab_size,
            output_dim=self.embedding_output_dim,
            embeddings_initializer=self.initializer,
            weights=[self.embedding_matrix],
            trainable=False,
        )(input_layer)
        x = Dropout(self.dropout_features)(x)

        x_1 = Embedding(
            input_dim=self.data_vocab_size,
            output_dim=self.embedding_output_dim,
            embeddings_initializer=self.initializer,
            weights=[self.embedding_matrix],
            trainable=True,
        )(input_layer)

        x_1 = Dropout(self.dropout_features)(x_1)
        
        x_stacked = tf.stack([x, x_1], axis=1)


        ### Conv2D Path ###
        # Reshape for Conv2D: (batch_size, height, width, channels)
        conv2d_input = Reshape((self.input_length, self.embedding_output_dim, 1))(
            x_stacked
        )  # (batch_size, 110, 300, 1)

        conv2d_block_1 = Conv2DBlock(
            filters=self.cnn_2d_attribute_1.filter_size, kernel_size=self.cnn_2d_attribute_1.kernel_size, activation=self.cnn_2d_attribute_1.activation, padding=self.cnn_2d_attribute_1.padding
        )
        conv2d_block_2 = Conv2DBlock(
            filters=self.cnn_2d_attribute_2.filter_size, kernel_size=self.cnn_2d_attribute_2.kernel_size, activation=self.cnn_2d_attribute_2.activation, padding=self.cnn_2d_attribute_2.padding
        )

        cnn_2d = conv2d_block_1(conv2d_input)
        cnn_2d = conv2d_block_2(cnn_2d)

        cnn_2d_pooled = GlobalMaxPooling2D(cnn_2d)


        dense_block_3 = DenseBlock(
            units=self.dense_attributes_3.units,
            dropout_rate=self.dense_attributes_3.dropout_rate,
            activation=self.dense_attributes_3.activation,
        )

        output = dense_block_3(cnn_2d_pooled)  # Final output layer

        self.model = Model(inputs=input_layer, outputs=output)

    def compile_model(self, learning_rate=1e-4, weight_decay=0.0):
        lr_schedule = WarmUp(initial_lr=learning_rate, warmup_steps=500, decay_steps=10000)
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(
        self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=500
    ):
        early_stop = EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                early_stop,
                tensorboard_callback,
            ],  # Thêm tensorboard callback vào
            verbose=1,
        )

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0)

    def generate_classification_report(
        self, y_true, y_pred, labels=["Negative", "Neutral", "Positive"]
    ):
        print(y_true)
        print(y_pred)
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        print(
            classification_report(
                y_true_labels,
                y_pred_labels,
                target_names=labels,
                zero_division=0,
                digits=3,
            )
        )

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        labels=["Negative", "Neutral", "Positive"],
        is_print_terminal=False,
    ):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2])

        # Optionally print confusion matrix in the terminal
        if is_print_terminal:
            print("\nConfusion Matrix:\n")
            print("             Negative        Neutral         Positive\n")
            print(f"Negative   {cm[0][0]}      {cm[0][1]}      {cm[0][2]}\n")
            print(f"Neutral    {cm[1][0]}      {cm[1][1]}      {cm[1][2]}\n")
            print(f"Positive   {cm[2][0]}      {cm[2][1]}      {cm[2][2]}\n")
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap="Blues")
            plt.grid(False)
            plt.title("Bi-LSTM with Multi-Head Attention")
            plt.show()
