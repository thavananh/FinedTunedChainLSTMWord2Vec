from calendar import c
from regex import B
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPool1D,
    Dropout,
    LayerNormalization,
    Bidirectional,
    LSTM,
    Concatenate,
    GlobalMaxPooling1D,
    Dense,
    MultiHeadAttention,
    BatchNormalization,
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
from base.BaseModel import BaseModel  # Quan trọng: Import BaseModel
from utils.Attribute import *
from utils.Block import *

log_dir = "logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


class CustomModel:
    def __init__(
        self,
        data_vocab_size,
        embedding_matrix,
        input_length=110,
        cnn_attributes_1=CnnAtribute(100, 1),
        cnn_attributes_2=CnnAtribute(100, 2),
        cnn_attributes_3=CnnAtribute(200, 3),
        cnn_attributes_4=CnnAtribute(200, 4),
        lstm_attributes_1=LSTMAttribute(300),
        lstm_attributes_2=LSTMAttribute(300),
        multi_head_attention_attributes=MultiHeadAttentionAttribute(4, 32),
        dense_attributes_1=DenseAttribute(256),
        dense_attributes_2=DenseAttribute(64),
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
        self.cnn_attributes_1 = cnn_attributes_1
        self.cnn_attributes_2 = cnn_attributes_2
        self.cnn_attributes_3 = cnn_attributes_3
        self.cnn_attributes_4 = cnn_attributes_4
        self.lstm_attributes_1 = lstm_attributes_1
        self.lstm_attributes_2 = lstm_attributes_2
        self.multi_head_attention_attributes = multi_head_attention_attributes
        self.dropout_combine = dropout_combine
        self.dense_attributes_1 = dense_attributes_1
        self.dense_attributes_2 = dense_attributes_2
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
        x = Dropout(0.5)(x)

        # Convolutional block
        cnn_block_1 = Conv1DBlock(
            filters=self.cnn_attributes_1.filter_size,
            kernel_size=self.cnn_attributes_1.kernel_size,
            dropout_rate=self.cnn_attributes_1.dropout_rate,
        )
        cnn_block_2 = Conv1DBlock(
            filters=self.cnn_attributes_2.filter_size,
            kernel_size=self.cnn_attributes_2.kernel_size,
            dropout_rate=self.cnn_attributes_2.dropout_rate,
        )
        cnn_block_3 = Conv1DBlock(
            filters=self.cnn_attributes_3.filter_size,
            kernel_size=self.cnn_attributes_3.kernel_size,
            dropout_rate=self.cnn_attributes_3.dropout_rate,
        )
        cnn_block_4 = Conv1DBlock(
            filters=self.cnn_attributes_4.filter_size,
            kernel_size=self.cnn_attributes_4.kernel_size,
            dropout_rate=self.cnn_attributes_4.dropout_rate,
        )

        cnn = cnn_block_1(x)
        cnn = cnn_block_2(cnn)
        cnn = cnn_block_3(cnn)
        cnn = cnn_block_4(cnn)

        # Bidirectional LSTM
        lstm_block_1 = LSTMBlock(
            units=self.lstm_attributes_1.units,
            dropout_rate=self.lstm_attributes_1.dropout_rate,
        )

        lstm_block_2 = LSTMBlock(
            units=self.lstm_attributes_2.units,
            dropout_rate=self.lstm_attributes_2.dropout_rate,
        )

        lstm = lstm_block_1(cnn)
        lstm = lstm_block_2(lstm)

        # Multi-head Attention
        multi_head_attention_block = MultiHeadAttentionBlock(
            num_heads=self.multi_head_attention_attributes.num_heads,
            key_dim=self.multi_head_attention_attributes.key_dim,
            dropout_rate=self.multi_head_attention_attributes.dropout_rate,
        )
        attention = multi_head_attention_block(lstm)

        # Feature pooling and concatenation
        cnn_pool = GlobalMaxPooling1D()(cnn)
        attn_pool = GlobalMaxPooling1D()(attention)
        combined = Concatenate()([cnn_pool, attn_pool])
        combined = LayerNormalization()(combined)
        combined = Dropout(self.dropout_combine)(combined)

        dense_block_1 = DenseBlock(
            units=self.dense_attributes_1.units,
            dropout_rate=self.dense_attributes_1.dropout_rate,
            activation=self.dense_attributes_1.activation,
        )
        dense_block_2 = DenseBlock(
            units=self.dense_attributes_2.units,
            dropout_rate=self.dense_attributes_2.dropout_rate,
            activation=self.dense_attributes_2.activation,
        )
        dense_block_3 = DenseBlock(
            units=self.dense_attributes_3.units,
            dropout_rate=self.dense_attributes_3.dropout_rate,
            activation=self.dense_attributes_3.activation,
        )

        dense = dense_block_1(combined)
        dense = dense_block_2(dense)
        output = dense_block_3(dense)

        self.model = Model(inputs=input_layer, outputs=output)

    def compile_model(self, learning_rate=1e-4, weight_decay=0.0):
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
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
            print("    Negative  Neutral  Positive\n")
            print(f"Negative   {cm[0][0]}      {cm[0][1]}      {cm[0][2]}\n")
            print(f"Neutral    {cm[1][0]}      {cm[1][1]}      {cm[1][2]}\n")
            print(f"Positive   {cm[2][0]}      {cm[2][1]}      {cm[2][2]}\n")
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap="Blues")
            plt.grid(False)
            plt.title("Bi-LSTM with Multi-Head Attention")
            plt.show()
