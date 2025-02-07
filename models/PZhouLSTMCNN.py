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


class CustomModel_4(BaseModel):
    def __init__(
        self,
        data_vocab_size,
        embedding_matrix,
        input_length=110,
        cnn_2d_attribute_1 = Cnn2DAttribute(filters=32, kernel_size=(3, 3)),
        lstm_attributes_1=LSTMAttribute(300),
        dense_attributes_3=DenseAttribute(3, activation="softmax"),
        dropout_features=0.0,
    ):
        self.data_vocab_size = data_vocab_size
        self.embedding_matrix = embedding_matrix
        self.input_length = input_length
        self.model = None
        self.history = None

        self.embedding_output_dim = embedding_matrix.shape[1]
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.cnn_2d_attribute_1 = cnn_2d_attribute_1
        self.dropout_features = dropout_features
        self.lstm_attributes_1 = lstm_attributes_1
        self.dense_attributes_3 = dense_attributes_3

    def build_model(self):
        input_layer = Input(shape=(self.input_length,))
        x = Embedding(
            input_dim=self.data_vocab_size,
            output_dim=self.embedding_output_dim,
            embeddings_initializer=self.initializer,
            weights=[self.embedding_matrix],
            trainable=False,
        )(input_layer)
        x = Dropout(self.dropout_features)(x)

        lstm_block_1 = LSTMBlock(
            units=self.lstm_attributes_1.units,
            dropout_rate=self.lstm_attributes_1.dropout_rate,
        )

        lstm = lstm_block_1(x)

        blstm_output_reshaped = tf.reshape(lstm, shape=[-1, tf.shape(lstm)[1], 2, self.lstm_attributes_1.units]) # for reduce sum
        conv_input = tf.reduce_sum(blstm_output_reshaped, axis=2) # for reduce sum
        conv_input_expanded = tf.expand_dims(conv_input, axis=-1) # Add channel dimension


        conv2d_block_1 = Conv2DBlock(
            filters=self.cnn_2d_attribute_1.filter_size, kernel_size=self.cnn_2d_attribute_1.kernel_size, activation=self.cnn_2d_attribute_1.activation, padding=self.cnn_2d_attribute_1.padding
        )

        cnn_2d = conv2d_block_1(conv_input_expanded)
        flatten_output = layers.Flatten()(cnn_2d)

        dense_block_3 = DenseBlock(
            units=self.dense_attributes_3.units,
            dropout_rate=self.dense_attributes_3.dropout_rate,
            activation=self.dense_attributes_3.activation,
        )
        output = dense_block_3(flatten_output)

        self.model = Model(inputs=input_layer, outputs=output)

