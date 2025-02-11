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
from base.BaseModel import BaseModel  # Quan trọng: Import BaseModel
from utils.Attribute import *
from utils.Block import *

log_dir = "logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


class CZhouLSTMCNNModel(BaseModel):
    def __init__(
        self,
        data_vocab_size,
        embedding_matrix,
        input_length=110,
        cnn_attributes_1=CnnAtribute(100, 1),
        cnn_attributes_2=CnnAtribute(100, 2),
        cnn_attributes_3=CnnAtribute(200, 3),
        cnn_attributes_4=CnnAtribute(200, 4),
        cnn_2d_attribute_1=Cnn2DAttribute(filter_size=32, kernel_size=(3, 3)),
        cnn_2d_attribute_2=Cnn2DAttribute(filter_size=32, kernel_size=(3, 3)),
        lstm_attributes_1=LSTMAttribute(300),
        lstm_attributes_2=LSTMAttribute(300),
        dense_attributes_1=DenseAttribute(256),
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
        self.cnn_2d_attribute_1 = cnn_2d_attribute_1
        self.cnn_2d_attribute_2 = cnn_2d_attribute_2
        self.lstm_attributes_1 = lstm_attributes_1
        self.lstm_attributes_2 = lstm_attributes_2

        self.dropout_combine = dropout_combine
        self.dense_attributes_1 = dense_attributes_1
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

        ### Conv1D Path ###
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

        lstm_block_1 = LSTMBlock(
            units=self.lstm_attributes_1.units,
            dropout_rate=self.lstm_attributes_1.dropout_rate,
        )
        lstm_block_2 = LSTMBlock(
            units=self.lstm_attributes_1.units,
            dropout_rate=self.lstm_attributes_2.dropout_rate,
        )

        lstm = lstm_block_1(cnn)
        lstm = lstm_block_2(lstm)

        conv2d_input = Reshape((self.input_length, self.embedding_output_dim, 1))(x)

        conv2d_block_1 = Conv2DBlock(
            filters=self.cnn_2d_attribute_1.filter_size,
            kernel_size=self.cnn_2d_attribute_1.kernel_size,
            activation=self.cnn_2d_attribute_1.activation,
            padding=self.cnn_2d_attribute_1.padding,
        )
        conv2d_block_2 = Conv2DBlock(
            filters=self.cnn_2d_attribute_2.filter_size,
            kernel_size=self.cnn_2d_attribute_2.kernel_size,
            activation=self.cnn_2d_attribute_2.activation,
            padding=self.cnn_2d_attribute_2.padding,
        )

        cnn_2d = conv2d_block_1(conv2d_input)
        cnn_2d = conv2d_block_2(cnn_2d)

        cnn_2d_pooled = GlobalMaxPooling2D()(cnn_2d)
        cnn_pooled = GlobalMaxPooling1D()(cnn)
        bi_lstm_pooled = GlobalMaxPooling1D()(lstm)

        ### Combine All Features ###
        combine_feature = Concatenate()([bi_lstm_pooled, cnn_pooled, cnn_2d_pooled])
        combine_feature = LayerNormalization()(combine_feature)
        combine_feature = Dropout(self.dropout_combine)(combine_feature)

        # Dense layers with L2 regularization (optional)
        dense_block_1 = DenseBlock(
            units=self.dense_attributes_1.units,
            dropout_rate=self.dense_attributes_1.dropout_rate,
            activation=self.dense_attributes_1.activation,
        )

        dense_block_3 = DenseBlock(
            units=self.dense_attributes_3.units,
            dropout_rate=self.dense_attributes_3.dropout_rate,
            activation=self.dense_attributes_3.activation,
        )

        dense = dense_block_1(combine_feature)

        output = dense_block_3(dense)

        self.model = Model(inputs=input_layer, outputs=output)
