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
from base.BaseModel import BaseModel  # Quan trọng: Import BaseModel
from utils.Attribute import *
from utils.Block import *

log_dir = "logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

class KimCNNModel(BaseModel):
    def __init__(
        self,
        data_vocab_size,
        embedding_matrix,
        input_length=110,
        cnn_2d_attribute_1 = Cnn2DAttribute(filter_size=32, kernel_size=(3, 3)),
        cnn_2d_attribute_2 = Cnn2DAttribute(filter_size=32, kernel_size=(3, 3)),
        dense_attributes_3=DenseAttribute(3, activation="softmax"),
        dropout_features=0.0,
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

        conv2d_input = Reshape((self.input_length, self.embedding_output_dim, 1))(
            x_stacked
        )

        conv2d_block_1 = Conv2DBlock(
            filters=self.cnn_2d_attribute_1.filter_size, kernel_size=self.cnn_2d_attribute_1.kernel_size, activation=self.cnn_2d_attribute_1.activation, padding=self.cnn_2d_attribute_1.padding
        )
        conv2d_block_2 = Conv2DBlock(
            filters=self.cnn_2d_attribute_2.filter_size, kernel_size=self.cnn_2d_attribute_2.kernel_size, activation=self.cnn_2d_attribute_2.activation, padding=self.cnn_2d_attribute_2.padding
        )

        cnn_2d = conv2d_block_1(conv2d_input)
        cnn_2d = conv2d_block_2(cnn_2d)
        cnn_2d_pooled = GlobalMaxPooling2D()(cnn_2d)

        dense_block_3 = DenseBlock(
            units=3,
            dropout_rate=self.dense_attributes_3.dropout_rate,
            activation=self.dense_attributes_3.activation,
        )

        output = dense_block_3(cnn_2d_pooled)  # Final output layer

        self.model = Model(inputs=input_layer, outputs=output)

    
