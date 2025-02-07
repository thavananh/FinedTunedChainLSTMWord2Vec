from calendar import c
from math import comb
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


class CustomModel_2(BaseModel):
    def __init__(
        self,
        data_vocab_size,
        embedding_matrix,
        input_length=110,
        lstm_attributes_1=LSTMAttribute(300),
        lstm_attributes_2=LSTMAttribute(300),
        lstm_attributes_3=LSTMAttribute(300),
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
        self.lstm_attributes_1 = lstm_attributes_1
        self.lstm_attributes_2 = lstm_attributes_2
        self.lstm_attributes_3 = lstm_attributes_3
        self.multi_head_attention_attributes = multi_head_attention_attributes
        self.dropout_combine = dropout_combine
        self.dense_attributes_1 = dense_attributes_1
        self.dense_attributes_2 = dense_attributes_2
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
        # Bidirectional LSTM
        lstm_block_1 = LSTMBlock(
            units=self.lstm_attributes_1.units,
            dropout_rate=self.lstm_attributes_1.dropout_rate,
        )
        lstm_block_2 = LSTMBlock(
            units=self.lstm_attributes_1.units,
            dropout_rate=self.lstm_attributes_2.dropout_rate,
        )
        lstm_block_3 = LSTMBlock(
            units=self.lstm_attributes_3.units,
            dropout_rate=self.lstm_attributes_3.dropout_rate,
        )
        lstm = lstm_block_1(x)
        lstm = lstm_block_2(lstm)
        lstm = lstm_block_3(lstm)
        # Multi-head Attention
        multi_head_attention_block = MultiHeadAttentionBlock(
            num_heads=self.multi_head_attention_attributes.num_heads,
            key_dim=self.multi_head_attention_attributes.key_dim,
            dropout_rate=self.multi_head_attention_attributes.dropout_rate,
        )
        attention = multi_head_attention_block(lstm)

        # Feature pooling and concatenation
        attn_pool = GlobalMaxPooling1D()(attention)
        lstm_pool = GlobalMaxPooling1D()(lstm)
        combined = Concatenate()([lstm_pool, attn_pool])
        combined = LayerNormalization()(combined)
        combined = Dropout(self.dropout_combine)(combined)

        dense_block_1 = DenseBlock(
            units=self.dense_attributes_1.units,
            dropout_rate=self.dense_attributes_1.dropout_rate,
            activation=self.dense_attributes_1.activation
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
