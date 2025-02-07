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


class CustomModel_3(BaseModel):
    def __init__(
        self,
        data_vocab_size,
        embedding_matrix,
        input_length=110,
        cnn_attributes_1=CnnAtribute(100, 1),
        cnn_attributes_2=CnnAtribute(100, 2),
        lstm_attributes_1=LSTMAttribute(300),
        multi_head_attention_attributes=MultiHeadAttentionAttribute(4, 32),
        dense_attributes_1=DenseAttribute(256),
        dense_attributes_3=DenseAttribute(3, activation="softmax"),
        dropout_features=0.0,
        dropout_combine=0.0,
        dropout_attention_pooled=0.0,
        attention_weight_activation='sigmoid'
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
        self.lstm_attributes_1 = lstm_attributes_1
        self.multi_head_attention_attributes = multi_head_attention_attributes
        self.dropout_attention_pooled = dropout_attention_pooled
        self.dropout_combine = dropout_combine
        self.dense_attributes_1 = dense_attributes_1
        self.dense_attributes_3 = dense_attributes_3
        self.attention_weight_activation = attention_weight_activation

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
        # Convolutional Path
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

        cnn = cnn_block_1(x)
        cnn = cnn_block_2(cnn)

        # Recurrent Path
        lstm_block_1 = LSTMBlock(units=self.lstm_attributes_1.units, dropout_rate=self.lstm_attributes_2.dropout_rate)

        lstm = lstm_block_1(cnn)
        
        # Self-Attention Layer
        multi_head_attention_block = MultiHeadAttentionBlock(
            num_heads=self.multi_head_attention_attributes.num_heads,
            key_dim=self.multi_head_attention_attributes.key_dim,
            dropout_rate=self.multi_head_attention_attributes.dropout_rate,
        )
        attention = multi_head_attention_block(lstm)

        attention_weights = Dense(1, activation=self.attention_weight_activation)(attention)  # Learnable attention weights
        attention_pooled = ReduceSumLayer(axis=1)(attention * attention_weights)  # Use custom layer
        attention_pooled = Dropout(self.dropout_attention_pooled)(attention_pooled)

        cnn_pooled = GlobalMaxPooling1D()(cnn)         

        # Concatenate the pooled features
        combine_feature = Concatenate()([cnn_pooled, attention_pooled])
        combine_feature = LayerNormalization()(combine_feature)
        combine_feature = Dropout(self.dropout_combine)(combine_feature)  
        
        dense_block_1 = DenseBlock(
            units=self.dense_attributes_1.units,
            dropout_rate=self.dense_attributes_1.dropout_rate,
            activation=self.dense_attributes_1.activation
        )
        
        dense_block_3 = DenseBlock(
            units=self.dense_attributes_3.units,
            dropout_rate=self.dense_attributes_3.dropout_rate,
            activation=self.dense_attributes_3.activation,
        )

        dense = dense_block_1(combine_feature)
        output = dense_block_3(dense)

        self.model = Model(inputs=input_layer, outputs=output)
