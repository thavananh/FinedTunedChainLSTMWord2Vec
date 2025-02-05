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

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras import layers
class Conv1DBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=100,
        kernel_size=3,
        padding="same",
        activation="relu",
        dropout_rate=0.0,
        **kwargs,
    ):
        super(Conv1DBlock, self).__init__(**kwargs)

        # Define sub-layers
        self.conv = Conv1D(filters, kernel_size, padding=padding, activation=activation)
        self.pool = MaxPool1D()
        self.bn = BatchNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return x


class LSTMBlock(tf.keras.layers.Layer):
    def __init__(self, units=300, dropout_rate=0.0, **kwargs):
        super(LSTMBlock, self).__init__(**kwargs)

        self.lstm = Bidirectional(LSTM(units, return_sequences=True))
        self.pool = MaxPool1D()
        self.bn = BatchNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.lstm(inputs)
        x = self.pool(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return x


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=12, key_dim=32, dropout_rate=0.0, **kwargs):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)

        self.attention = MultiHeadAttention(num_heads, key_dim)
        self.ln = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.attention(query=inputs, value=inputs, key=inputs)
        x = self.ln(x)
        x = self.dropout(x, training=training)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units=256, activation="relu", dropout_rate=0.5, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        # Define sub-layers
        self.dense = Dense(units, activation=activation)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.dropout(x, training=training)
        return x
    

class Conv2DBlock(Model):
    def __init__(self, dropout_threshold=0.5):
        super(Conv2DBlock, self).__init__()
        self.conv2d_1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')
        self.max_pool = MaxPooling2D(pool_size=(2,2))
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(dropout_threshold)
        
    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.max_pool(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class ReduceSumLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

# Define the custom Attention Layer as a Model
class AttentionWeightBlock(Model):
    def __init__(self, dropout_rate=0.2, **kwargs):
        super(AttentionWeightBlock, self).__init__(**kwargs)
        self.attention_weights = Dense(1, activation='sigmoid')
        self.attention_pool = ReduceSumLayer(axis=1)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        attention_output = self.attention_weights(inputs)  # Learnable attention weights
        attention_pooled = self.attention_pool(attention_output * inputs)  # Apply attention weights
        attention_pooled = self.dropout(attention_pooled)  # Apply dropout
        return attention_pooled