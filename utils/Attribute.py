import tensorflow as tf
class CnnAtribute:
    def __init__(
        self,
        filter_size,
        kernel_size,
        padding="valid",
        activation="relu",
        dropout_rate=0.0,
    ):
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate

#filters=32, kernel_size=(3, 3), activation="relu", padding="same"

class Cnn2DAttribute:
    def __init__(
        self,
        filter_size,
        kernel_size,
        padding="valid",
        activation="relu",
        dropout_rate=0.0,
    ):
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate


class LSTMAttribute:
    def __init__(self, units=300, dropout_rate=0.0):
        self.units = units
        self.dropout_rate = dropout_rate


class DenseAttribute:
    def __init__(self, units=256, dropout_rate=0.0, activation="relu"):
        self.units = units
        self.dropout_rate = dropout_rate
        self.activation = activation


class MultiHeadAttentionAttribute:
    def __init__(self, num_heads, key_dim, dropout_rate=0.0):
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps):
        super(WarmUp, self).__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=0.96
        )(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)

