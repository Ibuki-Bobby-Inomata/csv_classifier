import tensorflow as tf
import numpy as np


class WavenetResidualLayer(tf.keras.layers.Layer):
    def __init__(self, times):
        super(WavenetResidualLayer, self).__init__()
        self.dilated_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=2, dilation_rate=2 ** (times - 1),
                                                   padding='causal')
        self.conv1d1x1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1)
        self.tanh = tf.keras.activations.tanh
        self.sigmoid = tf.keras.activations.sigmoid

    def call(self, input_tensor):
        x = self.dilated_conv(input_tensor)
        tanh = self.tanh(x)
        sigmoid = self.sigmoid(x)
        z = tanh * sigmoid
        x_1 = self.conv1d1x1(z)
        return x * x_1, z


class WaveNetModule(tf.keras.models.Model):
    def __init__(self, input_shape):
        super(WaveNetModule, self).__init__()
        self.causal_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=3, input_shape=input_shape, padding='causal')

        self.waves = {}
        for i in range(1, 4):
            self.waves[i] = WavenetResidualLayer(i)

        self.reduce = tf.math.reduce_sum
        self.relu = tf.keras.layers.ReLU()
        self.conv1x1_0 = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding='same')
        self.conv1x1_1 = tf.keras.layers.Conv1D(filters=4, kernel_size=1, padding='same')
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.causal_conv(x)

        connections = []
        for i in range(1, 4):
            x, connection = self.waves[i](x)
            connections.append(connection)
        x = self.reduce(connections, axis=0)
        x = self.relu(x)
        x = self.conv1x1_0(x)
        x = self.relu(x)
        x = self.conv1x1_1(x)
        return x


class WaveNet(tf.keras.Model):
    """論文の図を見た限りのWaveNet"""

    def __init__(self, input_shape):
        super(WaveNet, self).__init__()
        self.wave_module = WaveNetModule(input_shape)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.wave_module(x)
        return self.softmax(x)


def WaveModel(train_x, train_y):
    i_s = np.array(train_x.shape)
    input_shape = (i_s[1], i_s[2])

    model = WaveNet(input_shape)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    model.fit(train_x, train_y, epochs=100,
              verbose=1, batch_size=128,
              validation_split=0.1
              )

    return model