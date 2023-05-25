import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Layer
import keras
import tensorflow as tf

class Encoder(keras.layers.Layer):
    def __init__(self, h1, h2, time_steps, n_features, l2_alpha=0.02, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layer1 = LSTM(h1, activation='relu', input_shape=(time_steps, n_features), return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(l2_alpha))
        self.layer2 = LSTM(h2, activation='relu', return_sequences=False)

    def call(self, x):
        x = self.layer1(x)
        return self.layer2(x)

class Decoder(keras.layers.Layer):
    def __init__(self, h1, h2, time_steps, n_features, l2_alpha=0.02, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.layer1 = LSTM(h2, activation='relu', return_sequences=True)
        self.layer2 = LSTM(h1, activation='relu', return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(l2_alpha))
        self.time_distributed = TimeDistributed(Dense(n_features, kernel_regularizer=tf.keras.regularizers.L2(l2_alpha)))

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.time_distributed(x)

class AutoEncoder(tf.keras.Model):
    def __init__(self, h1, h2, time_steps, n_features, l2_alpha=0.02, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.h1 = h1
        self.h2 = h2
        self.time_steps = time_steps
        self.n_features = n_features
        self.encoder = Encoder(h1, h2, time_steps, n_features, l2_alpha)
        self.repeat = RepeatVector(time_steps)
        self.decoder = Decoder(h1, h2, time_steps, n_features, l2_alpha)

    def call(self, x):
        x = self.encoder(x)
        x = self.repeat(x)
        return self.decoder(x)
        