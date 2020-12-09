import numpy as np
import tensorflow as tf

class Historic(tf.keras.Model):
    def __init__(self):
        """
        The Model class predicts future stock market prices given historic data
        """
        super(Historic, self).__init__()
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size = 50
        self.window_size = 64
        self.num_epochs = 128
        self.rnn_size = 128
        
        self.lstm1 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.D1 = tf.keras.layers.Dense(64, activation="relu")
        self.D2 = tf.keras.layers.Dense(32, activation="relu")
        self.D3 = tf.keras.layers.Dense(1)
        


    def call(self, inputs, initial_state=None):
        """
        Runs the model on inputs where inputs is a tensor and predicts the prices
        given the labels

        :param inputs: Stock data as tensor (batch_size, window_size, data_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch predictions as a tensor  of size(batch_size, window_size, 1), 
            final state of the LSTM which is list [state_h, state_c]
        """
        
        # No clue if this actually works
        layer1_out, state_h1, state_c1 = self.lstm1(inputs, initial_state=initial_state)
        # layer2_out = self.D1(layer1_out)
        layer2_out = self.D1(layer1_out)
        layer3_out = self.D2(layer2_out)
        layer4_out = self.D3(layer3_out)
        return layer4_out, (state_h1, state_c1)

    def loss(self, outputs, labels):
        """
        Calculates average loss across the batch. Uses MAPE so
        not biased towards "cheap" stocks

        :param outputs: a matrix of shape (batch_size, window_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.keras.losses.MAPE(labels, outputs)
        
        loss = tf.reduce_mean(loss)
        return loss
