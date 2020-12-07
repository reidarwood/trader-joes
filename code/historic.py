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
        self.num_epochs = 250
        
        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
        self.D1 = tf.keras.layers.Dense(32, activation="relu")
        self.D2 = tf.keras.layers.Dense(32, activation="relu")
        self.D3 = tf.keras.layers.Dense(1,  activation="hard_sigmoid")
        


    def call(self, inputs, initial_state=None):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
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
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.keras.losses.MAPE(labels, outputs)
        
        loss = tf.reduce_mean(loss)
        return loss
