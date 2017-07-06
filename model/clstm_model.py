from model.base_model import BaseModel
import numpy as np 
import tensorflow as tf

class CLSTMModel(BaseModel):

    def __init__(self, num_classes, embedding_dim, sequence_len, 
                 num_units=32, filter_num=64, filter_size=3, is_training=True):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.num_units = num_units
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.is_training = is_training

    def __conv2d(self, x, filter_r, filter_c, filter_num, use_l2_loss=True, name="conv"):
        if use_l2_loss:
            regularizer = tf.contrib.layers.l2_regularizer(0.001)
        else:
            regularizer = None

        with tf.variable_scope(name) as scope:
            filter_shape = [filter_r, filter_c, 1, filter_num]
            W = tf.get_variable('weights' ,
                initializer=tf.truncated_normal(filter_shape, stddev=0.35),
                regularizer=regularizer)
            b = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[filter_num]))
            
            conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID', name="baba")
            conv = tf.nn.bias_add(conv, b)
            conv = tf.contrib.layers.batch_norm(conv,
                                          center=True, scale=True,
                                          is_training=self.is_training)
            conv = tf.nn.relu(conv, name="output")
            self._activations_summary(conv)
            return conv

    def __dropout(self, x, keep_prob, name="dropout"):
    	with tf.variable_scope(name) as scope:
        	drop = tf.nn.dropout(x, keep_prob, name=scope.name)
        	return drop

    def __lstm(self, x, num_units, name="lstm"):
        with tf.variable_scope(name) as scope:
            lstm = tf.contrib.rnn.LSTMCell(self.num_units)
            out, state = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
            self._activations_summary(out)
            return out, state

    def __fc(self, x, h_dim, use_l2_loss=True, name="fc"):
        if use_l2_loss:
            regularizer = tf.contrib.layers.l2_regularizer(0.001)
        else:
            regularizer = None

        with tf.variable_scope(name) as scope:
            W = tf.get_variable('weights', 
                initializer=tf.truncated_normal([x.get_shape().as_list()[1], h_dim], stddev=0.35),
                regularizer=regularizer)
            b = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[h_dim]))

            out = tf.add(tf.matmul(x, W), b)
            self._activations_summary(out)
            return out

    def __batch_norm(self, x):
        norm = tf.contrib.layers.batch_norm(x, 
            center=True, scale=True,
            is_training=self.is_training)
        return norm

    def predict(self, x, lengths, keep_prob):

        # Conv Layer
        conv = self.__conv2d(x, self.filter_size, self.embedding_dim, 
            self.filter_num, name="layer1_conv")

        # Remove the empty 2nd dimension
        conv_out = tf.squeeze(conv, 2)

        # LSTMs
        lstm_out, _ = self.__lstm(conv_out, self.num_units, name="layer2_lstm")
        # lstm_out, _ = self.__lstm(lstm_out, self.num_units, "lstm2")

        # Get the last output from lstm_out
        batch_size = tf.shape(lstm_out)[0]
        batch_range = tf.range(batch_size)
        indices = tf.stack([batch_range, lengths - self.filter_size], axis=1)
        lstm_out = tf.gather_nd(lstm_out, indices)

        # normalize the lstm_out
        # lstm_out = out = self.__batch_norm(lstm_out)
        
        lstm_out = self.__dropout(lstm_out, keep_prob)

        # FC layer
        out = self.__fc(lstm_out, self.num_classes, name="layer3_fc")
         
        # normalize the output
        out = self.__batch_norm(out)

        out = lstm_out = self.__dropout(out, keep_prob) 

        return out