from model.base_model import BaseModel
import numpy as np 
import tensorflow as tf

class CLSTMModel(BaseModel):

    def __init__(self, num_classes, embedding_dim, sequence_len, 
                 num_units=16, filter_num=16, filter_size=3, is_training=True):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.num_units = num_units
        self.filter_num = filter_num
        self.filter_size = filter_size

    def __activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '_activations', x)

    def __conv2d(self, x, filter_r, filter_c, filter_num, name="conv"):
        with tf.variable_scope(name) as scope:
            filter_shape = [filter_r, filter_c, 1, filter_num]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='weights')
            b = tf.Variable(tf.zeros([filter_num]), name='biases')

            conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, b)
            # Batch Normalization
            conv = tf.contrib.layers.batch_norm(conv,
                                          center=True, scale=True,
                                          is_training=self.is_training)
            conv = tf.nn.relu(conv, name=scope.name)
            self.__activation_summary(conv)
            return conv

    def __dropout(self, x, keep_prob, name="dropout"):
    	with tf.variable_scope(name) as scope:
        	drop = tf.nn.dropout(x, keep_prob, name=scope.name)
        	return drop

    def __lstm(self, x, num_units, name):
        with tf.variable_scope(name) as scope:
            lstm = tf.contrib.rnn.LSTMCell(self.num_units)
            out, state = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
            self.__activation_summary(out)
            return out, state

    def __fc(self, x, h_dim, name):
        with tf.name_scope(name) as scope:
            W = tf.Variable(tf.truncated_normal([x.get_shape().as_list()[1], h_dim], 
                stddev=0.1),
                name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[h_dim]), name="biases")

            out = tf.add(tf.matmul(x, W), b)
            self.__activation_summary(out)
            return out

    def predict(self, x, keep_prob):

        # Conv Layer
        conv = self.__conv2d(x, self.filter_size, self.embedding_dim, self.filter_num, "conv")

        # Remove the empty 2nd dimension
        conv_out = tf.squeeze(conv, 2)

        # Dropout
        dropout = self.__dropout(conv_out, keep_prob)
        
        # LSTMs
        lstm_out1, _ = self.__lstm(dropout, self.num_units, "lstm1")
        lstm_out, _ = self.__lstm(lstm_out1, self.num_units, "lstm2")
        
        # Reshape the tensor to fit in fc layer
        shape = lstm_out.get_shape().as_list()
        squeeze_len = shape[1] * shape[2]
        lstm_out_reshape = tf.reshape(lstm_out,[-1, squeeze_len])

        # FC layer
        out = self.__fc(lstm_out_reshape, self.num_classes, "output")

        # Dropout
        out = self.__dropout(out, keep_prob)
        
        return out