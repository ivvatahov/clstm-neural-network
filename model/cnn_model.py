import numpy as np 
import tensorflow as tf
from model.base_model import BaseModel


class CNNModel(BaseModel):

    def __init__(self, num_classes, embedding_dim, sequence_len, 
                 filter_num=16, filter_size=5):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.filter_num = filter_num
        self.filter_size = filter_size

    def activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '_activations', x)

    def predict(self, x):  
        with tf.variable_scope('conv1') as scope:
            filter_shape = [self.filter_size, self.embedding_dim, 1, self.filter_num]
        
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                name='weights')
            b = tf.Variable(tf.zeros([self.filter_num]), name='biases')
            
            conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')
            pre_activation = tf.nn.bias_add(conv, b)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            self.activation_summary(conv1)

        # pool1
        pool1 = tf.nn.max_pool(conv1, 
                               ksize=[1, self.sequence_len - self.filter_size + 1, 1, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID',
                               name="pool1")
      
        with tf.name_scope("output"):
            shape = pool1.get_shape().as_list()
            W = tf.Variable(tf.truncated_normal([shape[2] * shape[3], self.num_classes], 
                stddev=0.1),
                name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="biases")

            shape = pool1.get_shape().as_list()
            # print(shape)
            reshape = tf.reshape(pool1, [-1, shape[2] * shape[3]])
            # print(reshape.get_shape())
            out = tf.add(tf.matmul(reshape, W), b, name=scope.name)
            self.activation_summary(out)

        return out
