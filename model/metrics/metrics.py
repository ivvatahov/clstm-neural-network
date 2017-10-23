import tensorflow as tf


class Metrics:
    def __init__(self, pred, y):
        self.pred = pred
        self.y = y

        self.__tp = None
        self.__tn = None
        self.__fp = None
        self.__fn = None
        self.__accuracy = None
        self.__precision = None
        self.__recall = None
        self.__f1_score = None

    @property
    def tp(self):
        with tf.name_scope(name='metrics'):
            if self.__tp is None:
                self.__tp = tf.count_nonzero(self.pred * self.y, name='TP')
            return self.__tp

    @property
    def tn(self):
        with tf.name_scope(name='metrics'):
            if self.__tn is None:
                self.__tn = tf.count_nonzero((self.pred - 1) * (self.y - 1), name='TN')
            return self.__tn

    @property
    def fp(self):
        with tf.name_scope(name='metrics'):
            if self.__fp is None:
                self.__fp = tf.count_nonzero(self.pred * (self.y - 1), name='FP')
            return self.__fp

    @property
    def fn(self):
        with tf.name_scope(name='metrics'):
            if self.__fn is None:
                self.__fn = tf.count_nonzero((self.pred - 1) * self.y, name='FN')
            return self.__fn

    @property
    def accuracy(self):
        with tf.name_scope(name='accuracy'):
            if self.__accuracy is None:
                self.__accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
            return self.__accuracy

    @property
    def precision(self):
        with tf.name_scope(name='precision'):
            if self.__precision is None:
                self.__precision = self.tp / (self.tp + self.fp)
            return self.__precision

    @property
    def recall(self):
        with tf.name_scope(name='recall'):
            if self.__recall is None:
                self.__recall = self.tp / (self.tp + self.fn)
            return self.__recall

    @property
    def f1_score(self):
        with tf.name_scope(name='f1_score'):
            if self.__f1_score is None:
                self.__f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
            return self.__f1_score
