import pandas as pd
import tensorflow as tf


class DataLoader(object):
    DEFAULT_VOCABULARY_SIZE = 20000

    def __init__(self, data_root, filename, data_column, labels_column, config):
        self.__data_root = data_root
        self.__filename = filename

        self.config = config
        self.data_column = data_column
        self.labels_column = labels_column

        self.vocab_len = 0
        self.dataset = None
        self.vocabulary = None
        self.total_batch = None

    def load_data(self):
        self.dataset = tf.data.TextLineDataset(self.__data_root + self.__filename)
        self.dataset = self.dataset.batch(self.config.batch_size)

        # self.source, self.labels = self.__read_file(self.__data_root, self.__filename)
        # self.sequence_len = self.source.shape[1]

    def load_vocab(self):
        self.vocabulary = pd.read_csv(self.__data_root + "vocabulary_Reviews", header=None)
        self.vocab_len = len(self.vocabulary)
