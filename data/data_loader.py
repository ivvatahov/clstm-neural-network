import pandas as pd
import tensorflow as tf

from config import Config


class DataLoader(object):

    def __init__(self, data_root, filename):
        self.__data_root = data_root
        self.__filename = filename

        self.vocab_len = 0
        self.dataset = None
        self.vocabulary = None
        self.total_batch = None

    def load_data(self):
        files = [self.__data_root + self.__filename + ".csv"]
        dataset = tf.data.TextLineDataset(files)

        dataset = dataset.map(self._read_row)
        dataset = dataset.repeat(Config.NUM_EPOCHS)
        # dataset = dataset.padded_batch(Config.BATCH_SIZE,
        #                                padded_shapes=([None, Config.MAX_SEQUENCE_LENGTH], [None]))
        # TODO: read CSV
        self.dataset = dataset
        return self.dataset

    def load_vocab(self):
        self.vocabulary = pd.read_csv(self.__data_root + "vocabulary_Reviews", header=None)
        return self.vocabulary

    @staticmethod
    def _read_row(csv_row):
        record_defaults = [[''], [0]]
        row = tf.decode_csv(csv_row, record_defaults=record_defaults)
        out = row[:-1], row[-1]
        return out
