import pandas as pd
import tensorflow as tf

from config import Config
from data.preprocessor import Preprocessor


class DataLoader(object):
    def __init__(self, data_root, filename):
        self.__data_root = data_root
        self.__filename = filename
        self.record_defaults = [[" "], [0]]
        self.padded_shapes = ((Config.MAX_SEQUENCE_LENGTH,), (1,))

        self.vocab_len = 0
        self.dataset = None
        self.vocabulary = None
        self.total_batch = None
        self.table = None

    def _read_row(self, csv_row, field_delimiter=","):
        example, label = tf.decode_csv(csv_row, self.record_defaults, field_delimiter)
        label = tf.reshape(label, [1])
        return example, label

    def _tokenize_examples(self, example, label):
        example = tf.string_split([example], " ").values
        return example, label

    def _convert_to_indexes(self, example, labels):
        ids = self.table.lookup(example)
        return ids, labels

    def _read_file(self):
        files = [self.__data_root + self.__filename + ".csv"]
        self.dataset = tf.data.TextLineDataset(files)
        return self.dataset

    def load_data(self):
        self.table = tf.contrib.lookup.index_table_from_file(self.__data_root + "vocabulary_Reviews",
                                                             default_value=Preprocessor.UNK_ID)
        dataset = self._read_file()
        return dataset \
            .skip(1) \
            .map(self._read_row) \
            .map(self._tokenize_examples) \
            .filter(lambda x, y: tf.size(x) < Config.MAX_SEQUENCE_LENGTH) \
            .map(self._convert_to_indexes) \
            .repeat(Config.NUM_EPOCHS) \
            .padded_batch(Config.BATCH_SIZE,
                          padded_shapes=self.padded_shapes)

    def read_vocab(self):
        self.vocabulary = pd.read_csv(self.__data_root + "vocabulary_Reviews", header=None)
        return self.vocabulary
