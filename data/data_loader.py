import pandas as pd
import tensorflow as tf

from config import Config


class DataLoader(object):
    def __init__(self, data_root, filename):
        self.__data_root = data_root
        self.__filename = filename
        self.record_defaults = [[''], [1]]

        self.vocab_len = 0
        self.dataset = None
        self.vocabulary = None
        self.total_batch = None

    def _read_row(self, csv_row):
        example, label = tf.decode_csv(csv_row, record_defaults=self.record_defaults)
        return example, label

    def _tokenize_examples(self, example, label):
        print(example.get_shape())
        example = tf.string_split([example], " ")
        return example, label

    def _convert_to_indexes(self, example, labels):
        if self.table is None:
            self.table = tf.contrib.lookup.index_table_from_tensor(self.vocabulary)
        example = tf.sparse_tensor_to_dense(example, default_value="")
        ids = self.table.lookup(example)
        return ids, labels

    def load_data(self):
        self._read_vocab()
        dataset = self._read_file()
        return dataset.map(self._read_row) \
            .map(self._tokenize_examples) \
            .map(self._convert_to_indexes) \
            .repeat(Config.NUM_EPOCHS) \
            .padded_batch(Config.BATCH_SIZE,
                          padded_shapes=([None, Config.MAX_SEQUENCE_LENGTH], [None]))

    def _read_file(self):
        files = [self.__data_root + self.__filename + ".csv"]
        self.dataset = tf.data.TextLineDataset(files)
        return self.dataset

    def _read_vocab(self):
        self.vocabulary = pd.read_csv(self.__data_root + "vocabulary_Reviews", header=None)
        return self.vocabulary
