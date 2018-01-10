import tensorflow as tf
import pandas as pd
from config import Config
from data.preprocessor import Preprocessor


class DataLoader(object):
    def __init__(self, data_root, filename):
        self.record_defaults = [[" "], [0]]
        self.padded_shapes = ((Config.MAX_SEQUENCE_LENGTH,), (1,))

        self.vocab_lookup = None
        self.__data_root = data_root
        self.__filename = filename

        self.dataset = None
        self.vocabulary = None

    def load_vocab(self):
        if self.vocabulary is None:
            self.vocabulary = pd.read_csv(self.__data_root + "vocabulary_Reviews", header=None)
        return self.vocabulary

    def load_data(self):
        self.vocab_lookup = tf.contrib.lookup.index_table_from_file(self.__data_root + "vocabulary_Reviews",
                                                                    default_value=Preprocessor.UNK_ID)
        return (self._read_data()
                .skip(1)
                .shuffle(buffer_size=10000)
                .map(self._parse_function, num_parallel_calls=8)
                .filter(lambda x, y: tf.size(x) <= Config.MAX_SEQUENCE_LENGTH)
                .map(self._convert_to_indexes, num_parallel_calls=8)
                # .repeat(Config.NUM_EPOCHS)
                .padded_batch(Config.BATCH_SIZE,
                              padded_shapes=self.padded_shapes)
                .prefetch(100 * Config.BATCH_SIZE))

    def _read_data(self):
        files = [self.__data_root + self.__filename + ".csv"]
        self.dataset = tf.data.TextLineDataset(files)
        return self.dataset

    def _parse_function(self, csv_row, field_delimiter=","):
        example, label = tf.decode_csv(csv_row, self.record_defaults, field_delimiter)
        label = tf.reshape(label, [1])

        # tokenize
        example = tf.string_split([example], " ").values

        return example, label

    def _convert_to_indexes(self, example, labels):
        ids = self.vocab_lookup.lookup(example)
        return ids, labels
