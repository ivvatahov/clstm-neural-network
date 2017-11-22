from collections import Counter

import nltk
import pandas as pd


class Preprocessor(object):
    TEST_PREFIX = 'test_'
    VOCABULARY_PREFIX = 'vocabulary_'
    TRAIN_PREFIX = 'train_'
    VALID_PREFIX = 'valid_'

    UNK_ID = 1
    MAX_DATA_LENGTH = 200

    _PAD = '<PAD>'
    _UNK = '<UNK>'
    _EOS = '<EOS>'

    def __init__(self, path, filename, vocabulary_size, train_size=0.6,
                 valid_size=.2, max_data_length=MAX_DATA_LENGTH, pad=_PAD,
                 unk=_UNK, eos=_EOS):
        self.path = path
        self.filename = filename
        self.vocabulary_size = vocabulary_size
        self.max_data_length = max_data_length
        self.train_size = train_size
        self.train_size = train_size
        self.valid_size = valid_size
        self.pad = pad
        self.unk = unk
        self.eos = eos

        self.separator = ','
        self._dictionary = {}
        self.tokenizer = nltk.TweetTokenizer()

        self.data = None
        self.new_data = None
        self.data_column = None
        self.label_column = None

    def read_data(self):
        self.data = pd.read_csv(self.path + self.filename + ".csv")
        return self.data

    def _build_dictionary(self, data, data_column):
        all_text = []

        for sentence in data[data_column]:
            all_text.extend(self.tokenizer.tokenize(sentence))

        all_words = [(self.pad, -1), (self.unk, -1), (self.eos, -1)]
        all_words.extend(Counter(all_text).most_common(self.vocabulary_size - 3))

        for word in all_words:
            if word[0] not in self._dictionary:
                self._dictionary[word[0]] = len(self._dictionary)
        self.vocabulary_size = len(self._dictionary)

        print("Saving vocabulary...")
        word_column = 'Word'
        vocabulary = pd.DataFrame(data=all_words, columns=[word_column, 'Frequency'])
        vocabulary.to_csv(self.path + self.VOCABULARY_PREFIX + "frequency_" + self.filename, sep=self.separator,
                          index=False,
                          encoding='utf-8')
        vocabulary[word_column].to_csv(self.path + self.VOCABULARY_PREFIX + self.filename, sep=self.separator,
                                       index=False,
                                       encoding='utf-8')
        return self._dictionary

    def preprocess(self, data_column, label_column):

        self.data_column = data_column
        self.label_column = label_column

        new_data = self.data[[data_column, label_column]].copy()
        new_data = new_data.loc[new_data[data_column].str.len() < self.max_data_length]

        print("Creating the dictionary...")
        self._build_dictionary(new_data, data_column)

        # print("Tokenize...")
        # new_data[data_column] = new_data[data_column].map(lambda x: self.tokenizer.tokenize(x))

        # TODO
        # new_data = new_data.loc[new_data[data_column].len() < self.max_data_length]

        # print("Replace the words with indexes...")
        # new_data[data_column] = new_data[data_column].map(
        #     lambda x: list(map(
        #         lambda y: self._dictionary[y] if y in self._dictionary else self.UNK_ID, x)))

        print("Convert labels...")
        new_data[label_column] = new_data[label_column].map(lambda x: 1 if x > 3 else 0)

        # print("Shuffle the data")
        # new_data = new_data.iloc[np.random.permutation(len(new_data))]

        # self.max_seq_len = new_data[data_column].map(len).max()
        self.new_data = new_data

    def save_data(self):
        train, valid, test = self.__train_validate_test_split(self.new_data)

        self._save_data(train, self.path, self.TRAIN_PREFIX + self.filename)
        self._save_data(valid, self.path, self.VALID_PREFIX + self.filename)
        self._save_data(test, self.path, self.TEST_PREFIX + self.filename)

    def _save_data(self, dataset, path, filename):
        dataset.to_csv(path + filename + ".csv", sep=self.separator, index=False, encoding='utf-8')

    def __train_validate_test_split(self, data):
        size = len(data)
        train_end = int(self.train_size * size)
        valid_end = int(self.valid_size * size) + train_end
        train = data[:train_end]
        valid = data[train_end:valid_end]
        test = data[valid_end:]
        return train, valid, test
