import numpy as np
import pandas as pd
from six.moves import cPickle as pickle

class DataLoader(object):

    DEFAULT_VOCABULARY_SIZE = 20000

    def __init__(self, data_root, filename, num_epochs, batch_size, 
                 data_column, labels_column):

        self.__data_root = data_root
        self.__filename = filename
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.data_column = data_column
        self.labels_column = labels_column

        self.current_batch = 0


    def load_data(self):
        self.source, self.labels = self.__read_file(self.__data_root, self.__filename)
        self.data_len = self.labels.shape[0]
        self.sequence_len = self.source.shape[1]
        self.vocabulary = pd.read_csv(self.__data_root + "vocabulary_Reviews", header=None)
        self.vocab_len = len(self.vocabulary)
        self.total_batch = int((self.data_len - 1) / self.batch_size) + 1
        
    def next_batch(self, shuffle=True):
        start = self.current_batch * self.batch_size
        end = min((self.current_batch + 1) * self.batch_size, self.data_len)
        self.current_batch = (self.current_batch + 1) % self.total_batch
       
        if shuffle and self.current_batch == 1:
            print("baba")
            shuffle_idxs = np.random.permutation(self.data_len)
            self.source = self.source[shuffle_idxs]
            self.labels = self.labels[shuffle_idxs]

        return self.source[start:end], self.labels[start:end]

    def __read_file(self, path, filename):
        with open(path + filename, 'rb') as f:
            save = pickle.load(f)
            source = save['source']
            labels = save['labels']
        return source, labels