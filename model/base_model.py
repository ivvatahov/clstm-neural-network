import numpy as np 
import tensorflow as tf

embedding_dim = 100
filter_size = 3
filters_num = 16


class BaseModel:

    def _activations_summary(self, x):
    	tf.summary.histogram(x.name + "/activations", x)