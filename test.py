import sys
import tensorflow as tf
import numpy as np
from config import Config
from data.data_loader import DataLoader
from model.clstm_model import CLSTMModel

data = DataLoader(Config.DATA_ROOT, Config.TRAIN_FILENAME)
vocab = data.load_vocab()
vocab_len = len(vocab)
vocab_dict = {}

for i in range(len(vocab)):
    vocab_dict[vocab.at[i, 0]] = i

model = CLSTMModel()
model.is_training = False

# ------------ placeholders --------------

x = tf.placeholder(tf.int32, [None, Config.MAX_SEQUENCE_LENGTH], name="x")
lengths = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

# ----------------------------------------

# ----------------model-------------------

with tf.name_scope("embeddings"):
    embedding_matrix = tf.Variable(
        tf.random_uniform([vocab_len, Config.EMBEDDING_DIM], -1.0, 1.0),
        name="embedding_matrix")
    emb = tf.nn.embedding_lookup(embedding_matrix, x)
    emb = tf.expand_dims(emb, -1)

with tf.name_scope(name='model'):
    logits = model.predict(emb, lengths, keep_prob)
    logits = tf.nn.softmax(logits)

with tf.name_scope(name='prediction'):
    pred = tf.reshape(tf.cast(tf.argmax(logits, axis=1), tf.int32),
                      shape=[-1, 1],
                      name="pred")

# ----------------------------------------
saver = tf.train.Saver()

with tf.Session() as sess:
    checkpoint_path = tf.train.latest_checkpoint(Config.CHECKPOINT_DIR)
    print(checkpoint_path)
    saver.restore(sess, checkpoint_path)

    tf.logging.info("Restoring full model from checkpoint file %s", checkpoint_path)

    print("Enter text: \n>", end='', flush=True)

    line = sys.stdin.readline()

    # tokenize
    example = line.split()
    sources = np.zeros((1, Config.MAX_SEQUENCE_LENGTH))

    # lookup
    for i in range(len(example)):
        if example[i] in vocab_dict:
            sources[0][i] = vocab_dict[example[i]]

    feed_dict = {
        x: sources,
        lengths: np.repeat(sources.shape[1], sources.shape[0]),
        keep_prob: 1
    }

    out_score = sess.run(logits, feed_dict=feed_dict)

    print('> Sentiment score: ', out_score, "\n", end='\n>', flush=True)
