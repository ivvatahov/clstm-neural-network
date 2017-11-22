from data.data_loader import DataLoader
from model.clstm_model import CLSTMModel
from model.metrics.metrics import Metrics
from model.trainer import *

print("Loading data...")
train_data_loader = DataLoader(Config.DATA_ROOT, Config.TRAIN_FILENAME)
valid_data_loader = DataLoader(Config.DATA_ROOT, Config.VALID_FILENAME)

train_data = train_data_loader.load_data()
valid_data = valid_data_loader.load_data()

vocab = train_data_loader.read_vocab()
vocab_len = len(vocab)

model = CLSTMModel()

# TODO: add buckets

# placeholders
x = tf.placeholder(tf.int32, [None, Config.MAX_SEQUENCE_LENGTH], name="x")
y = tf.placeholder(tf.int32, [None, 1], name="y")
lengths = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

oh_y = tf.squeeze(tf.one_hot(y, depth=Config.NUM_CLASSES, name='oh_y'))

with tf.name_scope("embeddings"):
    embedding_matrix = tf.Variable(
        tf.random_uniform([vocab_len, Config.EMBEDDING_DIM], -1.0, 1.0),
        name="embedding_matrix")
    emb = tf.nn.embedding_lookup(embedding_matrix, x)
    emb = tf.expand_dims(emb, -1)
    # emb = tf.nn.dropout(emb, ph['keep_prob'])

with tf.name_scope(name='model'):
    logits = model.predict(emb, lengths, keep_prob)

with tf.name_scope(name='prediction'):
    pred = tf.reshape(tf.cast(tf.argmax(logits, axis=1), tf.int32), shape=[-1, 1])

with tf.name_scope('loss'):
    pos_weight = tf.constant([1.0, 1.0])
    loss = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            targets=oh_y,
            logits=logits,
            pos_weight=pos_weight))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = loss + tf.add_n(regularization_losses)

with tf.name_scope(name='optimizer'):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(Config.LEARNING_RATE)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars,
                                         global_step=global_step)

# metrics
metrics = Metrics(pred, y)

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)

# Create a summary to monitor TP, TN, FP, FN
tf.summary.scalar("TP", metrics.tp)
tf.summary.scalar("TN", metrics.tn)
tf.summary.scalar("FP", metrics.fp)
tf.summary.scalar("FN", metrics.fn)

# Create a summary to monitor the accuracy and F1 score
tf.summary.scalar("accuracy", metrics.accuracy)
tf.summary.scalar("f1_score", metrics.f1_score)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

# Create summaries for all gradients
for grad, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', grad)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

ph = {
    'x': x,
    'y': y,
    'lengths': lengths,
    'keep_prob': keep_prob
}

ops = {
    'embedding_matrix': embedding_matrix,
    'merged_summary_op': merged_summary_op,
    'train_op': train_op,
    'loss': loss
}

sess = tf.Session()

model_train(sess=sess,
            model=model,
            ph=ph,
            ops=ops,
            metrics=metrics,
            train_data=train_data,
            valid_data=valid_data,
            vocabulary=vocab)
