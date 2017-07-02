from model.cnn_model import CNNModel
from model.clstm_model import CLSTMModel
from data.data_loader import DataLoader
from model.trainer import *

BATCH_SIZE = 32
NUM_CLASSES = 2
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
REGULARIZATION_RATE = 0.01

embedding_dim = 100

logs_path = "/app/tmp/logs/26/"
data_root = "/app/data/datasets/amazon-fine-food-reviews/"

train_filename = "train_Reviews"
test_filename = "test_Reviews"
valid_filename = "valid_Reviews"

print("Loading data...")
train_data = DataLoader(data_root, train_filename, NUM_EPOCHS, BATCH_SIZE, "Text", "Score")
train_data.load_data()
valid_data = DataLoader(data_root, valid_filename, NUM_EPOCHS, BATCH_SIZE, "Text", "Score")
valid_data.load_data()

model = CNNModel(num_classes=NUM_CLASSES, embedding_dim=embedding_dim, 
    sequence_len=train_data.sequence_len)

sess = tf.Session()
x = tf.placeholder(tf.int32, [None, train_data.sequence_len], name="x")
y = tf.placeholder(tf.int32, [None, 1], name="y")
keep_prob = tf.placeholder(tf.float32)


oh_y = tf.squeeze(tf.one_hot(y, depth=NUM_CLASSES, name='oh_y'))

with tf.name_scope("embeddings"):
    init_width = 0.5 / embedding_dim
    embedding_matrix = tf.Variable(
        tf.random_uniform([train_data.vocab_len, embedding_dim], -init_width, init_width),
        name="embedding_matrix")
    emb = tf.nn.embedding_lookup(embedding_matrix, x)
    emb = tf.expand_dims(emb, -1)

with tf.name_scope(name='model'):
    # logits = model.predict(emb, keep_prob)
    logits = model.predict(emb)
with tf.name_scope(name='prediction'):
    pred = tf.reshape(tf.cast(tf.argmax(logits, axis=1), tf.int32), shape=[-1, 1])
    

with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=oh_y, logits=logits))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = loss + REGULARIZATION_RATE * tf.reduce_sum(regularization_losses)

with tf.name_scope(name='optimizer'):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads_and_vars = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, 
        global_step=global_step)

with tf.name_scope(name='mertics'):
    TP = tf.count_nonzero(pred * y, name='TP')
    TN = tf.count_nonzero((pred - 1) * (y - 1), name='TN')
    FP = tf.count_nonzero(pred * (y - 1), name='FP')
    FN = tf.count_nonzero((pred - 1) * y, name='FN')

with tf.name_scope('accuracy'):
    accuracy = (TP + TN) / (TP + FP + FN + TN)

with tf.name_scope('precision'):
    precision = TP / (TP + FP)

with tf.name_scope('recall'):
    recall = TP / (TP + FN)

with tf.name_scope('F1'):
    f1_score = 2 * (precision * recall) / (precision + recall)

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", total_loss)

# Create a summary to monitor TP, TN, FP, FN
tf.summary.scalar("TP", TP)
tf.summary.scalar("TN", TN)
tf.summary.scalar("FP", FP)
tf.summary.scalar("FN", FN)

# Create a summary to monitor the accuracy and F1 score
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("f1_score", f1_score)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

# Create summaries for all gradients
for grad, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', grad)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

model_train(sess=sess, x=x, y=y, keep_prob=keep_prob, train_op=train_op, loss=total_loss, accuracy=accuracy, 
    f1_score=f1_score, tp=TP, tn=TN,fp=FP, fn=FN, global_step=global_step, 
    embedding_matrix=embedding_matrix, merged_summary_op=merged_summary_op, logs_path=logs_path,
    log_interval=100, num_epochs=NUM_EPOCHS, data=train_data, valid_data=valid_data)
