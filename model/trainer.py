import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.data import Iterator

from config import Config


def model_train(sess, model, ph, ops, metrics, train_data, valid_data, vocabulary):
    """
    """

    embedding_matrix = ops['embedding_matrix']
    x, y = ph['x'], ph['y']
    lengths = ph['lengths']
    keep_prob = ph['keep_prob']

    train_op = ops['train_op']
    loss = ops['loss']
    merged_summary_op = ops['merged_summary_op']
    global_step = ops['global_step']

    # Create a saver object
    saver = tf.train.Saver()

    # Embeddings
    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = embedding_matrix.name
    embedding.metadata_path = os.path.join(Config.LOGS_PATH, 'metadata.tsv')

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(Config.LOGS_PATH, sess.graph)

    with open(Config.LOGS_PATH + 'metadata.tsv', 'w', encoding='utf8') as f:
        for word in vocabulary.values:
            f.write(str(word[0]) + '\n')

    iterator = Iterator.from_structure(train_data.output_types,
                                       train_data.output_shapes)
    next_batch = iterator.get_next()

    training_init_op = iterator.make_initializer(train_data)
    validation_init_op = iterator.make_initializer(valid_data)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.tables_initializer())

    print("Start training...")
    sess.run(init_op)
    for ep in range(Config.NUM_EPOCHS):
        sess.run(training_init_op)

        while True:
            try:
                x_batch, y_batch = sess.run(next_batch)

                feed_dict = {
                    x: x_batch,
                    y: y_batch,
                    lengths: np.repeat(x_batch.shape[1], y_batch.shape[0]),
                    keep_prob: Config.DROPOUT
                }

                model.is_training = True
                _, train_loss, train_acc, f1_train = sess.run([train_op, loss, metrics.accuracy, metrics.f1_score],
                                                              feed_dict=feed_dict)

                current_step = tf.train.global_step(sess, global_step)

                # log the summaries and the embeddings
                if (current_step + 1) % Config.LOG_INTERVAL == 0:
                    summary = sess.run(merged_summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, global_step=current_step)
                    projector.visualize_embeddings(summary_writer, projector_config)

                # create checkpoint
                if (current_step + 1) % Config.CHECKPOINT_INTERVAL == 0:
                    saver.save(sess, os.path.join(Config.CHECKPOINT_DIR, "model.ckpt"), global_step=current_step)

            except tf.errors.OutOfRangeError:
                break

        valid_acc = 0.0
        valid_f1 = 0.0
        TP = TN = FP = FN = 0

        sess.run(validation_init_op)
        bn = 0
        while True:
            try:
                x_valid, y_valid = sess.run(next_batch)
                bn += 1

                feed_dict = {
                    x: x_valid,
                    y: y_valid,
                    lengths: np.repeat(x_valid.shape[1], y_valid.shape[0]),
                    keep_prob: 1
                }
                model.is_training = False
                va, tp, tn, fp, fn, f1 = sess.run(
                    [metrics.accuracy, metrics.tp, metrics.tn, metrics.fp, metrics.fn, metrics.f1_score, ],
                    feed_dict=feed_dict)

                valid_acc += va
                TP += tp
                TN += tn
                FP += fp
                FN += fn
                valid_f1 += f1
            except tf.errors.OutOfRangeError:
                break

        print("TN:", TN, "FP:", FP)
        print("FN:", FN, "TP:", TP)
        print("epoch {}, "
              "valid_acc {:g}, "
              "valid_f1_score {:g}".format(ep, valid_acc / bn, valid_f1 / bn))

    summary_writer.close()
