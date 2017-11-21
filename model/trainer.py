import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

from config import Config


def model_train(sess, model, ph, ops, metrics, train_data, valid_data, vocabulary):
    """
    """

    embedding_matrix = ops['embedding_matrix']

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

    sess.run(tf.global_variables_initializer())

    print("Start training...")
    iterator = train_data.make_one_shot_iterator()
    next_batch = iterator.get_next()

    # TODO: change it
    total_batch = 100
    for ep in range(Config.NUM_EPOCHS):
        for i in range(total_batch):

            # get next batch of data
            x_batch, y_batch = sess.run(next_batch)

            feed_dict = {
                ph['x']: x_batch,
                ph['y']: y_batch,
                ph['lengths']: np.repeat(train_data.sequence_len, y_batch.shape[0]),
                ph['keep_prob']: Config.DROPOUT
            }

            model.is_training = True
            _, cost, acc, f1_train = sess.run([ops['train_op'],
                                               ops['loss'],
                                               metrics.accuracy,
                                               metrics.f1_score],
                                              feed_dict)

            # current_step = tf.train.global_step(sess, global_step)

            if i % train_data.total_batch - 1 == 0:
                saver.save(sess, os.path.join(Config.LOGS_PATH, "model.ckpt"), i)

            if i % Config.LOG_INTERVAL == 0:
                model.is_training = False

                feed_dict = {
                    ph['x']: valid_data.source[:2048],
                    ph['y']: valid_data.labels[:2048],
                    ph['lengths']: np.repeat(train_data.sequence_len, 2048),
                    ph['keep_prob']: 1
                }

                val_acc, TP, TN, FP, FN, f1_valid, summary = sess.run([
                    metrics.accuracy, metrics.tp, metrics.tn, metrics.fp,
                    metrics.fn, metrics.f1_score, ops['merged_summary_op']],
                    feed_dict=feed_dict)

                # Write logs at every iteration
                summary_writer.add_summary(summary, ep * train_data.total_batch + i)
                projector.visualize_embeddings(summary_writer, projector_config)

                print("TN:", TN, "FP:", FP)
                print("FN:", FN, "TP:", TP)
                print("epoch {}, "
                      "step {}/{}, "
                      "train_loss {:g}, "
                      "train_acc {:g}, "
                      "train_f1_score {:g},"
                      "valid_acc {:g}, "
                      "valid_f1_score {:g}".format(ep, i, train_data.total_batch,
                                                   cost, acc, f1_train, val_acc, f1_valid))
    summary_writer.close()
