import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

def model_train(model, ph, ops, metrics, config, train_data, valid_data):
    """
    """

    sess = tf.Session()

    saver = tf.train.Saver()

    # Embeddings
    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = ops['embedding_matrix'].name
    embedding.metadata_path = os.path.join(config.LOGS_PATH, 'metadata.tsv')

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(config.LOGS_PATH, sess.graph)

    with open(config.LOGS_PATH + 'metadata.tsv', 'w', encoding='utf8') as f:
        for word in train_data.vocabulary.values:
            f.write(str(word[0]) + '\n')

    sess.run(tf.global_variables_initializer())

    print("Start training...")
    for ep in range(config.num_epochs):
        for i in range(train_data.total_batch):
            x_batch, y_batch = train_data.next_batch()

            feed_dict = {
                ph['x']: x_batch,
                ph['y']: y_batch,
                ph['lengths']: np.repeat(train_data.sequence_len, y_batch.shape[0]),
                ph['keep_prob']: 1.0 - config.dropout
            }

            model.is_training = True
            _, cost, acc, f1_train = sess.run([ops['train_op'],
                                               ops['loss'],
                                               metrics.accuracy,
                                               metrics.f1_score],
                                              feed_dict)

            if i % train_data.total_batch - 1 == 0:
                saver.save(sess, os.path.join(config.MODEL_CHECKPOINTS, "model.ckpt"), i)

            if i % config.log_interval == 0:
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
