import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

from tensorflow.python.data import Iterator

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

    iterator = Iterator.from_structure(train_data.output_types,
                                       train_data.output_shapes)

    next_batch = iterator.get_next()

    training_init_op = iterator.make_initializer(train_data)
    validation_init_op = iterator.make_initializer(valid_data)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.tables_initializer())

    sess.run(init_op)
    print("Start training...")
    for ep in range(Config.NUM_EPOCHS):
        i = 0
        while True:
            try:
                sess.run(training_init_op)
                x_batch, y_batch = sess.run(next_batch)

                feed_dict = {
                    ph['x']: x_batch,
                    ph['y']: y_batch,
                    ph['lengths']: np.repeat(x_batch.shape[1], y_batch.shape[0]),
                    ph['keep_prob']: Config.DROPOUT
                }

                model.is_training = True
                _, cost, acc, f1_train = sess.run([ops['train_op'],
                                                   ops['loss'],
                                                   metrics.accuracy,
                                                   metrics.f1_score],
                                                  feed_dict)
                # LOG
                if i % Config.LOG_INTERVAL == 0:
                    sess.run(validation_init_op)
                    model.is_training = False
                    VA = F1 = 0.0
                    TP = TN = FP = FN = 0
                    summary = None
                    bn = 10
                    for j in range(bn):
                        try:
                            x_valid, y_valid = sess.run(next_batch)

                            feed_dict = {
                                ph['x']: x_valid,
                                ph['y']: y_valid,
                                ph['lengths']: np.repeat(x_valid.shape[1], y_valid.shape[0]),
                                ph['keep_prob']: 1
                            }

                            va, tp, tn, fp, fn, f1, summary = sess.run([
                                metrics.accuracy, metrics.tp, metrics.tn, metrics.fp,
                                metrics.fn, metrics.f1_score, ops['merged_summary_op']],
                                feed_dict=feed_dict)

                            VA += va
                            TP += tp
                            TN += tn
                            FP += fp
                            FN += fn
                            F1 += f1
                        except tf.errors.OutOfRangeError:
                            print("End of valid dataset")
                            break
                    # Write logs at every iteration
                    summary_writer.add_summary(summary, ep * j + i)
                    projector.visualize_embeddings(summary_writer, projector_config)

                    print("TN:", TN, "FP:", FP)
                    print("FN:", FN, "TP:", TP)
                    print("epoch {}, "
                          "step {}/{}, "
                          "train_loss {:g}, "
                          "train_acc {:g}, "
                          "train_f1_score {:g},"
                          "valid_acc {:g}, "
                          "valid_f1_score {:g}".format(ep, i, j,
                                                       cost, acc, f1_train, VA / bn, F1 / bn))
                i += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break

            # Save
            saver.save(sess, os.path.join(Config.LOGS_PATH, "model.ckpt"), i)

        summary_writer.close()
