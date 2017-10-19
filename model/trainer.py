import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

dropout = 0.5


def model_train(model, x, y, lengths, keep_prob, sess, train_op, loss, accuracy, f1_score,
                tp, tn, fp, fn, embedding_matrix, merged_summary_op, logs_path,
                log_interval, num_epochs, data, valid_data):
    """
    """
    saver = tf.train.Saver()

    # Embeddings
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_matrix.name
    embedding.metadata_path = os.path.join(logs_path, 'metadata.tsv')

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    with open(logs_path + 'metadata.tsv', 'w', encoding='utf8') as f:
        for word in data.vocabulary.values:
            f.write(str(word[0]) + '\n')

    sess.run(tf.global_variables_initializer())

    print("Start training...")
    for ep in range(num_epochs):
        for i in range(data.total_batch):
            x_batch, y_batch = data.next_batch()

            feed_dict = {
                x: x_batch,
                y: y_batch,
                lengths: np.repeat(data.sequence_len, y_batch.shape[0]),
                keep_prob: dropout
            }

            model.is_training = True
            _, cost, acc, f1_train = sess.run([train_op, loss, accuracy, f1_score], feed_dict)

            # current_step = tf.train.global_step(sess, global_step)

            if i % data.total_batch - 1 == 0:
                saver.save(sess, os.path.join(logs_path, "model.ckpt"), i)

            if i % log_interval == 0:
                model.is_training = False
                val_acc, TP, TN, FP, FN, f1_valid, summary = sess.run([
                    accuracy, tp, tn, fp, fn, f1_score, merged_summary_op],
                    feed_dict={
                        x: valid_data.source[:2048],
                        y: valid_data.labels[:2048],
                        lengths: np.repeat(data.sequence_len, 2048),
                        keep_prob: 1
                    })

                # Write logs at every iteration
                summary_writer.add_summary(summary, ep * data.total_batch + i)
                projector.visualize_embeddings(summary_writer, config)

                print("TN:", TN, "FP:", FP)
                print("FN:", FN, "TP:", TP)
                print("epoch {}, " \
                      "step {}/{}, " \
                      "train_loss {:g}, " \
                      "train_acc {:g}, " \
                      "train_f1_score {:g}," \
                      "valid_acc {:g}, " \
                      "valid_f1_score {:g}".format(ep, i, data.total_batch,
                                                   cost, acc, f1_train, val_acc, f1_valid))
