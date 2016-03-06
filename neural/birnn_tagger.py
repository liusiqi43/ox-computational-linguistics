from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
from reader import Reader
from tensorflow.models.rnn import rnn
from visualise import tsne

import numpy as np
import tensorflow as tf
import time

logging = tf.logging

tf.flags.DEFINE_boolean("plot_tsne", False, "Should plot t-SNE visualisation.")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability.")
tf.flags.DEFINE_float("init_scale", 0.1, "Initialization scale.")
tf.flags.DEFINE_float("learning_rate", 1.0, "Initial LR.")
tf.flags.DEFINE_float("lr_decay", 0.5, "LR decay.")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Maximum gradient norm.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size.")
tf.flags.DEFINE_integer("tsne_size", 1000, "Size of the sample to plot tSNE visualisation.")
tf.flags.DEFINE_integer("hidden_size", 128,
                        "Dimensionality of character embedding and lstm hidden size.")
tf.flags.DEFINE_integer("max_epoch", 4, "Max number of training epochs before LR decay.")
tf.flags.DEFINE_integer("max_max_epoch", 13, "Stop after max_max_epoch epochs.")
tf.flags.DEFINE_integer("num_layers", 3, "Number of stacked RNN layers.")

FLAGS = tf.flags.FLAGS


class BiRNNTagger(object):
  """The RNN POS tagger model."""

  def __init__(self, is_training, vocab_size, tag_size, maxlen):
    self._batch_size = FLAGS.batch_size
    self._hidden_size = FLAGS.hidden_size
    self._num_layers = FLAGS.num_layers
    self._dropout_keep_prob = FLAGS.dropout_keep_prob
    self._vocab_size = vocab_size
    self._tag_size = tag_size
    self._is_training = is_training

    self._input_data = tf.placeholder(tf.int32, [self._batch_size, maxlen])
    self._targets = tf.placeholder(tf.int32, [self._batch_size, maxlen])
    self._mask = tf.placeholder(tf.bool, [self._batch_size, maxlen])

    lstm_cell = tf.nn.rnn_cell.LSTMCell(self._hidden_size, self._hidden_size)
    if is_training and self._dropout_keep_prob < 1:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=self._dropout_keep_prob)

    cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self._num_layers)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self._num_layers)

    self._initial_state_fw = cell_fw.zero_state(self._batch_size, tf.float32)
    self._initial_state_bw = cell_bw.zero_state(self._batch_size, tf.float32)

    with tf.device("/cpu:0"):
      self._embedding = tf.get_variable("embedding", [self._vocab_size,
                                                      self._hidden_size])
      inputs = tf.nn.embedding_lookup(self._embedding, self._input_data)

    inputs = [input_ for input_ in tf.unpack(tf.transpose(inputs, [1, 0, 2]))]
    if is_training and self._dropout_keep_prob < 1:
        inputs = tf.nn.dropout(tf.pack(inputs), self._dropout_keep_prob)
        inputs = tf.unpack(inputs)
    outputs = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                    initial_state_fw=self._initial_state_fw,
                                    initial_state_bw=self._initial_state_bw)
    # output from forward and backward cells.
    output = tf.reshape(tf.concat(1, outputs), [-1, 2 * self._hidden_size])
    softmax_w = tf.get_variable("softmax_w", [2 * self._hidden_size, self._tag_size])
    softmax_b = tf.get_variable("softmax_b", [self._tag_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.reshape(tf.cast(self._mask, tf.float32), [-1])], self._tag_size)
    self._cost = cost = tf.reduce_sum(loss) / self._batch_size

    equality = tf.equal(tf.argmax(logits, 1),
                        tf.cast(tf.reshape(self._targets, [-1]), tf.int64))
    masked = tf.boolean_mask(equality, tf.reshape(self.mask, [-1]))
    self._misclass = 1 - tf.reduce_mean(tf.cast(masked, tf.float32))

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      FLAGS.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def mask(self):
    return self._mask

  @property
  def targets(self):
    return self._targets

  @property
  def embedding(self):
    return self._embedding

  @property
  def cost(self):
    return self._cost

  @property
  def is_training(self):
    return self._is_training

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def misclass(self):
    return self._misclass

def run_epoch(session, m, x_data, y_data, mask, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(x_data) // m.batch_size) - 1)
  start_time = time.time()
  costs = 0.0
  iters = 0
  misclass_ = []
  for step, (x, y, mask) in enumerate(Reader.iterator(x_data, y_data, mask, m.batch_size)):
      cost, misclass, _ = session.run([m.cost, m.misclass, eval_op],
                                      {m.input_data: x, m.targets: y, m.mask: mask})
      costs += cost
      iters += m.batch_size

      if verbose and step % (epoch_size // 10) == 0:
          print("[%s] %.3f perplexity: %.3f misclass:%.3f speed: %.0f wps" %
                ('train' if m.is_training else 'test', step * 1.0 / epoch_size,
                 np.exp(costs / iters), misclass,
                 iters * m.batch_size / (time.time() - start_time)))
      misclass_.append(misclass)
  return np.exp(costs / iters), np.mean(misclass_)

def main(unused_args):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.iteritems()):
        print("{}={}".format(attr, value))
    print("")

    reader = Reader(split = 0.9)
    (x_train, y_train, mask_train,
     x_test, y_test, mask_test) = reader.get_data(glob('../../WSJ-2-12/*/*.POS'))
    print('len(x_train)', len(x_train), 'len(x_test)', len(x_test))
    print('reader.ignore_ids', reader.ignore_ids)
    print('len(reader.word_to_id)',len(reader.word_to_id),
          'len(reader.tag_to_id)', len(reader.tag_to_id))
    best_misclass = 1.0

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = BiRNNTagger(True, len(reader.word_to_id), len(reader.tag_to_id), reader.maxlen)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = BiRNNTagger(False, len(reader.word_to_id), len(reader.tag_to_id),
                                reader.maxlen)

        tf.initialize_all_variables().run()

        saver = tf.train.Saver()
        for i in range(FLAGS.max_max_epoch):
            lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
            m.assign_lr(session, FLAGS.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity, _ = run_epoch(session, m, x_train, y_train, mask_train,
                                            m.train_op, verbose=True)
            _, misclass = run_epoch(session, mtest, x_test, y_test, mask_test,
                                    tf.no_op(), verbose=True)
            if misclass < best_misclass:
                best_misclass = misclass
                fname = 'models/dropout_bid3rnn_tagger_' + str(best_misclass)
                saver.save(session, fname, global_step=i)
                print('saving', fname)

        if FLAGS.plot_tsne:
            tsne(session.run(m.embedding), reader.word_to_id, FLAGS.tsne_size)


if __name__ == "__main__":
    tf.app.run()

