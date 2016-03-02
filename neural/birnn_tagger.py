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
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding.")
tf.flags.DEFINE_integer("hidden_size", 200, "Hidden size of LSTM cell.")
tf.flags.DEFINE_integer("max_epoch", 4, "Max number of training epochs before LR decay.")
tf.flags.DEFINE_integer("max_max_epoch", 13, "Stop after max_max_epoch epochs.")
tf.flags.DEFINE_integer("num_steps", 15, "Sequence length of RNN.")

FLAGS = tf.flags.FLAGS


class BiRNNTagger(object):
  """The RNN POS tagger model."""

  def __init__(self, is_training, vocab_size, tag_size):
    self._batch_size = FLAGS.batch_size
    self._num_steps = FLAGS.num_steps
    self._hidden_size = FLAGS.hidden_size
    self._embedding_dim = FLAGS.embedding_dim
    self._dropout_keep_prob = FLAGS.dropout_keep_prob
    self._vocab_size = vocab_size
    self._tag_size = tag_size
    self._is_training = is_training

    self._input_data = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])
    self._targets = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])

    cell_fw = tf.nn.rnn_cell.LSTMCell(self._hidden_size, self._embedding_dim)
    cell_bw = tf.nn.rnn_cell.LSTMCell(self._hidden_size, self._embedding_dim)
    if is_training and self._dropout_keep_prob < 1:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw, output_keep_prob=self._dropout_keep_prob)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw, output_keep_prob=self._dropout_keep_prob)

    self._initial_state_fw = cell_fw.zero_state(self._batch_size,
                                                     tf.float32)
    self._initial_state_bw = cell_bw.zero_state(self._batch_size,
                                                     tf.float32)

    with tf.device("/cpu:0"):
      self._embedding = tf.get_variable("embedding", [self._vocab_size,
                                                      self._embedding_dim])
      inputs = tf.nn.embedding_lookup(self._embedding, self._input_data)

    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, self._num_steps, inputs)]
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
        [tf.ones([self._batch_size * self._num_steps])], self._tag_size)
    self._cost = cost = tf.reduce_sum(loss) / self._batch_size

    pred = tf.argmax(logits, 1)
    labels = tf.cast(tf.reshape(self._targets, [-1]), tf.int64)
    self._misclass = 1 - tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

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
  def num_steps(self):
    return self._num_steps

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def misclass(self):
    return self._misclass

def run_epoch(session, m, x_data, y_data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(x_data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  misclass_ = []
  for step, (x, y) in enumerate(Reader.iterator(x_data, y_data, m.batch_size, m.num_steps)):
      cost, misclass, _ = session.run([m.cost, m.misclass, eval_op],
                                      {m.input_data: x, m.targets: y})
      costs += cost
      iters += m.num_steps

      if verbose and step % (epoch_size // 10) == 10:
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
    x_train, y_train, x_test, y_test = reader.get_data(glob('../../WSJ-2-12/*/*.POS'))
    print('len(reader.word_to_id)',len(reader.word_to_id),
          'len(reader.tag_to_id)', len(reader.tag_to_id))
    print('len(x_train)',len(x_train),
          'len(x_test)', len(x_test))
    best_misclass = 1.0

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = BiRNNTagger(True, len(reader.word_to_id), len(reader.tag_to_id))
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = BiRNNTagger(False, len(reader.word_to_id), len(reader.tag_to_id))

        tf.initialize_all_variables().run()

        saver = tf.train.Saver()
        for i in range(FLAGS.max_max_epoch):
            lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
            m.assign_lr(session, FLAGS.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity, _ = run_epoch(session, m, x_train, y_train,
                                            m.train_op, verbose=True)
            _, misclass = run_epoch(session, mtest, x_test, y_test, tf.no_op(), verbose=True)
            if misclass < best_misclass:
                best_misclass = misclass
                fname = 'models/dropout_bidrnn_tagger_' + str(best_misclass)
                saver.save(session, fname, global_step=i)
                print('saving', fname)

        if FLAGS.plot_tsne:
            tsne(session.run(m.embedding), reader.word_to_id, FLAGS.tsne_size)


if __name__ == "__main__":
    tf.app.run()

