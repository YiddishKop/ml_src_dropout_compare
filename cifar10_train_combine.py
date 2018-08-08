# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10.train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_combine.eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import os

import numpy as np
import tensorflow as tf

import cifar10_combine

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', sys.argv[1],
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('start_sparse_step', int(sys.argv[5]),
                            """Number of step to change mode.""")
tf.app.flags.DEFINE_float('sparsity', float(sys.argv[4]),
                            """The sparsity of the sparse layer.""")
tf.app.flags.DEFINE_float('keep_prob', float(sys.argv[3]),
                            """The keep probablility of the dropout rate.""")
tf.app.flags.DEFINE_string('checkpoint_dir', sys.argv[1],
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('record', sys.argv[2],
                           """File where to write records.""")


def eval_once(saver, top_k_op, keep_prob, epoch, mask_placeholder, mask):
  """Run Eval once.

  Args:
    saver: Saver.
    top_k_op: Top K op.
    keep_prob: placeholder of dropout ratio.
    epoch: steps of batch so far.
    mask_placeholder: the placeholder of the mask.
    mask: a dictionary of the mask of weights so far.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_combine.train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op],feed_dict={
            keep_prob : 1.0,
            mask_placeholder['conv2'] : mask['conv2'],
            mask_placeholder['local3'] : mask['local3'],
            mask_placeholder['local4'] : mask['local4']
          })
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return precision

def evaluate(step, mask):
  """Eval CIFAR-10 for a number of steps.

  Args:
    step: Step so far.
    mask: a dictionary of the mask of weights so far.
  """

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10_combine.inputs(eval_data=eval_data)
    keep_prob = tf.placeholder(tf.float32)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    _, mask_placeholder, logits = cifar10_combine.inference(images, keep_prob)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10_combine.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    return eval_once(saver, top_k_op, keep_prob, step, mask_placeholder, mask)

def filter_weight(weights,shape,sparsity,method_type):
  """Make the mask for sparse step.

  Args:
    weights: 2D or 4D tensor, the weights of the layer in the network.
    shape: list, the shape of the weights.
    sparsity: float32, the sparsity of the layer.
    method_type: string, the type of the mask out weight method.
  
  Return:
    mask: 2D or 4D numpy array, the mask of the layer
  """
  if method_type == "DSD":
    sort_abs_weights = np.sort(np.absolute(weights.ravel()))
    top_k_value = sort_abs_weights[int(math.ceil(len(sort_abs_weights)*sparsity))]
    abs_weights = np.absolute(weights.ravel())
    mask = np.asarray([0 if weight < top_k_value else 1 for weight in abs_weights]).reshape(shape)
  elif method_type == "Drop_connections":
    weights = weights.ravel()
    mask = np.asarray([0 if np.random.uniform(0,1) < sparsity else 1 for weight in weights]).reshape(shape)
  elif method_type == "DSD_Uniform":
    abs_weights = np.absolute(weights.ravel())
    mask = np.asarray([0 if np.random.uniform(0,1) < sparsity  else 1 for weight in weights]).reshape(shape)

  return mask

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10_combine.distorted_inputs()
    keep_prob = tf.placeholder(tf.float32)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    network_weights_placeholder, mask_placeholder, logits = cifar10_combine.inference(images, keep_prob)

    # Calculate loss.
    loss = cifar10_combine.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10_combine.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
          
    config = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement,
        allow_soft_placement = True)

    config.gpu_options.allow_growth = True
    
    f = open(FLAGS.record,"w")
    f.write("step,loss,accuracy\n")
    
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=300,
        config=config) as mon_sess:

      # validation accurarcy
      accuracy = 0
      step = 0

      print ("Dense !")
      f.write("Dense!\n")
      mask = {
        'conv2' : np.ones([5,5,64,64]),
        'local3' : np.ones([2304,384]),
        'local4' : np.ones([384,192])
      }

      while step < FLAGS.start_sparse_step:
        _, network_weights, step_loss = mon_sess.run([train_op, network_weights_placeholder, loss], feed_dict={
          keep_prob : FLAGS.keep_prob,
          mask_placeholder['conv2'] : mask['conv2'],
          mask_placeholder['local3'] : mask['local3'],
          mask_placeholder['local4'] : mask['local4']
          })

        step += 1
        if step % 500 == 0:
          temp_acc = evaluate(step,mask)
          if temp_acc != accuracy:
            f.write("{},{},{}\n".format(step, step_loss, temp_acc))
            print ("Write Record! {} _ {}".format(accuracy,temp_acc))
            accuracy = temp_acc

      # current_min_loss records the converge state. [min_loss, min_loss_step]
      current_min_loss = [0,step]
      converge = False

      while not converge:
        _, network_weights, step_loss = mon_sess.run([train_op, network_weights_placeholder, loss], feed_dict={
          keep_prob : FLAGS.keep_prob,
          mask_placeholder['conv2'] : mask['conv2'],
          mask_placeholder['local3'] : mask['local3'],
          mask_placeholder['local4'] : mask['local4']
          })

        step += 1
        if step % 500 == 0:
          mask['conv2'] = filter_weight(network_weights['conv2'], [5,5,64,64], FLAGS.sparsity, "DSD")
          mask['local3'] = filter_weight(network_weights['local3'], [2304,384], FLAGS.sparsity, "DSD")
          mask['local4'] = filter_weight(network_weights['local4'], [384,192], FLAGS.sparsity, "DSD")
          temp_acc = evaluate(step,mask)
          if temp_acc != accuracy:
            f.write("{},{},{}\n".format(step, step_loss, temp_acc))
            print ("Write Record! {} _ {}".format(accuracy,temp_acc))
            accuracy = temp_acc

          if temp_acc - current_min_loss[0] > 1e-5:
            current_min_loss = [temp_acc,step]
          elif step - current_min_loss[1] > 10000:
            f.write(str(step)+","+str(current_min_loss)+"\n")
            converge = True


def main(argv=None):  # pylint: disable=unused-argument
  cifar10_combine.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
