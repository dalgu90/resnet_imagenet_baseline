#
# This code is modified from the TensorFlow tutorial below.
#
# TensorFlow Tutorial - Convolutional Neural Networks
#  (https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html)
#
# ==============================================================================

"""Builds the bypass network from

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs(data_class, shuffle)
 inputs, labels = inputs(data_class, shuffle)

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import resnet.resnet as resnet
import resnet.utils as utils

from IPython import embed

import cPickle as pickle
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('batch_size', 32, "Number of images to process in a batch.")
tf.app.flags.DEFINE_float('l2_weight', 0.0001, "L2 loss weight applied to all the weights except the last fc layers")
tf.app.flags.DEFINE_float('l1_weight', 0.001, "L1 loss weight applied to the last fc layers")
tf.app.flags.DEFINE_float('initial_lr', 0.1, "Initial learning rate")
tf.app.flags.DEFINE_float('lr_step_epoch', 50.0, "Epochs after which learing rate decays")
tf.app.flags.DEFINE_float('lr_decay', 0.1, "Learning rate decay factor")
tf.app.flags.DEFINE_float('momentum', 0.9, "The momentum used in MomentumOptimizer")
tf.app.flags.DEFINE_string('pretrained_dir', './pretrain', "Directory where to load pretrained model.(Only for --finetune True")

import imagenet_input as data_input


# Global constants describing the data set.
IMAGE_HEIGHT = data_input.IMAGE_HEIGHT
IMAGE_WIDTH = data_input.IMAGE_WIDTH
NUM_CLASSES = data_input.NUM_CLASSES

# Constants describing the network
print('[Network Configuration]')
NUM_GPUS = FLAGS.num_gpus
BATCH_SIZE = FLAGS.batch_size
L2_LOSS_WEIGHT = FLAGS.l2_weight
L1_LOSS_WEIGHT = FLAGS.l1_weight
print('\tNumber of GPUs: %d' % NUM_GPUS)
print('\tBatch size: %d' % BATCH_SIZE)
print('\tL2 loss weight: %f' % L2_LOSS_WEIGHT)
print('\tL1 loss weight: %f' % L1_LOSS_WEIGHT)

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.99     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = FLAGS.lr_step_epoch      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = FLAGS.lr_decay  # Learning rate decay factor.
MOMENTUM = FLAGS.momentum  # Momentum.
INITIAL_LEARNING_RATE = FLAGS.initial_lr       # Initial learning rate.
print('\tMoving average decay: %f' % MOVING_AVERAGE_DECAY)
print('\tNumber of epochs per decay: %f' % NUM_EPOCHS_PER_DECAY)
print('\tLearning rate decay factor: %f' % LEARNING_RATE_DECAY_FACTOR)
print('\tInitial learning rate %f' % INITIAL_LEARNING_RATE)

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
_histogram_summary_name_list = []
def _histogram_summary(x):
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    global _histogram_summary_name_list
    if tensor_name not in _histogram_summary_name_list:
      _histogram_summary_name_list.append(tensor_name)
      print('%s histogram summary added' % tensor_name)
      with tf.device('/cpu:0'):
        tf.histogram_summary(tensor_name + '/activations', x)

_sparsity_summary_name_list = []
def _sparsity_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    global _sparsity_summary_name_list
    if tensor_name not in _sparsity_summary_name_list:
      sparsity = tf.nn.zero_fraction(x)
      _sparsity_summary_name_list.append(tensor_name)
      print('%s sparsity summary added' % tensor_name)
      with tf.device('/cpu:0'):
        tf.scalar_summary(tensor_name + '/sparsity', sparsity)

_almost_sparsity_summary_name_list = []
def _almost_sparsity_summary(x, eps):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    global _almost_sparsity_summary_name_list
    key = '%s_%f' % (tensor_name, eps)
    if key not in _almost_sparsity_summary_name_list:
      eps_t = tf.constant(eps, dtype=x.dtype, name="threshold")
      fraction = tf.reduce_mean(tf.cast(tf.less(tf.abs(x), eps_t), tf.float32))
      _almost_sparsity_summary_name_list.append(key)
      print('%s almost sparsity summary added' % key)
      with tf.device('/cpu:0'):
        tf.scalar_summary(tensor_name + '/less_than_' + ("%g" % eps), fraction)


def distorted_inputs(data_class, shuffle=True):
    """Construct input for training using the Reader ops.

    Args:
      data_class: string, indicating if one should use the 'train' or 'eval' or 'test' data set.
      shuffle: bool, to shuffle dataset list to read

    Returns:
      images: Images. 4D tensor of [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [BATCH_SIZE] size.

    Raises:
      ValueError: If no data_dir
    """
    return data_input.distorted_inputs(data_class=data_class,
                                       batch_size=BATCH_SIZE,
                                       shuffle=shuffle)


def inputs(data_class, shuffle=True):
    """Construct input for evaluation using the Reader ops.

    Args:
      data_class: string, indicating if one should use the 'train' or 'eval' or 'test' data set.
      shuffle: bool, to shuffle dataset list to read

    Returns:
      images: Images. 4D tensor of [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [BATCH_SIZE] size.

    Raises:
      ValueError: If no data_dir
    """
    return data_input.inputs(data_class=data_class,
                             batch_size=BATCH_SIZE,
                             shuffle=shuffle)


def inference(images, scope=""):
  """Build a single tower of the resnet model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    logits: Logits. 2D tensor of [BATCH_SIZE, NUM_CLASSES] size.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().

  ###### Pretrained ResNet ######
  resnet_model = resnet.Model(50, True)
  resnet_model.build(images)

  graph = tf.get_default_graph()
#  conv1 = graph.get_operation_by_name(scope+"conv1/relu").outputs[0]
#  res2a = graph.get_operation_by_name(scope+"res2a/relu").outputs[0]
#  res2b = graph.get_operation_by_name(scope+"res2b/relu").outputs[0]
#  res2c = graph.get_operation_by_name(scope+"res2c/relu").outputs[0]
#  res3a = graph.get_operation_by_name(scope+"res3a/relu").outputs[0]
#  res3b = graph.get_operation_by_name(scope+"res3b/relu").outputs[0]
#  res3c = graph.get_operation_by_name(scope+"res3c/relu").outputs[0]
#  res3d = graph.get_operation_by_name(scope+"res3d/relu").outputs[0]
#  res4a = graph.get_operation_by_name(scope+"res4a/relu").outputs[0]
#  res4b = graph.get_operation_by_name(scope+"res4b/relu").outputs[0]
#  res4c = graph.get_operation_by_name(scope+"res4c/relu").outputs[0]
#  res4d = graph.get_operation_by_name(scope+"res4d/relu").outputs[0]
#  res4e = graph.get_operation_by_name(scope+"res4e/relu").outputs[0]
#  res4f = graph.get_operation_by_name(scope+"res4f/relu").outputs[0]
#  res5a = graph.get_operation_by_name(scope+"res5a/relu").outputs[0]
#  res5b = graph.get_operation_by_name(scope+"res5b/relu").outputs[0]
##  res5c = graph.get_operation_by_name(scope+"res5c/relu").outputs[0] # Not used due to the duplication of pool5
#  pool5 = graph.get_operation_by_name(scope+"pool5").outputs[0]
  logits = graph.get_operation_by_name(scope+"BiasAdd").outputs[0]

  return logits


def loss_acc(logits, labels):
  """ Calculate loss and accuracy

  Args:
    logits: Logits from inference(). 2D tensor of shape [batch_size, NUM_CLASSES].
    labels: Labels from distorted_inputs() or inputs(). 1-D tensor of shape [batch_size]

  Returns:
    Loss, accuracy tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # Calculate accuracy
  ones = tf.constant(np.ones([BATCH_SIZE]), dtype=tf.float32)
  zeros = tf.constant(np.zeros([BATCH_SIZE]), dtype=tf.float32)
  preds = tf.argmax(logits, 1, name='preds')
  correct = tf.select(tf.equal(preds, labels), ones, zeros)
  accuracy = tf.reduce_mean(correct)

  return cross_entropy_mean, accuracy


def _add_loss_summaries(total_loss):
  """Add summaries for losses in the bypass model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().

  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply([total_loss])

  tf.scalar_summary(total_loss.op.name + ' (raw)', total_loss)
  tf.scalar_summary(total_loss.op.name, loss_averages.average(total_loss))

  return loss_averages_op


def train(total_loss, global_step):
  """Train the bypass model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  print('\t%d batches per epoch' % num_batches_per_epoch)
  print('\t%d batches per learning rate decay' % decay_steps)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  BASENET_TR_VAR_NUM = 161

  # Compute gradients.
  print('Compute gradient')
  all_tr_vars = tf.trainable_variables()
  basenet_tr_vars = all_tr_vars[:BASENET_TR_VAR_NUM]
  basenet_opt = tf.train.MomentumOptimizer(lr, MOMENTUM)
  basenet_grads = basenet_opt.compute_gradients(total_loss, basenet_tr_vars)

  # Apply gradients.
  apply_grad_op_list = []
  basenet_apply_grad_op = basenet_opt.apply_gradients(basenet_grads,
                                                      global_step=global_step)
  apply_grad_op_list.append(basenet_apply_grad_op)

  with tf.control_dependencies(apply_grad_op_list):
    train_op = tf.no_op(name='train')

  return train_op, lr



def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # If no gradient for a variable, exclude it from output
    if grad_and_vars[0][0] is None:
      continue

    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)

  return average_grads


def proximal_step(train_op, lr):
  # Apply weight decay for the variables with l2 loss
  # If basenet weights are trained together, do not set a weight decay on the
  # conv layers of the basenet
  l2_op_list = []
  l1_op_list = []
  with tf.control_dependencies([train_op]):
    if L2_LOSS_WEIGHT > 0:
      for var in tf.get_collection(utils.WEIGHT_DECAY_KEY):
        assign_op = var.assign_add(- lr * tf.convert_to_tensor(L2_LOSS_WEIGHT) * var)
        l2_op_list.append(assign_op)
        print('\tL2 loss added: %s(strength: %f)' % (var.name, L2_LOSS_WEIGHT))

    # Apply proximal gradient for the variables with l1 lasso loss
    # Non-negative weights constraint
    if L1_LOSS_WEIGHT > 0:
      for var in tf.get_collection(utils.LASSO_KEY):
        th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(L1_LOSS_WEIGHT) * lr)
        zero_t = tf.zeros(tf.shape(var))
        var_temp = var - th_t * tf.sign(var)
        assign_op = var.assign(tf.select(tf.less(var, th_t), zero_t, var_temp))
        l1_op_list.append(assign_op)
        print('\tL1 loss added: %s(strength: %f)' % (var.name, L1_LOSS_WEIGHT))

  with tf.control_dependencies(l2_op_list + l1_op_list):
    train_op = tf.no_op(name='proximal_step')

  return train_op


import time
import datetime
def _get_time_op(text):
  print('%s %s' % (str(datetime.datetime.now()), text))
  return time.time()


def inf_loss_train_multi_gpu(images_list, labels_list, global_step):
  # Check whether the length of images_list is equal to NUM_GPUS
  assert len(images_list) == NUM_GPUS
  assert len(labels_list) == NUM_GPUS

  # Variables that affect learning rate.
  num_batches_per_epoch = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE / FLAGS.num_gpus
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  print('\t%d batches per epoch' % num_batches_per_epoch)
  print('\t%d batches per learning rate decay' % decay_steps)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  BASENET_TR_VAR_NUM = 161

  # Momentum optimizer
  basenet_opt = tf.train.MomentumOptimizer(lr, MOMENTUM)

  # build towers of bypass network
  logits_list = []
  loss_list = []
  acc_list = []
  basenet_grads_list = []
  bypass_grads_list = []
  with tf.control_dependencies(images_list):
    time_1 = tf.py_func(_get_time_op, ['before inf,grad'], [tf.float64])
    time_1 = tf.to_float(time_1[0])

  with tf.control_dependencies([time_1]):
    for i in range(NUM_GPUS):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
          # Inference for one tower of the model. This function constructs the entire model
          # but shares the variables across all towers.
          print('\tBuild a network for ' + scope)
          logits = inference(images_list[i], scope)
          logits_list.append(logits)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          print('\tCompute loss and accuracy for ' + scope)
          loss, acc = loss_acc(logits_list[i], labels_list[i])
          loss_list.append(loss)
          acc_list.append(acc)

          print('\tCompute gradient for ' + scope)
          # Get all trainable variables for this tower.
          tower_tr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

          # Calculate the gradients for the batch of data on this tower.
          basenet_tr_vars = tower_tr_vars[:BASENET_TR_VAR_NUM]
          basenet_grads = basenet_opt.compute_gradients(loss_list[i], basenet_tr_vars)
          basenet_grads_list.append(basenet_grads)

  with tf.control_dependencies([gv[0] for gv in basenet_grads_list[0]]):
    time_2 = tf.py_func(_get_time_op, ['before avg grad'], [tf.float64])
    time_2 = tf.to_float(time_2[0])

  with tf.control_dependencies([time_2]):
    with tf.device('/cpu:0'):
      # Average losses and accuracies
      loss = tf.identity(tf.add_n(loss_list) / NUM_GPUS, name='train/loss')
      acc = tf.identity(tf.add_n(acc_list) / NUM_GPUS, name='train/accuracy')
      # Generate moving averages of all losses and associated summaries.
      loss_averages_op = _add_loss_summaries(loss)
      tf.scalar_summary('train/accuracy', acc)

      # Average gradients and apply
      apply_grad_op_list = []
      basenet_grads = average_gradients(basenet_grads_list)
      with tf.control_dependencies([gv[0] for gv in basenet_grads]):
        time_3 = tf.py_func(_get_time_op, ['before apply grad'], [tf.float64])
        time_3 = tf.to_float(time_3[0])
      with tf.control_dependencies([time_3]):
        basenet_apply_grad_op = basenet_opt.apply_gradients(basenet_grads,
                                                            global_step=global_step)
        apply_grad_op_list.append(basenet_apply_grad_op)

      with tf.control_dependencies(apply_grad_op_list + [loss_averages_op]):
        time_4 = tf.py_func(_get_time_op, ['after apply grad'], [tf.float64])
        time_4 = tf.to_float(time_4[0])
      with tf.control_dependencies([time_4]):
        train_op = tf.no_op(name='train')

  return loss, acc, train_op, lr
