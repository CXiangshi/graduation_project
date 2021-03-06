import os
import sys
import tarfile
import re

import tepnsorflow as tf

import GP_input_cifar10
from shuffle_net import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128)
tf.app.flags.DEFINE_string('data_dir', '/')
tf.app.flags.DEFINE_boolean('use_fp16', False)

IMAGE_SIZE = GP_input_cifar10.IMAGE_SIZE
NUM_CLASSES = GP_input_cifar10.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = GP_input_cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = GP_input_cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'

def activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activation', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def variable_on_cpu(name, shape, initializer):

    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def variable_with_weight_decay(name, shape, stddev, wd):

    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        var = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    return var

def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = GP_input_cifar10.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = GP_input_cifar10.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inference(images):

    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weight', shape=[5,5,3,64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding="SAME")
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha= 0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5,5,64,64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 /9.0, beta=0.75, name='norm2')

    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME', name='conv2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0],-1])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        activation_summary(local3)

    with tf.variable_scope('local4') as scope:
        weights = variable_with_weight_decay('weights', shape=[384,192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        activation_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=None)
        biases = variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        activation_summary(softmax_linear)
    
    return softmax_linear

def inference1(images):
    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weight', shape=[3,3,3,32], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(images, kernel, strides=[1,1,1,1], padding="SAME")
        biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv1)
    
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    norm1 = tf.layers.batch_normalization(pool1)

    with tf.variable_scope('dw_conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3,3,32,1], stddev=5e-2, wd=None)
        conv = tf.nn.depthwise_conv2d(norm1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biaes', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        dw_conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
    norm2 = tf.layers.batch_normalization(dw_conv1)

    with tf.variable_scope('pw_conv1') as scope:
        kernel = variable_with_weight_decay('weghits', shape=[1,1,32,64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm2, kernel, strides=[1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        pw_conv1 = tf.nn.relu(pre_activation, name=scope.name)

    norm3 = tf.layers.batch_normalization(pw_conv1)

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, [images.get_shape().as_list()[0],-1])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        activation_summary(local3)

    with tf.variable_scope('local4') as scope:
        weights = variable_with_weight_decay('weights', shape=[384,192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        activation_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=None)
        biases = variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        activation_summary(softmax_linear)

    return softmax_linear

def shuffle_net_infer(images):
    conv_o = conv2d('conv1', images, w=None, num_filters=64, kernel_size=(3,3), activation=True)
    shuf_o1 = shufflenet_unit('shuffle_u1', conv_o, w=None, num_filters=64, num_groups=2)
    shuf_o2 = shufflenet_unit('shuffle_u2', shuf_o1, w=None, num_groups=2, num_filters=128, stride=(2,2))
    


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection(tf.get_collection('losses'), cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    
    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=False)

    loss_averages_op = add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies(apply_gradient_op):
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    return variable_averages_op
    