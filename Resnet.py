import tensorflow as tf 
from collections import namedtuple
import six

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams', 'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                    'num_residual_units, use_bottleneck, weight_decay_rate,'
                    'relu_leakiness, optimizer')

tf.logging.warning()

class ResNet(object):

    def __init__(self, hps, images, labels, mode):
        self.hps = hps
        self._images = images
        self._labels = labels
        self.mode = mode
        self._extra_train_ops = []

    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        return [1, stride, stride, 1]
    
    def _build_model(self):
        with tf.variable_scope('init'):
            x = self._images
            x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual 
            filters = [16, 16, 32, 64]
        
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), activate_before_residual[0])
        
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[2], filters[3], self._stride_arr(1), False)
        
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), activate_before_residual[1])

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(strides[2]), False)
        
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr)