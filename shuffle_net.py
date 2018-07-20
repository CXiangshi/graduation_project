import tensorflow as tf 
import numpy as np

def _conv2d_p(name, inputs, w=None, num_filters=16, kernel_size=(3,3), padding='SAME', stride=(1,1),
                initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.shape[2], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer,l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_bias'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(inputs, w, strides, padding)
            out = tf.nn.bias_add(conv, bias)
    
    return out 

def conv2d(name, inputs, w=None, num_filters=16, kernel_size=(3,3), padding='SAME', stride=(1,1),
            initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
            activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
            is_training=True):
    
    with tf.variable_scope(name) as scope:
        conv_o_b = _conv2d_p('conv', inputs=inputs, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
        padding=padding, initializer=initializer, l2_strength=l2_strength, bias=bias)

        activation = tf.nn.relu
        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
        
        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)
        
        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)
        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a
        
        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)
    return conv_o

def grouped_conv2d(name, inputs, w=None, num_filters=16, kernel_size=(3,3), padding="SAME",
                    stride=(1,1), initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, 
                    l2_strength=0.0, bias=0.0, activation=None, batchnorm_enabled=False, 
                    dropout_keep_prob=-1, is_training=True):
    with tf.variable_scope(name) as scope:
        sz = inputs.get_shape()[3].value // num_groups
        conv_side_layers = [conv2d(name + "_" + str(i), inputs[:,:,:,i*sz:i*sz+sz], w, num_filters // num_groups, kernel_size,
                                    padding, stride, initializer, l2_strength, bias, activation=None,
                                    batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                                    is_training=is_training) for i in range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)
        return conv_a

def _depthwise_conv2d_p(name, inputs, w=None, kernel_size=(3,3), padding='SAME', stride=(1,1),
                        initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layers_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [inputs.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(inputs, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)
    return out

def depthwise_conv2d(name, inputs, w=None, kernel_size=(3,3), padding='SAME', stride=(1,1), 
                    initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias= 0.0, activation=None,
                    batchnorm_enabled=False, is_training=False):
    with tf.variable_scope(name) as scope:
        conv_o_b = _depthwise_conv2d_p(name='dw_conv', inputs=inputs, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)
        
        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    return conv_a

def shufflenet_unit(name, inputs, w=None, num_groups=1, group_conv_bottleneck=True, num_filters=16, stride=(1,1),
                    l2_strength=0.0, bias=0.0, batchnorm_enabled=True, is_training=True, fusion='add'):
    activation = tf.nn.relu

    with tf.variable_scope(name) as scope:
        residual = inputs
        bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[3].value) // 4

        if group_conv_bottleneck:
            bottleneck = grouped_conv2d('Gbottleneck', inputs=inputs, w=None, num_filters=bottleneck_filters, kernel_size=(1,1),
                                        padding='VALID', num_groups=num_groups, l2_strength=l2_strength, bias=bias, batchnorm_enabled=batchnorm_enabled,
                                        is_training=is_training)
            shuffled = channel_shuffle('channel_shuffle', bottleneck, num_groups)
        else:
            bottleneck = conv2d('bottleneck', inputs=inputs, w=None, num_filters=bottleneck_filters, kernel_size=(1,1),
                                padding='VALID', l2_strength=l2_strength, bias=bias, activation=activation,
                                batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = bottleneck
        padded = tf.pad(shuffled, [[0,0], [1,1], [1,1], [0,0]], 'CONSTANT')
        depthwise = depthwise_conv2d('depthwise', inputs=padded, w=None, stride=stride, l2_strength=l2_strength, padding='VALID',
                                    bias=bias, activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
        if stride == (2, 2):
            residual_pooled = avg_pool_2d(residual, size=(3,3), stride=(2,1), padding='SAME')
        else:
            residual_pooled = residual
        
        if fusion == 'concat':
            group_conv1x1 = grouped_conv2d('Gconv1x1', inputs=depthwise, w=None,
                                            num_filters=num_filters - residual.get_shape()[3].value,
                                            kernel_size=(1,1), padding='VALID', num_groups=num_groups,
                                            l2_strength=l2_strength, bias=bias,
                                            activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
        elif fusion == 'add':
            group_conv1x1 = grouped_conv2d('Gconv1x1', inputs=depthwise, w=None,
                                            num_filters=num_filters - residual.get_shape()[3].value,
                                            kernel_size=(1,1),
                                            padding='VALID',
                                            num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                            activation=None,
                                            batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            residual_match = residual_pooled

            if num_filters != residual_pooled.get_shape()[3].value:
                residual_match = conv2d('residual_match', inputs=residual_pooled, num_filters=num_filters,
                                        kernel_size=(1,1), padding='VALID', l2_strength=l2_strength,
                                        bias=bias, activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
                return activation(group_conv1x1 + residual_match)
            else:
                raise ValueError("Specify whether the fusion is \'concat\' or \'add\' ")
            
def channel_shuffle(name, inputs, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = inputs.shape.as_list()
        inputs_reshape = tf.reshape(inputs, [-1, h, w, num_groups, c//num_groups])
        inputs_transposed = tf.transpose(inputs_reshape, [0, 1, 2, 4, 3])
        output = tf.reshape(inputs_transposed, [-1, h, w, c])
        return output

def _dense_p(name, inputs, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
            bias=0.0):
    n_in = inputs.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable('layer_biases', [output_dim], tf.float32, tf.constant_initializer(bias))
        __variable_summaries(bias)
        output = tf.nn.bias_add(tf.matmul(inputs, w), bias)
        return output

def dense(name, inputs, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
            bias=0.0, activation=None, batchnorm_enabled=False, dropout_keep_prob=-1, is_training=True):
    with tf.variable_scope(name) as scope:
        dense_o_b = _dense_p(name='dense', inputs=inputs, w=w, output_dim=output_dim, initializer=initializer,
                            l2_strength=l2_strength, bias=bias)
        
        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)
        
        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)
        
        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)
        
        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a
        dense_o = dense_o_dr
    return dense_o

def flatten(inputs):
    all_dims_exc_first = np.prod([inputs.value for vi in inputs.get_shape()[1:]])
    o = tf.reshape(inputs, [-1, all_dims_exc_first])
    return o

def max_pool_2d(inputs, size=(2,2), stride=(2,2), name='pooling'):
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(inputs, ksize=[1,size_x, size_y, 1], strides=[1,stride_x, stride_y, 1], name=name)

def avg_pool_2d(inputs, size=(2,2), stride=(2,2), name='avg_pooling', padding='VALID'):
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(inputs, ksize=[1,size_x,size_y,1], stride=[1,stride_x,stride_y,1],padding=padding, name=name)

def __variable_with_weight_decay(kernel_shape, initializer, wd):
    w = tf.get_variable('weight', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w

def __variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
