import tensorflow as tf
import keras
from fcos import FCOSModule
from keras import Sequential
from keras.layers import Conv2D
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1, resnet_utils


def resnet_arg_scope(
        is_training=True, weight_decay=cfg.SOLVER.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def backbone(cfg, img, scope_name='resnet_v1_101', is_training=True):
    blocks = [
        resnet_v1.resnet_v1_block(
            'block1', base_depth=64, num_units=3, stride=2),
        resnet_v1.resnet_v1_block(
            'block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block(
            'block3', base_depth=256, num_units=23, stride=2),
        resnet_v1.resnet_v1_block(
            'block4', base_depth=512, num_units=3, stride=2)
    ]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            net = resnet_utils.conv2d_same(
                img, cfg.MODEL.FCOS.NUM_CLASSES, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfg.MODEL.RESNETS.FIXED_BLOCKS + \
        (4-cfg.MODEL.RESNETS.FIXED_BLOCKS)*[True]

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, _ = resnet_v1.resnet_v1(net,
                                    blocks[0:1],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, _ = resnet_v1.resnet_v1(C2,
                                    blocks[1:2],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, _ = resnet_v1.resnet_v1(C3,
                                    blocks[2:3],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    # add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, _ = resnet_v1.resnet_v1(C4,
                                    blocks[3:4],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    feature_dict = {'C2': C2,
                    'C3': C3,
                    'C4': C4,
                    'C5': C5
                    }

    pyramid_dict = {}
    with tf.variable_scope('build_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfg.SOLVER.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):

            conv_channels = 256
            last_fm = None
            for i in range(3):
                fm = feature_dict['C{}'.format(5-i)]
                fm_1x1_conv = slim.conv2d(fm,  num_outputs=conv_channels, kernel_size=[1, 1],
                                          stride=1, scope='p{}_1x1_conv'.format(5-i))
                if last_fm is not None:
                    h, w = tf.shape(fm_1x1_conv)[1], tf.shape(fm_1x1_conv)[2]
                    last_resize = tf.image.resize_bilinear(last_fm,
                                                           size=[h, w],
                                                           name='p{}_up2x'.format(5-i))

                    fm_1x1_conv = fm_1x1_conv + last_resize

                last_fm = fm_1x1_conv

                fm_3x3_conv = slim.conv2d(fm_1x1_conv,
                                          num_outputs=conv_channels, kernel_size=[3, 3], padding="SAME",
                                          stride=1, scope='p{}_3x3_conv'.format(5 - i))
                pyramid_dict['P{}'.format(5-i)] = fm_3x3_conv

            p6 = slim.conv2d(pyramid_dict['P5'],
                             num_outputs=conv_channels, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p6_conv')
            pyramid_dict['P6'] = p6

            p7 = tf.nn.relu(p6)

            p7 = slim.conv2d(p7,
                             num_outputs=conv_channels, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p7_conv')

            pyramid_dict['P7'] = p7

    return pyramid_dict


def fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)


def heads():
    return
