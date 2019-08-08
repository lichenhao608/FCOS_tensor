import tensorflow as tf
import keras
from fcos import FCOSModule
from keras import Sequential
from keras.layers import Conv2D
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1, resnet_utils


def backbone(cfg, img, scope_name='resnet_v1_101'):
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

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        with tf.variable_scope(scope_name, scope_name):
            net = resnet_utils.conv2d_same(
                img, cfg.MODEL.FCOS.NUM_CLASSES, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    feature_dict = {'C2': end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                    'C3': end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                    'C4': end_points_C4['{}/block3/unit_22/bottleneck_v1'.format(scope_name)],
                    'C5': end_points_C5['{}/block4/unit_3/bottleneck_v1'.format(scope_name)]
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
