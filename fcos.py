import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, ReLU
from tensorflow.contrib.layers import group_norm
from tensorflow.contrib.distributions import Normal


class FCOSHead():

    def __init__(self, cfg, in_channels):
        super(FCOSHead, self).__init__()

        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []

        for _ in range(cfg.MODEL.FCOS.NUM_CONVS):
            conv = Conv2D(in_channels, kernel_size=3, strides=1,
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.constant_initializer())
            cls_tower.append(conv)
            cls_tower.append(group_norm(conv))
            cls_tower.append(ReLU())

            bbox_tower.append(conv)
            bbox_tower.append(group_norm(conv))
            bbox_tower.append(ReLU())

        self.cls_tower = Sequential(layers=cls_tower, name='cls_tower')
        self.bbx_tower = Sequential(layers=bbox_tower, name='bbox_tower')

        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -tf.log((1 - prior_prob) / prior_prob)

        self.cls_logit = Conv2D(num_classes, kernel_size=3, strides=1,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.constant_initializer(bias_value))
        self.bbox_pred = Conv2D(4, kernel_size=3, strides=1,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.constant_initializer())
        self.centerness = Conv2D(1, kernel_size=3, strides=1,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.constant_initializer())

    def __call__(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for _, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logit(cls_tower))
            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(tf.exp(self.bbox_pred(self.bbx_tower(feature))))

        return logits, bbox_reg, centerness


def fcos_postprocessor(cfg):
    return


def fcos_loss_evaluator(cfg):
    return


class FCOSModule():

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = fcos_postprocessor(cfg)
        loss_evaluator = fcos_loss_evaluator(cfg)
