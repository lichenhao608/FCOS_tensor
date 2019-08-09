import tensorflow as tf
import tensorflow_transform as tft

INF = 1e10


class IoUloss():
    def __call__(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
            (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = tf.minimum(pred_left, target_left) + \
            tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(
            pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        loss = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        if weight is not None and tft.sum(weight) > 0:
            return tft.sum(loss * weight) / tft.sum(weight)
        else:
            assert tf.size(loss) != 0
            return tft.mean(loss)


class SigmoidFocalLoss():
    def __init__(self, gamma, alpha):
        self.gamma = gamma
        self.alpha = alpha

    def __cal__(self, logits, targets):
        def loss_func():
            num_classes = logits.shape[1]
            gamma = self.gamma[0]
            alpha = self.alpha[0]
            dtype = targets.dtype
            class_range = tf.expand_dims(
                tf.range(1, limit=num_classes + 1, dtype=dtype), 0)

            t = tf.expand_dims(targets)
            p = tf.sigmoid(logits)
            term1 = (1 - p)**gamma * tf.log(p)
            term2 = p ** gamma * tf.log(1 - p)
            return -(t == class_range).as_type(tf.float64) * term1 * alpha - ((t != class_range) * (t >= 0)).as_type(tf.float64) * term2 * (1-alpha)
        loss = loss_func()
        return tft.sum(loss)


class FCOSLossEvaluator(object):
    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA, cfg.MODEL.FCOS.LOSS_ALPHA)
        self.boxreg_loss_func = IoUloss()
        self.centerness_loss_func = tf.nn.sigmoid_cross_entropy_with_logits()

    def prepare_target(self, points, targets):
        size_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF]
        ]

        expand_interest = []
        for i, point_per_level in enumerate(points):
            size_interest_per_level = tf.constant(size_of_interest[i])
            expand_interest.append(tf.broadcast_to(
                size_interest_per_level[None], [len(point_per_level), size_interest_per_level.shape[-1]]))

        expand_interest_list = tf.stack(expand_interest, axis=0)
        num_point_per_level = [len(point_per_level)
                               for point_per_level in points]
        points_all = tf.stack(points, axis=0)
        labels, reg_targets = self.compute_target_lc(
            points_all, targets, expand_interest_list)

        for i in range(len(labels)):
            labels[i] = tf.split(labels[i], num_point_per_level, axis=0)
            reg_targets[i] = tf.split(
                reg_targets[i], num_point_per_level, axis=0)

        label_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            label_level_first.append(
                tf.stack([label_per_im[level] for label_per_im in labels], axis=0))
            reg_targets_level_first.append(tf.stack(
                [reg_targets_per_im[level] for reg_targets_per_im in reg_targets], axis=0))

        return label_level_first, reg_targets_level_first

    def compute_target_lc(self, locations, targets, object_size_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = tf.stack([l, t, r, b], axis=2)

            is_in_box = reg_targets_per_im.min(axis=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(axis=2)[0]

            is_cared_in_level = (max_reg_targets_per_im >= object_size_of_interest[:, [0]]) & (
                max_reg_targets_per_im <= object_size_of_interest[:, [1]])

            lc_to_gt_area = tf.tile(area[None], [len(locations), 1])
            lc_to_gt_area[is_in_box == 0] = INF
            lc_to_gt_area[is_cared_in_level == 0] = INF

            lc_to_min_area, lc_to_gt_inds = lc_to_gt_area.min(axis=1)

            reg_targets_per_im = reg_targets_per_im[range(
                len(locations)), lc_to_gt_inds]
            labels_per_im = labels_per_im[lc_to_gt_inds]
            labels_per_im[lc_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_target(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(axis=-1)[0] / left_right.max(axis=-1)[0]) * (
            top_bottom.min(axis=-1)[0] / top_bottom.max(axis=-1)[0])
        return tf.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        N = box_cls[0].shape.as_list()[0]
        num_classes = box_cls[0].shape.as_list()[1]
        labels, reg_targets = self.prepare_target(locations, targets)

        box_cls_flat = []
        box_reg_flat = []
        centerness_flat = []
        labels_flat = []
        reg_targets_flat = []
        for l in range(len(labels)):
            box_cls_flat.append(tf.reshape(tf.transpose(
                box_cls[l], perm=[0, 2, 3, 1]), [-1, num_classes]))
            box_reg_flat.append(tf.reshape(tf.transpose(
                box_regression[l], perm=[0, 2, 3, 1]), [-1, 4]))
            labels_flat.append(tf.reshape(labels[l], [-1]))
            reg_targets_flat.append(tf.reshape(reg_targets[l], [-1, 4]))
            centerness_flat.append(tf.reshape(centerness[l], [-1]))


def fcos_loss_evaluator(cfg):
    return FCOSLossEvaluator(cfg)
