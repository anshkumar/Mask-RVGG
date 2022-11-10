import tensorflow as tf
import time
from utils import utils

class Loss(object):
    def __init__(self, config):
        self.img_h = config.IMAGE_SHAPE[0]
        self.img_w = config.IMAGE_SHAPE[1]
        self._loss_weight_cls = config.LOSS_WEIGHTS['loss_weight_cls']
        self._loss_weight_box = config.LOSS_WEIGHTS['loss_weight_box']
        self._loss_weight_mask = config.LOSS_WEIGHTS['loss_weight_mask']
        self._loss_weight_mask_iou = config.LOSS_WEIGHTS['loss_weight_mask_iou']
        self._neg_pos_ratio = config.NEG_POS_RATIO
        self.use_mask_iou = config.USE_MASK_IOU
        self.config = config

    def __call__(self, model, pred, label, num_classes, image = None):
        """
        :param num_classes: Number of classes including background
        :param anchors:
        :param label: labels dict from dataset
            all_offsets: the transformed box coordinate offsets of each pair of 
                      prior and gt box
            conf_gt: the foreground and background labels according to the 
                     'pos_thre' and 'neg_thre',
                     '0' means background, '>0' means foreground.
            prior_max_box: the corresponding max IoU gt box for each prior
            prior_max_index: the index of the corresponding max IoU gt box for 
                      each prior
        :param pred:
        :return:
        """
        self.image = image
        # all prediction component
        self.pred_cls = pred['classification']
        self.pred_offset = pred['regression']
        self.pred_bbox = pred['detection_boxes']
        if self.config.PREDICT_MASK:
            self.pred_mask = pred['detection_masks']

        # all label component
        self.gt_offset = label['all_offsets']
        self.conf_gt = label['conf_gt']
        self.gt_bbox = label['boxes_norm']
        self.prior_max_box = label['prior_max_box']
        self.prior_max_index = label['prior_max_index']

        if self.config.PREDICT_MASK:
            self.masks = label['mask_target']
        self.classes = label['classes']
        self.num_classes = num_classes
        self.model = model

        loc_loss = self._loss_location() 

        if self.config.LOSS_CLASSIFICATION == 'OHEM':
            conf_loss = self._loss_class_ohem() 
        elif self.config.LOSS_CLASSIFICATION == 'CROSSENTROPY':
            conf_loss = self._loss_class()
        else:
            conf_loss = self._focal_conf_loss()

        if self.config.PREDICT_MASK:
            mask_loss, mask_iou_loss = self._loss_mask() 
        else:
            mask_loss, mask_iou_loss = [0.0], [0.0]
        
        return loc_loss, conf_loss, mask_loss, mask_iou_loss

    def _loss_location(self, sigma=3.0):
        sigma_squared = sigma ** 2

        # only compute losses from positive samples
        # get postive indices
        pos_indices = tf.where(self.conf_gt > 0 )
        pred_offset = tf.gather_nd(self.pred_offset, pos_indices)
        gt_offset = tf.gather_nd(self.gt_offset, pos_indices)

        num_pos = tf.cast(tf.reduce_sum(tf.shape(gt_offset)[0]), tf.float32)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        # regression_diff = pred_offset - gt_offset
        # regression_diff = tf.keras.backend.abs(regression_diff)
        # loss_loc = tf.where(
        #     tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
        #     0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
        #     regression_diff - 0.5 / sigma_squared
        # )

        smoothl1loss = tf.keras.losses.Huber(delta=1., reduction=tf.keras.losses.Reduction.NONE)
        if tf.reduce_sum(tf.cast(num_pos, tf.float32)) > 0.0:
            loss_loc = smoothl1loss(gt_offset, pred_offset)
        else:
            loss_loc = 0.0
        tf.debugging.assert_all_finite(loss_loc, "Loss Location NaN/Inf")
        return [tf.math.divide_no_nan(tf.reduce_sum(loss_loc), num_pos)*self._loss_weight_box]

    def _focal_conf_loss(self, alpha=0.25, gamma=1.5):
        """
        Focal loss but using sigmoid like the original paper.
        """
        labels = tf.one_hot(self.conf_gt, depth=self.num_classes)
        # filter out "neutral" anchors
        indices = tf.where(self.conf_gt >= 0)
        labels = tf.gather_nd(labels, indices)
        pred_cls = tf.gather_nd(self.pred_cls, indices)

        # compute the focal loss
        alpha_factor = tf.keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(tf.keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(tf.keras.backend.equal(labels, 1), 1 - pred_cls, pred_cls)
        focal_weight = alpha_factor * focal_weight ** gamma
        loss = focal_weight * tf.keras.backend.binary_crossentropy(labels, pred_cls, from_logits=False)
                
        pos_indices = tf.where(self.conf_gt > 0 )
        num_pos = tf.shape(pos_indices)[0]
        return [tf.math.divide_no_nan(tf.reduce_sum(loss), tf.cast(num_pos, tf.float32))*self._loss_weight_cls]
        # return [tf.reduce_mean(loss)*self._loss_weight_cls]

    def _loss_class(self):
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE)

        loss_conf = scce(tf.cast(self.conf_gt, dtype=tf.int32), self.pred_cls, 
                            self._loss_weight_cls)
        pos_indices = tf.where(self.conf_gt > 0 )
        num_pos = tf.shape(pos_indices)[0]
        return [tf.math.divide_no_nan(tf.reduce_sum(loss_conf), tf.cast(num_pos, tf.float32))*self._loss_weight_cls]

    def _loss_class_ohem(self):
        # num_cls includes background
        batch_conf = tf.reshape(self.pred_cls, [-1, self.num_classes])

        # Hard Negative Mining
        # Using tf.nn.softmax or tf.math.log(tf.math.reduce_sum(tf.math.exp(batch_conf), 1)) to calculate log_sum_exp
        # might cause NaN problem. This is a known problem https://github.com/tensorflow/tensorflow/issues/10142
        # To get around this using tf.math.reduce_logsumexp and softmax_cross_entropy_with_logit

        # This will be used to determine unaveraged confidence loss across all examples in a batch.
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/modules/multibox_loss.py#L251
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/box_utils.py#L316
        # log_sum_exp = tf.math.log(tf.math.reduce_sum(tf.math.exp(batch_conf), 1))

        # Using inbuild reduce_logsumexp to avoid NaN
        # This function is more numerically stable than log(sum(exp(input))). It avoids overflows caused by taking the exp of large inputs and underflows caused by taking the log of small inputs.
        log_sum_exp = tf.math.reduce_logsumexp(batch_conf, 1)
        # tf.print(log_sum_exp)
        loss_c = log_sum_exp - batch_conf[:,0]

        loss_c = tf.reshape(loss_c, (tf.shape(self.pred_cls)[0], -1))  # (batch_size, 27429)
        pos_indices = tf.where(self.conf_gt > 0 )
        loss_c = tf.tensor_scatter_nd_update(loss_c, pos_indices, tf.zeros(tf.shape(pos_indices)[0])) # filter out pos boxes
        num_pos = tf.math.count_nonzero(tf.greater(self.conf_gt,0), axis=1, keepdims=True)
        num_neg = tf.clip_by_value(num_pos * self._neg_pos_ratio, clip_value_min=tf.constant(self._neg_pos_ratio, dtype=tf.int64), clip_value_max=tf.cast(tf.shape(self.conf_gt)[1]-1, tf.int64))

        neutrals_indices = tf.where(self.conf_gt < 0 )
        loss_c = tf.tensor_scatter_nd_update(loss_c, neutrals_indices, tf.zeros(tf.shape(neutrals_indices)[0])) # filter out neutrals (conf_gt = -1)

        idx = tf.argsort(loss_c, axis=1, direction='DESCENDING')
        idx_rank = tf.argsort(idx, axis=1)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        # Filter out neutrals and positive
        neg_indices = tf.where((tf.cast(idx_rank, dtype=tf.int64) < num_neg) & (self.conf_gt == 0))

        # neg_indices shape is (batch_size, no_prior)
        # pred_cls shape is (batch_size, no_prior, no_class)
        neg_pred_cls_for_loss = tf.gather_nd(self.pred_cls, neg_indices)
        neg_gt_for_loss = tf.gather_nd(self.conf_gt, neg_indices)
        pos_pred_cls_for_loss = tf.gather_nd(self.pred_cls, pos_indices)
        pos_gt_for_loss = tf.gather_nd(self.conf_gt, pos_indices)

        target_logits = tf.concat([pos_pred_cls_for_loss, neg_pred_cls_for_loss], axis=0)
        target_labels = tf.concat([pos_gt_for_loss, neg_gt_for_loss], axis=0)
        target_labels = tf.one_hot(tf.squeeze(target_labels), depth=self.num_classes)

        if tf.reduce_sum(tf.cast(num_pos, tf.float32)+tf.cast(num_neg, tf.float32)) > 0.0:
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE)
            loss_conf = tf.reduce_sum(cce(target_labels, target_logits)) / tf.reduce_sum(tf.cast(num_pos, tf.float32)+tf.cast(num_neg, tf.float32))
        else:
            loss_conf = 0.0
        return [loss_conf*self._loss_weight_cls]

    def _loss_mask(self):
        
        p_mask = tf.reshape(self.pred_mask, (-1, tf.shape(self.pred_mask)[2], tf.shape(self.pred_mask)[3], tf.shape(self.pred_mask)[4]))
        gt_mask = tf.reshape(self.masks, (-1, tf.shape(self.masks)[2], tf.shape(self.masks)[3]))
        # p_mask = tf.gather_nd(p_mask, iou_max_id)
        # gt_mask = tf.gather_nd(gt_mask, iou_max_id)
        classes = tf.reshape(self.classes, [-1])
        class_gt_id = tf.where(classes > 0)

        pos_p_masks = tf.gather_nd(p_mask, class_gt_id)
        pos_gt_masks = tf.gather_nd(gt_mask, class_gt_id)

        # crop gt mask
        gt_bbox = tf.reshape(self.gt_bbox, (-1,4))
        gt_bbox = tf.gather_nd(gt_bbox, class_gt_id)
        pos_gt_masks = tf.expand_dims(pos_gt_masks, axis=-1)
        pos_gt_masks = tf.image.crop_and_resize(pos_gt_masks, 
            boxes=gt_bbox,
            box_indices=tf.range(tf.shape(gt_bbox)[0]),
            crop_size=self.config.MASK_SHAPE)
        pos_gt_masks = tf.squeeze(pos_gt_masks)
        pos_gt_masks = tf.cast(pos_gt_masks + 0.5, tf.uint8)
        pos_gt_masks = tf.cast(pos_gt_masks, tf.float32)

        # Gathering positive mask from ground truth
        pos_classes = tf.gather_nd(classes, class_gt_id)-1
        pos_p_masks = tf.transpose(pos_p_masks, (3,0,1,2))
        _idx = tf.stack((pos_classes, tf.range(tf.shape(pos_classes)[0], dtype=tf.int64)),axis=1)
        pos_p_masks = tf.gather_nd(pos_p_masks, _idx)

        #Resizing to the input size
        pos_p_masks = tf.expand_dims(pos_p_masks, axis=-1)
        pos_p_masks = tf.image.resize(pos_p_masks, [self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1]], method=tf.image.ResizeMethod.BILINEAR)
        pos_p_masks = pos_p_masks[:, :, :, 0]

        cce = tf.keras.losses.BinaryCrossentropy(from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE)
        loss = cce(pos_gt_masks, pos_p_masks)
        tf.debugging.assert_all_finite(loss, "Loss Mask NaN/Inf")
        loss = tf.reduce_mean(loss)
        if tf.math.is_nan(loss):
            return [0.0], [0.0]
        else:
            return [loss*self._loss_weight_mask], [0.0]

        '''
        pred_bbox = tf.reshape(self.pred_bbox, (-1, 4))
        gt_bbox = tf.reshape(self.gt_bbox, (-1,4))
        iou = utils._iou(pred_bbox, gt_bbox)
        iou_max = tf.reduce_max(iou, axis=-1)
        iou_max_id = tf.where(iou_max > 0.5)
        
        classes = tf.reshape(self.classes, [-1])
        classes = tf.gather_nd(classes, iou_max_id)
        class_gt_id = tf.where(classes > 0)
        
        if tf.shape(iou_max_id)[0] == 0 or tf.shape(class_gt_id)[0] == 0:
            return [0.0], [0.0]
            p_mask = tf.reshape(self.pred_mask, (-1, tf.shape(self.pred_mask)[2], tf.shape(self.pred_mask)[3], tf.shape(self.pred_mask)[4]))
            gt_mask = tf.reshape(self.masks, (-1, tf.shape(self.masks)[2], tf.shape(self.masks)[3]))
            p_mask = tf.gather_nd(p_mask, iou_max_id)
            gt_mask = tf.gather_nd(gt_mask, iou_max_id)
            
            pred_bbox = tf.gather_nd(pred_bbox, iou_max_id)
            gt_mask = tf.expand_dims(gt_mask, axis=-1)
            gt_mask = tf.image.crop_and_resize(gt_mask, 
                boxes=pred_bbox,
                box_indices=tf.range(tf.shape(pred_bbox)[0]),
                crop_size=self.config.MASK_SHAPE)

            gt_mask = tf.squeeze(gt_mask)
            gt_mask = tf.cast(gt_mask + 0.5, tf.uint8)
            gt_mask = tf.cast(gt_mask, tf.float32)

            # Take only positive samples
            pos_p_masks = tf.gather_nd(p_mask, class_gt_id)
            pos_gt_masks = tf.gather_nd(gt_mask, class_gt_id)
            pos_classes = tf.gather_nd(classes, class_gt_id)-1

            pos_p_masks = tf.transpose(pos_p_masks, (3,0,1,2))
            _idx = tf.stack((pos_classes, tf.range(tf.shape(pos_classes)[0], dtype=tf.int64)),axis=1)
            pos_p_masks = tf.gather_nd(pos_p_masks, _idx)

            #Resizing to the input size
            pos_p_masks = tf.expand_dims(pos_p_masks, axis=-1)
            pos_p_masks = tf.image.resize(pos_p_masks, [self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1]], method=tf.image.ResizeMethod.BILINEAR)
            pos_p_masks = pos_p_masks[:, :, :, 0]

            loss = tf.keras.backend.binary_crossentropy(pos_gt_masks, pos_p_masks, from_logits=False)
            return [tf.reduce_mean(loss)*self._loss_weight_mask], [0.0]
        '''
        '''
        # Mask IOU loss
        if self.use_mask_iou:
            pos_mask_gt_area = tf.reduce_sum(pos_mask_gt, axis=(0,1))

            # Area threshold of 25 pixels
            select_indices = tf.where(pos_mask_gt_area > 25 ) 

            if tf.shape(select_indices)[0] == 0: # num_positives are zero
                continue

            _pos_prior_box = tf.gather_nd(_pos_prior_box, select_indices)
            mask_p = tf.gather(mask_p, tf.squeeze(select_indices), axis=-1)
            pos_mask_gt = tf.gather(pos_mask_gt, tf.squeeze(select_indices), 
                axis=-1)
            pos_class_gt = tf.gather_nd(pos_class_gt, select_indices)

            mask_p = tf.cast(mask_p + 0.5, tf.uint8)
            mask_p = tf.cast(mask_p, tf.float32)
            maskiou_t = self._mask_iou(mask_p, pos_mask_gt)

            if tf.size(maskiou_t) == 1:
                maskiou_t = tf.expand_dims(maskiou_t, axis=0)
                mask_p = tf.expand_dims(mask_p, axis=-1)

            maskiou_net_input_list.append(mask_p)
            maskiou_t_list.append(maskiou_t)
            class_t_list.append(pos_class_gt)

            maskiou_t = tf.concat(maskiou_t_list, axis=0)
            class_t = tf.concat(class_t_list, axis=0)
            maskiou_net_input = tf.concat(maskiou_net_input_list, axis=-1)

            maskiou_net_input = tf.transpose(maskiou_net_input, (2,0,1))
            maskiou_net_input = tf.expand_dims(maskiou_net_input, axis=-1)
            num_samples = tf.shape(maskiou_t)[0]
            # TODO: train random sample (maskious_to_train)

            maskiou_p = self.model.fastMaskIoUNet(maskiou_net_input)

            # Using index zero for class label.
            # Indices are K-dimensional. 
            # [number_of_selections, [1st_dim_selection, 2nd_dim_selection, ..., 
            #  kth_dim_selection]]
            indices = tf.concat(
                (
                    tf.expand_dims(tf.range((num_samples), 
                        dtype=tf.int64), axis=-1), 
                    tf.expand_dims(class_t-1, axis=-1)
                ), axis=-1)
            maskiou_p = tf.gather_nd(maskiou_p, indices)

            smoothl1loss = tf.keras.losses.Huber(delta=1.)
            loss_i = smoothl1loss(maskiou_t, maskiou_p)

            loss_iou += loss_i/tf.cast(num_samples, dtype=tf.float32)
        '''

    def _mask_iou(self, mask1, mask2):
        intersection = tf.reduce_sum(mask1*mask2, axis=(0, 1))
        area1 = tf.reduce_sum(mask1, axis=(0, 1))
        area2 = tf.reduce_sum(mask2, axis=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret
