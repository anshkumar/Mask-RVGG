import numpy as np
import tensorflow as tf
from utils import standard_fields
from utils import utils

class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, max_output_size, per_class_max_output_size, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.max_output_size = max_output_size
        self.per_class_max_output_size = per_class_max_output_size

    @tf.function
    def __call__(self, net_outs, trad_nms=False, use_cropped_mask=True):
        """
        Args:
             pred_offset: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            pred_cls: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            pred_mask_coef: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            priors: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_out: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.
            Note that the outputs are sorted only if cross_class_nms is False
        """

        box_p = net_outs['regression']
        class_p = net_outs['classification']
        anchors = net_outs['priors']  # [cx, cy, w, h] format. Normalized.

        num_class = tf.shape(class_p)[2] - 1

        # Apply softmax to the prediction class
        #class_p = tf.nn.softmax(class_p, axis=-1)

        # exclude the background class
        class_p = class_p[:, :, 1:]
        
        # get the max score class of 27429 predicted boxes
        class_p_max = tf.reduce_max(class_p, axis=-1)
        batch_size = tf.shape(class_p_max)[0]

        detection_boxes = tf.zeros((batch_size, self.max_output_size, 4), tf.float32)
        detection_classes = tf.zeros((batch_size, self.max_output_size), tf.float32)
        detection_scores = tf.zeros((batch_size, self.max_output_size), tf.float32)
        num_detections = tf.zeros((batch_size), tf.int32)

        for b in range(batch_size):
            # filter predicted boxes according the class score
            class_thre = tf.boolean_mask(class_p[b], class_p_max[b] > self.conf_thresh)
            raw_boxes = tf.boolean_mask(box_p[b], class_p_max[b] > self.conf_thresh)
            raw_anchors = tf.boolean_mask(anchors, class_p_max[b] > self.conf_thresh)

            # decode only selected boxes
            boxes = utils._decode(raw_boxes, raw_anchors)

            if tf.size(class_thre) > 0:
                if not trad_nms:
                    boxes, class_ids, class_thre = utils._cc_fast_nms(boxes, class_thre, iou_threshold=self.nms_thresh, top_k=self.max_output_size)
                else:
                    boxes, class_ids, class_thre = utils._traditional_nms(boxes, class_thre, score_threshold=self.conf_thresh, iou_threshold=self.nms_thresh, max_class_output_size=self.per_class_max_output_size, max_output_size=self.max_output_size)

                num_detection = [tf.shape(boxes)[0]]
                boxes = self._sanitize(boxes, width=1, height=1)

                _ind_boxes = tf.stack((tf.tile([b], num_detection), tf.range(0, tf.shape(boxes)[0])), axis=-1) # Shape: (Number of updates, index of update)
                detection_boxes = tf.tensor_scatter_nd_update(detection_boxes, _ind_boxes, boxes)
                detection_classes = tf.tensor_scatter_nd_update(detection_classes, _ind_boxes, class_ids)
                detection_scores = tf.tensor_scatter_nd_update(detection_scores, _ind_boxes, class_thre)
                num_detections = tf.tensor_scatter_nd_update(num_detections, [[b]], num_detection)

        result = {'detection_boxes': detection_boxes,'detection_classes': detection_classes, 'detection_scores': detection_scores, 'num_detections': num_detections}
        return result

    def _sanitize(self, boxes, width, height,  padding: int = 0):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """        
        x1, x2 = utils.sanitize_coordinates(boxes[:, 1], boxes[:, 3], tf.cast(width, dtype=tf.float32), normalized=False)
        y1, y2 = utils.sanitize_coordinates(boxes[:, 0], boxes[:, 2], tf.cast(height, dtype=tf.float32), normalized=False)

        boxes = tf.stack((y1, x1, y2, x2), axis=1)

        return boxes
