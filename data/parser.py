import tensorflow as tf

from data import tfrecord_decoder
from utils import augmentation
from functools import partial

class Parser(object):

    def __init__(self,
                 config,
                 anchor_instance,
                 mode=None):

        self._mode = mode
        self._is_training = (mode == "train")

        self._example_decoder = tfrecord_decoder.TfExampleDecoder()

        self.config = config
        self._output_size_h = config.IMAGE_SHAPE[0]
        self._output_size_w = config.IMAGE_SHAPE[1]
        self._anchor_instance = anchor_instance
        self._match_threshold = config.MATCH_THRESHOLD
        self._unmatched_threshold = config.UNMATCHED_THRESHOLD

        # output related
        # for classes and mask to be padded to fix length
        self._num_max_fix_padding = config.NUM_MAX_FIX_PADDING

        # Data is parsed depending on the model.
        if mode == "train":
            self._parse_fn = partial(self._parse, augment=True)
        elif mode == "val":
            self._parse_fn = partial(self._parse, augment=False)
        elif mode == "test":
            self._parse_fn = self._parse_predict_data
        else:
            raise ValueError('mode is not defined.')

    def __call__(self, value):
        with tf.name_scope('parser'):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse(self, data, augment):
        classes = data['gt_classes']
        boxes = data['gt_bboxes'] # [ymin, xmin, ymax, xmax ]
        masks = data['gt_masks']
        is_crowds = data['gt_is_crowd']
        image = data['image']
        
        # ignore crowd annotation
        # https://github.com/AlexeyAB/darknet/issues/5567#issuecomment-626758944
        non_crowd_idx = tf.where(tf.logical_not(is_crowds))[:, 0]
        classes = tf.gather(classes, non_crowd_idx)
        boxes = tf.gather(boxes, non_crowd_idx)
        masks = tf.gather(masks, non_crowd_idx)
        
        # resize mask
        masks = tf.cast(masks, tf.bool)
        masks = tf.cast(masks, tf.float32)

        # Mask values should only be either 0 or 1
        masks = tf.cast(masks + 0.5, tf.uint8)

        # data augmentation randomly
        if augment:
            image, boxes, masks, classes = augmentation.random_augmentation(
                image, boxes, masks, [self._output_size_h, self._output_size_w],
                classes, self.config)

        image = tf.image.resize(image, 
            [self._output_size_h, self._output_size_w])
        masks = tf.expand_dims(masks, axis=-1)
        masks = tf.image.resize(masks, 
            [self._output_size_h, self._output_size_w])
        masks = tf.image.resize(masks, 
            [self._output_size_h, self._output_size_w])

        # masks = tf.image.crop_and_resize(masks, 
        #     boxes=boxes,
        #     box_indices=tf.range(tf.shape(boxes)[0]),
        #     crop_size=self.config.MASK_SHAPE)

        masks = tf.squeeze(masks)
        masks = tf.cast(masks + 0.5, tf.uint8)
        masks = tf.cast(masks, tf.float32)
        
        boxes_norm = boxes
        
        # matching anchors
        all_offsets, conf_gt, prior_max_box, prior_max_index = \
        self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, boxes_norm, classes, self.config)

        # number of object in training sample
        num_obj = tf.size(classes)

        # Padding classes and mask to fix length [None, num_max_fix_padding,...]
        num_padding = self._num_max_fix_padding - tf.shape(classes)[0]
        pad_classes = tf.zeros([num_padding], dtype=tf.int64)
        pad_boxes = tf.zeros([num_padding, 4])
        pad_masks = tf.zeros([num_padding, self._output_size_h, 
            self._output_size_w])
        boxes_norm = tf.concat([boxes_norm, pad_boxes], axis=0)

        if tf.shape(classes)[0] == 1:
            masks = tf.expand_dims(masks, axis=0)

        masks = tf.concat([masks, pad_masks], axis=0)
        classes = tf.concat([classes, pad_classes], axis=0)

        labels = {
            'all_offsets': all_offsets,
            'conf_gt': conf_gt,
            'prior_max_box': prior_max_box,
            'prior_max_index': prior_max_index,
            'boxes_norm': boxes_norm,
            'classes': classes,
            'num_obj': num_obj,
            'mask_target': masks,
        }
        return image, labels

    def _parse_predict_data(self, data):
        pass
