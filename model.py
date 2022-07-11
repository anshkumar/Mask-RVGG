import tensorflow as tf

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
from tensorflow.keras import layers
from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule, FastMaskIoUNet
from layers.maskNet import MaskHead
assert tf.__version__.startswith('2')
from detection import Detect
from data import anchor
import numpy as np

class MaskED(tf.keras.Model):
    """
        Creating the MaskED Architecture
        Arguments:

    """

    def __init__(self, config):
        super(MaskED, self).__init__()

        backbones = {'efficientnetv2b0': EfficientNetV2B0, 
                    'efficientnetv2b1': EfficientNetV2B1, 
                    'efficientnetv2b2': EfficientNetV2B2, 
                    'efficientnetv2b3': EfficientNetV2B3, 
                    'efficientnetv2s': EfficientNetV2S, 
                    'efficientnetv2m': EfficientNetV2M, 
                    'efficientnetv2l': EfficientNetV2L}
        out_layers = {'efficientnetv2b0': ['block3b_add','block5e_add','block6h_add'],
                    }

        base_model = backbones[config.BACKBONE](
                            include_top=False,
                            weights='imagenet',
                            input_shape=config.IMAGE_SHAPE,
                            include_preprocessing=True
                        )

        # whether to freeze the convolutional base
        base_model.trainable = config.BASE_MODEL_TRAINABLE 

        # Freeze BatchNormalization in pre-trained backbone
        #if config.FREEZE_BACKBONE_BN:
        #    for layer in base_model.layers:
        #        if isinstance(layer, tf.keras.layers.BatchNormalization):
        #          layer.trainable = False

        outputs=[base_model.get_layer(x).output for x in out_layers[config.BACKBONE]]
        self.backbone_fpn = FeaturePyramidNeck(config.FPN_FEATURE_MAP_SIZE)
        self.predictionHead = PredictionModule(config.FPN_FEATURE_MAP_SIZE, sum(len(x)*len(config.ANCHOR_SCALES[0]) for x in config.ANCHOR_RATIOS[0]), 
                                               config.NUM_CLASSES+1)
        
        # extract certain feature maps for FPN
        self.backbone = tf.keras.Model(inputs=base_model.input,
                                       outputs=outputs)

        self.mask_head = MaskHead(config)

        # Calculating feature map size
        # https://stackoverflow.com/a/44242277/4582711
        # https://github.com/tensorflow/tensorflow/issues/4297#issuecomment-\
        # 246080982
        self.feature_map_size = np.array(
            [list(base_model.get_layer(x).output.shape[1:3]) for x in out_layers[config.BACKBONE]])
        out_height_p6 = np.ceil(
            (self.feature_map_size[-1, 0]).astype(np.float32) / float(2))
        out_width_p6  = np.ceil(
            (self.feature_map_size[-1, 1]).astype(np.float32) / float(2))
        out_height_p7 = np.ceil(out_height_p6 / float(2))
        out_width_p7  = np.ceil(out_width_p6/ float(2))
        self.feature_map_size = np.concatenate(
            (self.feature_map_size, 
            [[out_height_p6, out_width_p6], [out_height_p7, out_width_p7]]), 
            axis=0)

        anchorobj = anchor.Anchor(img_size_h=config.IMAGE_SHAPE[0],img_size_w=config.IMAGE_SHAPE[1],
                              feature_map_size=self.feature_map_size,
                              aspect_ratio=config.ANCHOR_RATIOS,
                              scale=config.ANCHOR_SCALES)

        self.num_anchors = anchorobj.num_anchors
        self.priors = anchorobj.anchors

        # post-processing for evaluation
        self.detect = Detect(config.NUM_CLASSES+1, max_output_size=config.MAX_OUTPUT_SIZE, 
            per_class_max_output_size=config.PER_CLASS_MAX_OUTPUT_SIZE,
            conf_thresh=config.CONF_THRESH, nms_thresh=config.NMS_THRESH)
        self.max_output_size = config.MAX_OUTPUT_SIZE
        self.num_classes = config.NUM_CLASSES
        self.config = config

    @tf.function
    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)

        c3, c4, c5 = self.backbone(inputs, training=True)  
        features = self.backbone_fpn(c3, c4, c5)    

        # Prediction Head branch
        pred_cls = []
        pred_offset = []
        boxes_feature_level = []

        # all output from FPN use same prediction head
        for i, feature in enumerate(features):
            cls, offset = self.predictionHead(feature)
            pred_cls.append(cls)
            pred_offset.append(offset)
            boxes_feature_level.append(tf.tile([[i+1]],[tf.shape(offset)[0], tf.shape(offset)[1]]))

        classification = tf.concat(pred_cls, axis=1, name='classification')
        regression = tf.concat(pred_offset, axis=1, name='regression')
        boxes_feature_level = tf.concat(boxes_feature_level, axis=1, name='boxes_feature_level')

        pred = {
            'regression': regression,
            'classification': classification,
            'boxes_feature_level': boxes_feature_level,
            'priors': self.priors
        }

        pred.update(self.detect(pred, img_shape=tf.shape(inputs)))

        pos_idx = tf.zeros(tf.shape(pred['detection_boxes'])[:2], dtype=tf.int32) 
        pos_range = [tf.range(i) for i in pred['num_detections']]

        pos_update_idx = [tf.stack((tf.tile([i], tf.shape(j)), j), axis=1) for i,j in zip(range(tf.shape(pos_idx)[0]), pos_range)]

        pos_update_idx = tf.concat(pos_update_idx, axis=0)
        pos_idx = tf.tensor_scatter_nd_update(pos_idx, pos_update_idx, tf.tile([1], [tf.reduce_sum(pred['num_detections'])]))
        masks = self.mask_head(pred['detection_boxes'],
                        pos_idx,
                        features[:-2],
                        self.num_classes,
                        self.config)
        pred.update({'detection_masks': masks})

        return pred
