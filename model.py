import tensorflow as tf

# from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras import layers
from backbone import efficientnet_v2
# from backbone import efficientnet_v2_ws_gn as efficientnet_v2
from layers.biFPN import build_wBiFPN, build_BiFPN 
from layers.fpn import build_FPN
from layers.boxNet import BoxNet
from layers.classNet import ClassNet
from layers.maskNet import MaskHead
from layers.head import PredictionModule, FastMaskIoUNet
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

        backbones = {'efficientnetv2b0': efficientnet_v2.EfficientNetV2B0, 
                    'efficientnetv2b1': efficientnet_v2.EfficientNetV2B1, 
                    'efficientnetv2b2': efficientnet_v2.EfficientNetV2B2, 
                    'efficientnetv2b3': efficientnet_v2.EfficientNetV2B3, 
                    'efficientnetv2s': efficientnet_v2.EfficientNetV2S, 
                    'efficientnetv2m': efficientnet_v2.EfficientNetV2M, 
                    'efficientnetv2l': efficientnet_v2.EfficientNetV2L,
                    'resnet50': tf.keras.applications.resnet50.ResNet50}
        out_layers = {'efficientnetv2b0': ['block3b_add','block5e_add','block6h_add'],
                      'resnet50': ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
                    }

        if config.BACKBONE in ['resnet50']:
            base_model = backbones[config.BACKBONE](
                            include_top=False,
                            weights='imagenet',
                            input_shape=config.IMAGE_SHAPE,
                        )
        else:
            base_model = backbones[config.BACKBONE](
                                include_top=False,
                                weights='imagenet',
                                input_shape=config.IMAGE_SHAPE,
                                include_preprocessing=True
                            )

        # whether to freeze the convolutional base
        base_model.trainable = config.BASE_MODEL_TRAINABLE 

        # Freeze BatchNormalization in pre-trained backbone
        if config.FREEZE_BACKBONE_BN:
          for layer in base_model.layers:
              if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        outputs=[base_model.get_layer(x).output for x in out_layers[config.BACKBONE]]
        if not config.USE_FPN:
            if config.WEIGHTED_BIFPN:
                fpn_features = [None, None]+outputs
                for i in range(config.D_BIFPN):
                    fpn_features = build_wBiFPN(fpn_features, config.W_BIFPN, i)
            else:
                fpn_features = [None, None]+outputs
                for i in range(config.D_BIFPN):
                    fpn_features = build_BiFPN(fpn_features, config.W_BIFPN, i)
        else:
            fpn_features = build_FPN(outputs, config.FPN_FEATURE_MAP_SIZE)
        
        # extract certain feature maps for FPN
        self.backbone = tf.keras.Model(inputs=base_model.input,
                                       outputs=fpn_features)

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

        if not config.USE_FPN:
            self.box_net = BoxNet(config.W_BIFPN, config.D_HEAD, num_anchors=9, separable_conv=config.SEPARABLE_CONV, freeze_bn=config.FPN_FREEZE_BN,
                         detect_quadrangle=config.DETECT_QUADRANGLE, name='box_net')
            self.class_net = ClassNet(config.W_BIFPN, config.D_HEAD, num_classes=config.NUM_CLASSES+1, num_anchors=9,
                                 separable_conv=config.SEPARABLE_CONV, freeze_bn=config.FPN_FREEZE_BN, 
                                 activation=config.ACTIVATION, name='class_net')
        else:
            self.predictionHead = PredictionModule(config.FPN_FEATURE_MAP_SIZE, 9, config.NUM_CLASSES+1)

        self.num_anchors = anchorobj.num_anchors
        self.priors = anchorobj.anchors

        # post-processing for evaluation
        self.detect = Detect(config.NUM_CLASSES+1, max_output_size=config.MAX_OUTPUT_SIZE, 
            per_class_max_output_size=config.PER_CLASS_MAX_OUTPUT_SIZE,
            conf_thresh=config.CONF_THRESH, nms_thresh=config.NMS_THRESH)
        self.max_output_size = config.MAX_OUTPUT_SIZE
        self.num_classes = config.NUM_CLASSES
        self.config = config

    @tf.function()
    def call(self, inputs, training=False):
        inputs, gt_boxes = inputs[0], inputs[1]
        inputs = tf.cast(inputs, tf.float32)
        #image_norm = tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[np.square(0.299), np.square(0.224), np.square(0.225)])
        #features = self.backbone(image_norm(inputs/255), training=True)
        if self.config.BACKBONE == 'resnet50':
            inputs = tf.keras.applications.resnet50.preprocess_input(inputs)

        features = self.backbone(inputs, training=False)
        
        if not self.config.USE_FPN:
            classification = self.class_net(features)
            classification = layers.Concatenate(axis=1, name='classification')(classification)
            regression = self.box_net(features)
            regression = layers.Concatenate(axis=1, name='regression')(regression)
        else:
            # Prediction Head branch
            pred_cls = []
            pred_offset = []

            # all output from FPN use same prediction head
            for f_map in features:
                cls, offset = self.predictionHead(f_map)
                pred_cls.append(cls)
                pred_offset.append(offset)
                
            classification = tf.concat(pred_cls, axis=1)
            regression = tf.concat(pred_offset, axis=1)

        pred = {
            'regression': regression,
            'classification': classification,
            'priors': self.priors
        }

        pred.update(self.detect(pred, img_shape=tf.shape(inputs)))

        if self.config.PREDICT_MASK:
            if training:
                masks = self.mask_head(gt_boxes,
                                features[:-2],
                                self.num_classes,
                                self.config,
                                training)
            else:
                masks = self.mask_head(pred['detection_boxes'],
                                features[:-2],
                                self.num_classes,
                                self.config,
                                training)
            pred.update({'detection_masks': masks})

        return pred
