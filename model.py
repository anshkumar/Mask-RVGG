import tensorflow as tf
from backbone import repVGG
from layers.fpn import build_FPN
from layers.maskNet import MaskHead
from layers.head import PredictionModule
assert tf.__version__.startswith('2')
from detection import Detect
from data import anchor
import numpy as np

class MaskRVGG(tf.keras.Model):
    """
        Creating the Mask-RVGG Architecture
        Arguments:

    """

    def __init__(self, config, base_model=None, deploy=False):
        super(MaskRVGG, self).__init__()

        backbones = {
                    'resnet50': tf.keras.applications.resnet50.ResNet50,
                    'repVGG-A0': repVGG.create_RepVGG_A0,
                    'repVGG-A1': repVGG.create_RepVGG_A1,
                    'repVGG-A2': repVGG.create_RepVGG_A2,
                    'repVGG-B0': repVGG.create_RepVGG_B0,
                    'repVGG-B1': repVGG.create_RepVGG_B1,
                    'repVGG-B1g2': repVGG.create_RepVGG_B1g2,
                    'repVGG-B1g4': repVGG.create_RepVGG_B1g4,
                    'repVGG-B2': repVGG.create_RepVGG_B2,
                    'repVGG-B2g2': repVGG.create_RepVGG_B2g2,
                    'repVGG-B2g4': repVGG.create_RepVGG_B2g4,
                    'repVGG-B3': repVGG.create_RepVGG_B3,
                    'repVGG-B3g2': repVGG.create_RepVGG_B3g2,
                    'repVGG-B3g4': repVGG.create_RepVGG_B3g4,
                    }
        out_layers = {
                        'resnet50': ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'],
                        'repVGG-A0': ['stage2', 'stage3', 'stage4'],
                        'repVGG-A1': ['stage2', 'stage3', 'stage4'],
                        'repVGG-A2': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B0': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B1': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B1g2': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B1g4': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B2': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B2g2': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B2g4': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B3': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B3g2': ['stage2', 'stage3', 'stage4'],
                        'repVGG-B3g4': ['stage2', 'stage3', 'stage4']
                    }

        if config.BACKBONE in ['resnet50']:
            self.base_model = backbones[config.BACKBONE](
                            include_top=False,
                            weights='imagenet',
                            input_shape=config.IMAGE_SHAPE,
                        )
            outputs=[self.base_model.get_layer(x).output for x in out_layers[config.BACKBONE]]
        else:
            if deploy:
                self.base_model = backbones[config.BACKBONE](
                                input_shape=config.IMAGE_SHAPE,
                                include_preprocessing=True,
                                include_top=False,
                                deploy=True)

                for layer, deploy_layer in zip(base_model.base_model.layers, self.base_model.layers):
                    if hasattr(layer, "repvgg_convert"):
                        kernel, bias = layer.repvgg_convert()
                        deploy_layer.rbr_reparam.set_weights([kernel, bias])
                    elif isinstance(layer, tf.keras.Sequential):
                        assert isinstance(deploy_layer, tf.keras.Sequential)
                        for sub_layer, deploy_sub_layer in zip(
                            layer.layers, deploy_layer.layers
                        ):
                            kernel, bias = sub_layer.repvgg_convert()
                            deploy_sub_layer.rbr_reparam.set_weights(
                                [kernel, bias]
                            )
                    elif isinstance(layer, tf.keras.layers.Dense):
                        assert isinstance(deploy_layer, tf.keras.layers.Dense)
                        weights = layer.get_weights()
                        deploy_layer.set_weights(weights)
            else:
                self.base_model = backbones[config.BACKBONE](
                                input_shape=config.IMAGE_SHAPE,
                                include_preprocessing=True,
                                include_top=False,)
            outputs=self.base_model.output

        # whether to freeze the convolutional base
        self.base_model.trainable = config.BASE_MODEL_TRAINABLE 

        # Freeze BatchNormalization in pre-trained backbone
        if config.FREEZE_BACKBONE_BN:
          for layer in self.base_model.layers:
              if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        self.fpn_features = build_FPN(outputs, config.FPN_FEATURE_MAP_SIZE)
        
        # extract certain feature maps for FPN
        self.backbone = tf.keras.Model(inputs=self.base_model.input,
                                       outputs=self.fpn_features)

        self.mask_head = MaskHead(config)

        # Calculating feature map size
        # https://stackoverflow.com/a/44242277/4582711
        # https://github.com/tensorflow/tensorflow/issues/4297#issuecomment-\
        # 246080982
        self.feature_map_size = np.array(
            [list(self.base_model.get_layer(x).output.shape[1:3]) for x in out_layers[config.BACKBONE]])
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


        self.predictionHead = PredictionModule(config.FPN_FEATURE_MAP_SIZE, 9, config.NUM_CLASSES+1)

        self.num_anchors = anchorobj.num_anchors
        self.priors = anchorobj.anchors

        # post-processing for evaluation
        self.detect = Detect(config.NUM_CLASSES+1, max_output_size=config.MAX_OUTPUT_SIZE, 
            per_class_max_output_size=config.PER_CLASS_MAX_OUTPUT_SIZE,
            conf_thresh=config.CONF_THRESH, nms_thresh=config.NMS_THRESH,
            include_variances=config.INCLUDE_VARIANCES)
        self.max_output_size = config.MAX_OUTPUT_SIZE
        self.num_classes = config.NUM_CLASSES
        self.config = config

    @tf.function()
    def call(self, inputs, training=False):
        inputs, gt_boxes = inputs[0], inputs[1]

        if self.config.BACKBONE == 'resnet50':
            inputs = tf.keras.applications.resnet50.preprocess_input(inputs)

        features = self.backbone(inputs, training=False)
        
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

        pred.update(self.detect(pred, trad_nms=self.config.TRAD_NMS))

        if self.config.PREDICT_MASK:
            masks = self.mask_head(gt_boxes,
                            features[:-2],
                            self.num_classes,
                            self.config,
                            training)
            pred.update({'detection_masks': masks})

        return pred
