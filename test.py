import tensorflow as tf
from layers.biFPN import build_wBiFPN, build_BiFPN 
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras import layers
from layers.boxNet import BoxNet
from layers.classNet import ClassNet
from layers.maskNet import MaskHead
from config import Config
import numpy as np
from IPython import embed

config = Config()
base_model = EfficientNetV2B0(
                            include_top=False,
                            weights='imagenet',
                            input_shape=(512,512,3),
                            include_preprocessing=True
                        )

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
      layer.trainable = False

out = ['block3b_add','block5e_add','block6h_add']
outputs=[base_model.get_layer(x).output for x in out]

fpn_features = [None, None]+outputs
for i in range(3):
    fpn_features = build_wBiFPN(fpn_features, 64, i, freeze_bn=False)

backbone = tf.keras.Model(inputs=base_model.input,
                           outputs=fpn_features)

box_net = BoxNet(64, 3, num_anchors=9, separable_conv=True, freeze_bn=False,
             detect_quadrangle=False, name='box_net')
class_net = ClassNet(64, 3, num_classes=91, num_anchors=9,
                     separable_conv=True, freeze_bn=False, name='class_net')

feature_map_size = np.array(
    [list(base_model.get_layer(x).output.shape[1:3]) for x in out])
out_height_p6 = np.ceil(
    (feature_map_size[-1, 0]).astype(np.float32) / float(2))
out_width_p6  = np.ceil(
    (feature_map_size[-1, 1]).astype(np.float32) / float(2))
out_height_p7 = np.ceil(out_height_p6 / float(2))
out_width_p7  = np.ceil(out_width_p6/ float(2))
feature_map_size = np.concatenate(
    (feature_map_size, 
    [[out_height_p6, out_width_p6], [out_height_p7, out_width_p7]]), 
    axis=0)

inputs = np.random.randn(4,512,512,3)
features = backbone(inputs)

classification = [class_net([feature, i]) for i, feature in enumerate(features)]
classification = layers.Concatenate(axis=1, name='classification')(classification)
regression = [box_net([feature, i]) for i, feature in enumerate(features)]
regression = layers.Concatenate(axis=1, name='regression')(regression)

mask_head = MaskHead(config)
mask = mask_head(np.random.randn(4, 300, 4),
        features,
        90,
        np.random.randint(1,4,size=(4,300)),
        config)

embed()
