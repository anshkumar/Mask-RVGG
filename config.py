"""
EMaskRCNN_V2
Base Configurations class.
"""

import numpy as np

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Backbone network architecture
    # Supported values are: 'resnet50', 'repVGG-A0', 'repVGG-A1', 'repVGG-A2', 'repVGG-B0', 'repVGG-B1', 'repVGG-B1g2', 'repVGG-B1g4', 'repVGG-B2', 'repVGG-B2g2', 'repVGG-B2g4', 'repVGG-B3', 'repVGG-B3g2', 'repVGG-B3g4'
    BACKBONE = "repVGG-B3g4"
    BASE_MODEL_TRAINABLE = True
    FREEZE_BACKBONE_BN = False

    BATCH_SIZE = 16 # Batch size per GPU
    # (Height, Width, Channels)
    # [512, 640, 768, 896, 1024, 1280, 1408]
    IMAGE_SHAPE = [512, 512, 3]

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    ANCHOR_RATIOS =  [ [[1, 1/2, 2]] ]*5

    # Length of square anchor side in pixels
    # ANCHOR_SCALES = [list(i*np.array([2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)])) for i in [16, 32, 64, 128, 256]]
    ANCHOR_SCALES = [[24.0, 30.238105197476955, 38.097625247236785], [48.0, 60.47621039495391, 76.19525049447357], [96.0, 120.95242078990782, 152.39050098894714], [192.0, 241.90484157981564, 304.7810019778943], [384.0, 483.8096831596313, 609.5620039557886]]

    # Weather to use FPN 
    FPN_FEATURE_MAP_SIZE = 256

    # Number of classification classes (excluding background)
    NUM_CLASSES = 90  # Override in sub-classes

    MAX_OUTPUT_SIZE = 100
    PER_CLASS_MAX_OUTPUT_SIZE = 100
    CONF_THRESH = 0.05
    NMS_THRESH = 0.5

    # Maximum number of ground truth instances to use in one image
    NUM_MAX_FIX_PADDING = 100 

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    PREDICT_MASK = False
    # Pooled ROIs
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    # TRAIN_BN = True 

    # Loss weights for more precise optimization.
    LOSS_WEIGHTS = {
        "loss_weight_cls": 1.,
        "loss_weight_box": 1.,
        "loss_weight_mask": 1.,
        "loss_weight_mask_iou": 1.,
    }

    # Allowed are : ['OHEM', 'FOCAL', 'CROSSENTROPY']
    LOSS_CLASSIFICATION = 'OHEM'
    ACTIVATION = 'SOFTMAX' # ['SOFTMAX', 'SIGMOID']
    NEG_POS_RATIO = 3
    USE_MASK_IOU = False

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    # Allowed optimizer: ['SGD', 'Adam', 'SGDW', 'AdamW', 'AdaBelief']
    OPTIMIZER = 'SGD'
    LEARNING_RATE = 4e-2
    N_WARMUP_STEPS = 1000
    WARMUP_LR = 0.0
    LEARNING_MOMENTUM = 0.9
    LR_SCHEDULE = False
    TRAIN_ITER = 800000 #88716 # 1x
    LR_TOTAL_STEPS = 800000 #88716

    # Weight decay regularization
    WEIGHT_DECAY = 1e-4 # 5*1e-4

    # Gradient norm clipping or AGC (Will use either one of them.)
    GRADIENT_CLIP_NORM = 10
    USE_AGC = False

    MATCH_THRESHOLD = 0.5
    UNMATCHED_THRESHOLD = 0.5

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("\n")
 
