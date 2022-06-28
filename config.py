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
    # Supported values are: efficientnetb0, efficientnetb1, efficientnetb2,
    #         efficientnetb3, efficientnetb4, efficientnetb5, efficientnetb6
    #		  efficientnetv2s, efficientnetv2m, efficientnetv2l
    BACKBONE = "efficientnetv2b0"
    BASE_MODEL_TRAINABLE = True
    FREEZE_BACKBONE_BN = True

    BATCH_SIZE = 8 # Batch size per GPU
    # (Height, Width, Channels)
    # [512, 640, 768, 896, 1024, 1280, 1408]
    IMAGE_SHAPE = [512, 512, 3]

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    ANCHOR_RATIOS =  [ [[1, 1/2, 2]] ]*5

    # Length of square anchor side in pixels
    ANCHOR_SCALES = [[24.0, 30.238105197476955, 38.097625247236785], [48.0, 60.47621039495391, 76.19525049447357], [96.0, 120.95242078990782, 152.39050098894714], [192.0, 241.90484157981564, 304.7810019778943], [384.0, 483.8096831596313, 609.5620039557886]]

    # [64, 88, 112, 160, 224, 288, 384]
    W_BIFPN = 64

    # [3, 4, 5, 6, 7, 7, 8]
    D_BIFPN = 3

    # [3, 3, 3, 4, 4, 4, 5]
    D_HEAD = 3

    WEIGHTED_BIFPN = False

    FPN_FREEZE_BN = False

    SEPARABLE_CONV = True

    DETECT_QUADRANGLE = False

    # Number of classification classes (excluding background)
    NUM_CLASSES = 90  # Override in sub-classes

    MAX_OUTPUT_SIZE = 300
    PER_CLASS_MAX_OUTPUT_SIZE = 100
    CONF_THRESH = 0.05
    NMS_THRESH = 0.5

    # Maximum number of ground truth instances to use in one image
    NUM_MAX_FIX_PADDING = 300 

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Pooled ROIs
    MASK_POOL_SIZE = 7

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = True  # Defaulting to False since batch size is often small

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "loss_weight_cls": 1.,
        "loss_weight_box": 1.,
        "loss_weight_mask": 1.,
        "loss_weight_mask_iou": 1.,
    }

    # Allowed are : ['OHEM', 'FOCAL', 'CROSSENTROPY']
    LOSS_CLASSIFICATION = 'FOCAL'

    NEG_POS_RATIO = 3
    MAX_MASKS_FOR_TRAIN = 100
    USE_MASK_IOU = False

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    # Allowed optimizer: ['SGD', 'Adam', 'SGDW', 'AdamW']
    OPTIMIZER = 'Adam'
    LEARNING_RATE = 1e-3
    N_WARMUP_STEPS = 500
    WARMUP_LR = 1e-4
    LEARNING_MOMENTUM = 0.9
    LR_SCHEDULE = False
    TRAIN_ITER = 800000
    LR_TOTAL_STEPS = 800000

    # Weight decay regularization
    WEIGHT_DECAY = 5*1e-4

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 10.0

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
 
