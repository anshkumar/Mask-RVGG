import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)

class BatchNorm(keras.layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

class PyramidROIAlign(keras.layers.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def get_config(self):
        config = super(PyramidROIAlign, self).get_config()
        config['pool_shape'] = self.pool_shape
        return config

    def call(self, boxes, feature_maps, box_fpn_level, config):
        # boxes: Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        # feature_maps: List of feature maps from different level of the
        #               feature pyramid. Each is [batch, height, width, channels]

        # Loop through levels and apply ROI pooling to each.
        pooled = []
        box_batch_indices = []

        for level in range(len(feature_maps)):
            ix = tf.where(tf.equal(box_fpn_level, level+1))
            level_boxes = tf.gather_nd(boxes, ix)
            
            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[level], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))
            box_batch_indices.append(box_indices)
             
        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)
        box_batch_indices = tf.concat(box_batch_indices, axis=0)

        _, idx, _ = tf.unique_with_counts(box_batch_indices)
        pooled_batch = tf.dynamic_partition(pooled, idx, config.BATCH_SIZE)

        pooled_masked = []
        for pool in pooled_batch:
             num_padding = config.MAX_OUTPUT_SIZE - tf.shape(pool)[0]
             pad_pool = tf.zeros([num_padding, tf.shape(pool)[1], tf.shape(pool)[2], tf.shape(pool)[3]])
             pool = tf.concat([pool, pad_pool], axis=0)
             pooled_masked.append(pool)

        pooled_batch = [tf.expand_dims(pool, axis=0) for pool in pooled_masked]
        pooled_batch = tf.concat(pooled_batch, axis=0)

        return pooled_batch

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

class MaskHead(keras.layers.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - MASK_POOL_SIZE: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_shape: Image Dimensions
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, config, **kwargs):
        super(MaskHead, self).__init__(**kwargs)
        # ROI Pooling
        # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        self.roiAlign = PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],
                            name="roi_align_mask")

        # Conv layers   
        self.conv_1 = layers.TimeDistributed(layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")
        self.batch_norm_1 = layers.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')
        self.act_1 = layers.Activation('relu')

        self.conv_2 = layers.TimeDistributed(layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")
        self.batch_norm_2 = layers.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')
        self.act_2 = layers.Activation('relu')

        self.conv_3 = layers.TimeDistributed(layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")
        self.batch_norm_3 = layers.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')
        self.act_3 = layers.Activation('relu')

        self.conv_4 = layers.TimeDistributed(layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")
        self.batch_norm_4 = layers.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')
        self.act_4 = layers.Activation('relu')

        self.deconv = layers.TimeDistributed(layers.Conv2DTranspose(config.TOP_DOWN_PYRAMID_SIZE, (2, 2), strides=2, activation="relu"),
                            name="mrcnn_mask_deconv")

        self.use_bigger_mask = False
        # Use Bigger Mask Shapes
        if config.MASK_SHAPE[0] > 28:
            # Assert Sqaure Mask
            assert config.MASK_SHAPE[0] == config.MASK_SHAPE[1]
            # Assert Divisable by 28
            assert config.MASK_SHAPE[0] % 28 == 0
            self.use_bigger_mask = True
            self.extra_layers = keras.Sequential()
            for idx in  range(1 , int(math.log(config.MASK_SHAPE[0] // 28, 2)) + 1):
                self.extra_layers.add(layers.TimeDistributed(layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same"), name=f"mrcnn_mask_conv{4+idx}"))
                self.extra_layers.add(layers.TimeDistributed(BatchNorm(), name=f'mrcnn_mask_bn{4+idx}'))
                self.extra_layers.add(layers.Activation('relu'))
                self.extra_layers.add(layers.TimeDistributed(layers.Conv2DTranspose(config.TOP_DOWN_PYRAMID_SIZE, (2, 2), strides=2, activation="relu"), name=f"mrcnn_mask_deconv{idx+1}"))

        self.final_conv = layers.TimeDistributed(layers.Conv2D(config.NUM_CLASSES, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")

    def call(self, rois, feature_maps,
        num_classes, boxes_feature_level, config):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_shape: Image Dimensions
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        """
        # ROI Pooling
        # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        x = self.roiAlign(rois, feature_maps, boxes_feature_level, config)

        # Conv layers
        x = self.conv_1(x)
        x = self.batch_norm_1(x, training=config.TRAIN_BN)
        x = self.act_1(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x, training=config.TRAIN_BN)
        x = self.act_2(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x, training=config.TRAIN_BN)
        x = self.act_3(x)

        x = self.conv_4(x)
        x = self.batch_norm_4(x, training=config.TRAIN_BN)
        x = self.act_4(x)

        x = self.deconv(x)

        if self.use_bigger_mask:
            x = self.extra_layers(x, training=config.TRAIN_BN)

        x = self.final_conv(x)
        return x
