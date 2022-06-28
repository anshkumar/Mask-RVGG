import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import initializers
from tensorflow.keras import layers
import numpy as np
import math

MOMENTUM = 0.997
EPSILON = 1e-4

class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)

        return result

class ClassNet(models.Model):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, freeze_bn=False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                                 **options)
                          for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                        **options)
                          for i in range(depth)]
            self.head = layers.Conv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.bns = [
            [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/class-{i}-bn-{j}') for j
             in range(3, 8)]
            for i in range(depth)]
        # self.bns = [[BatchNormalization(freeze=freeze_bn, name=f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)]
        #             for i in range(depth)]
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_classes))
        self.activation = layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        self.level += 1
        return outputs
