import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import initializers
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import math

MOMENTUM = 0.997
EPSILON = 1e-4

class WeightStandardizedConv2D(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="VALID",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )

class WeightStandardizedSeparableConv(layers.SeparableConv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="VALID",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )

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
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, freeze_bn=False, activation='SIGMOID', **kwargs):
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
            self.convs = [WeightStandardizedSeparableConv(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                                 **options)
                          for i in range(depth)]
            self.head = WeightStandardizedSeparableConv(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [WeightStandardizedConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                        **options)
                          for i in range(depth)]
            self.head = WeightStandardizedConv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.gns = [
            [tfa.layers.GroupNormalization(name=f'{self.name}/class-{i}-gn-{j}') for j
             in range(3, 8)]
            for i in range(depth)]
        # self.bns = [
        #     [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/class-{i}-bn-{j}') for j
        #      in range(3, 8)]
        #     for i in range(depth)]
        # self.bns = [[BatchNormalization(freeze=freeze_bn, name=f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)]
        #             for i in range(depth)]
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_classes))
        if activation == 'SIGMOID':
            self.activation = layers.Activation('sigmoid')
        else:
            self.activation = layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        features = inputs
        outputs = []
        for level, feature in enumerate(features):
            for i in range(self.depth):
                feature = self.convs[i](feature)
                feature = self.gns[i][level](feature)
                feature = self.relu(feature)
            output = self.head(feature)
            output = self.reshape(output)
            output = self.activation(output)
            outputs.append(output)
        return outputs
