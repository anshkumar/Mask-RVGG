import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import initializers
from tensorflow.keras import layers
import tensorflow_addons as tfa

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

class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, separable_conv=True, freeze_bn=False, detect_quadrangle=False, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        self.detect_quadrangle = detect_quadrangle
        num_values = 9 if detect_quadrangle else 4
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        if separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [WeightStandardizedSeparableConv(filters=width, name=f'{self.name}/box-{i}', **options) for i in
                          range(depth)]
            self.head = WeightStandardizedSeparableConv(filters=num_anchors * num_values,
                                               name=f'{self.name}/box-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [WeightStandardizedConv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = WeightStandardizedConv2D(filters=num_anchors * num_values, name=f'{self.name}/box-predict', **options)
        self.gns = [
            [tfa.layers.GroupNormalization(name=f'{self.name}/box-{i}-gn-{j}') for j in
             range(3, 8)]
            for i in range(depth)]
        # self.bns = [
        #     [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/box-{i}-bn-{j}') for j in
        #      range(3, 8)]
        #     for i in range(depth)]
        # self.bns = [[BatchNormalization(freeze=freeze_bn, name=f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)]
        #             for i in range(depth)]
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_values))

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
            outputs.append(output)
        return outputs
