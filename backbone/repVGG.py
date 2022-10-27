import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.engine import training

AUTOTUNE = tf.data.experimental.AUTOTUNE

def conv_bn(out_channels, kernel_size, strides, groups=1):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                groups=groups,
                use_bias=False,
                name="conv",
            ),
            tf.keras.layers.BatchNormalization(name="bn"),
        ]
    )

class RepVGGBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        strides=1,
        dilation=1,
        groups=1,
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3

        self.nonlinearity = tf.keras.layers.ReLU()

        if deploy:
            self.rbr_reparam = tf.keras.layers.Conv2D(
                        filters=out_channels,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding="same",
                        dilation_rate=dilation,
                        groups=groups,
                        use_bias=True,
                    )
        else:
            self.rbr_identity = (
                tf.keras.layers.BatchNormalization()
                if out_channels == in_channels and strides == 1
                else None
            )
            self.rbr_dense = conv_bn(
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                out_channels=out_channels,
                kernel_size=1,
                strides=strides,
                groups=groups,
            )
            print("RepVGG Block, identity = ", self.rbr_identity)

    def call(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    #     for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(
                kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]]) # Kernel Shape: [H, W, C_i, C_o]. Padding to H,W on top, bottom, left and right.
            )

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, tf.keras.Sequential):
            kernel = branch.get_layer("conv").weights[0]
            running_mean = branch.get_layer("bn").moving_mean
            running_var = branch.get_layer("bn").moving_variance
            gamma = branch.get_layer("bn").gamma
            beta = branch.get_layer("bn").beta
            eps = branch.get_layer("bn").epsilon
        else:
            assert isinstance(branch, tf.keras.layers.BatchNormalization)
            if not hasattr(self, "id_tensor"):
                # For an identity block input and output channel are same.
                # For a non zero group kernel input weight channel dimension is divided by group size.
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((3, 3, input_dim, self.in_channels), dtype=np.float32) # Shape: [H, W, C_i, C_o]

                for i in range(self.in_channels):
                    kernel_value[1, 1, i % input_dim, i] = 1
                self.id_tensor = tf.convert_to_tensor(
                    kernel_value, dtype=np.float32
                )
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias

def _make_stage(override_groups_map, input_filters, output_filters, num_blocks, stride, cur_layer_idx, deploy, name):
    strides = [stride] + [1] * (num_blocks - 1)
    blocks = []
    for stride in strides:
        cur_groups = override_groups_map.get(cur_layer_idx, 1)
        blocks.append(
            RepVGGBlock(
                in_channels=input_filters,
                out_channels=output_filters,
                kernel_size=3,
                strides=stride,
                groups=cur_groups,
                deploy=deploy,
            )
        )
        cur_layer_idx += 1
    return tf.keras.Sequential(blocks, name=name), cur_layer_idx

def RepVGG(
            input_shape,
            num_blocks,
            num_classes=1000,
            width_multiplier=None,
            override_groups_map=None,
            deploy=False,
            include_preprocessing=True,
            include_top=True,
            model_name=None
            ):
        assert len(width_multiplier) == 4
        override_groups_map = override_groups_map or dict()

        assert 0 not in override_groups_map

        filters = min(64, int(64 * width_multiplier[0]))

        stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=filters,
            kernel_size=3,
            strides=2,
            deploy=deploy,
        )
        cur_layer_idx = 1
        stage1, cur_layer_idx = _make_stage(
            override_groups_map, 
            input_filters=filters,
            output_filters=int(64 * width_multiplier[0]), 
            num_blocks=num_blocks[0], 
            stride=2, 
            cur_layer_idx=cur_layer_idx, 
            deploy=deploy, 
            name='stage1',
        )
        stage2, cur_layer_idx = _make_stage(
            override_groups_map, 
            input_filters=int(64 * width_multiplier[0]),
            output_filters=int(128 * width_multiplier[1]), 
            num_blocks=num_blocks[1], 
            stride=2, 
            cur_layer_idx=cur_layer_idx, 
            deploy=deploy, 
            name='stage2',
        )
        stage3, cur_layer_idx = _make_stage(
            override_groups_map, 
            input_filters=int(128 * width_multiplier[1]),
            output_filters=int(256 * width_multiplier[2]), 
            num_blocks=num_blocks[2], 
            stride=2, 
            cur_layer_idx=cur_layer_idx, 
            deploy=deploy, 
            name='stage3',
        )
        stage4, cur_layer_idx = _make_stage(
            override_groups_map, 
            input_filters=int(256 * width_multiplier[2]),
            output_filters=int(512 * width_multiplier[3]), 
            num_blocks=num_blocks[3], 
            stride=2, 
            cur_layer_idx=cur_layer_idx, 
            deploy=deploy, 
            name='stage4',
        )

        bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1
        img_input = tf.keras.Input(shape=input_shape, name="input")
        x = img_input
        if include_preprocessing:
            num_channels = input_shape[bn_axis - 1]
            x = tf.keras.layers.Rescaling(scale=1. / 255)(x)
            x = tf.keras.layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229**2, 0.224**2, 0.225**2],
                axis=bn_axis,
            )(x)
        
        out0 = stage0(x)
        out1 = stage1(out0)
        out2 = stage2(out1)
        out3 = stage3(out2)
        out4 = stage4(out3)
        if include_top:
            out = tfa.layers.AdaptiveAveragePooling2D(output_size=1, name="avg_pool")(out4)
            out = tf.keras.layers.Dense(num_classes, name="predictions")(out)
        else:
            out = [out2, out3, out4]
        
        # Create model.
        model = training.Model(inputs=img_input, outputs=out, name=model_name)

        return model

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_A0'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_A1(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_A1'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_A2(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_A2'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B0(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B0'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B1(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B1'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B1g2(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B1g2'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B1g4(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B1g4'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B2(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B2'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B2g2(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B2g2'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B2g4(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B2g4'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B3(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B3'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B3g2(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG_B3g2'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )


def create_RepVGG_B3g4(input_shape, include_preprocessing=True, include_top=True, deploy=False, model_name='RepVGG-B3g4'):
    return RepVGG(
        input_shape=input_shape,
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
        deploy=deploy,
        include_preprocessing=include_preprocessing,
        include_top=include_top,
        model_name=model_name
    )

func_dict = {
    "RepVGG-A0": create_RepVGG_A0,
    "RepVGG-A1": create_RepVGG_A1,
    "RepVGG-A2": create_RepVGG_A2,
    "RepVGG-B0": create_RepVGG_B0,
    "RepVGG-B1": create_RepVGG_B1,
    "RepVGG-B1g2": create_RepVGG_B1g2,
    "RepVGG-B1g4": create_RepVGG_B1g4,
    "RepVGG-B2": create_RepVGG_B2,
    "RepVGG-B2g2": create_RepVGG_B2g2,
    "RepVGG-B2g4": create_RepVGG_B2g4,
    "RepVGG-B3": create_RepVGG_B3,
    "RepVGG-B3g2": create_RepVGG_B3g2,
    "RepVGG-B3g4": create_RepVGG_B3g4,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


def repvgg_model_convert(
    model: tf.keras.Model, build_func, save_path=None, image_size=(224, 224, 3)
):
    deploy_model = build_func(deploy=True)
    deploy_model.build(input_shape=(None, *image_size))
    for layer, deploy_layer in zip(model.layers, deploy_model.layers):
        if hasattr(layer, "repvgg_convert"):
            kernel, bias = layer.repvgg_convert()
            deploy_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.Sequential):
            assert isinstance(deploy_layer, tf.keras.Sequential)
            for sub_layer, deploy_sub_layer in zip(
                layer.layers, deploy_layer.layers
            ):
                kernel, bias = sub_layer.repvgg_convert()
                deploy_sub_layer.rbr_reparam.layers[1].set_weights(
                    [kernel, bias]
                )
        elif isinstance(layer, tf.keras.layers.Dense):
            assert isinstance(deploy_layer, tf.keras.layers.Dense)
            weights = layer.get_weights()
            deploy_layer.set_weights(weights)

    if save_path is not None:
        deploy_model.save_weights(save_path)

    return deploy_model
