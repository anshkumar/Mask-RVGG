import tensorflow as tf

class FastMaskIoUNet(tf.keras.layers.Layer):

    def __init__(self, num_class):
        super(FastMaskIoUNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), 2,
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(16, (3, 3), 2,
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), 2,
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           activation="relu")
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), 2,
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           activation="relu")
        self.conv5 = tf.keras.layers.Conv2D(128, (3, 3), 2,
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           activation="relu")
        self.conv6 = tf.keras.layers.Conv2D(num_class-1, (1, 1), 1,
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           activation="relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        maskiou_p = tf.reduce_max(x, axis=(1,2))

        return maskiou_p

class PredictionModule(tf.keras.layers.Layer):

    def __init__(self, out_channels, num_anchors, num_class, activation='SOFTMAX'):
        super(PredictionModule, self).__init__()
        self.num_anchors = num_anchors
        self.num_class = num_class
        self.activation = activation

        self.Conv = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same",
                                           kernel_initializer=  tf.keras.initializers.RandomNormal(stddev=0.01),
                                          )

        self.classConv = tf.keras.layers.Conv2D(self.num_class * self.num_anchors, (3, 3), 1, padding="same",
                                                kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.03),
                                              )

        self.boxConv = tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding="same",
                                              kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.03),
                                            )

    def call(self, p):
        p = self.Conv(p)
        p = tf.keras.activations.relu(p)

        pred_class = self.classConv(p)
        pred_box = self.boxConv(p)

        # pytorch input  (N,Cin,Hin,Win) 
        # tf input (N,Hin,Win,Cin) 
        # so no need to transpose like (0, 2, 3, 1) as in original yolact code

        # reshape the prediction head result for following loss calculation
        pred_class = tf.reshape(pred_class, [tf.shape(pred_class)[0], -1, self.num_class])
        pred_box = tf.reshape(pred_box, [tf.shape(pred_box)[0], -1, 4])
        if self.activation == 'SOFTMAX':
            pred_class = tf.nn.softmax(pred_class, axis=-1)
        else:
            pred_class = tf.math.sigmoid(pred_class)
        
        return pred_class, pred_box 
