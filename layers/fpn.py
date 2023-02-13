import tensorflow as tf

def build_FPN(features, num_fpn_filters):
    c3, c4, c5 = features
    p5 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
        kernel_initializer=tf.keras.initializers.glorot_uniform())(c5)

    tmp_p4 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
        kernel_initializer=tf.keras.initializers.glorot_uniform())(c4)
    p4 = tf.add(tf.image.resize(p5, [tf.shape(c4)[1],tf.shape(c4)[2]]), tmp_p4)

    tmp_c3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
        kernel_initializer=tf.keras.initializers.glorot_uniform())(c3)
    p3 = tf.add(tf.image.resize(p4, [tf.shape(c3)[1],tf.shape(c3)[2]]), tmp_c3)

    p3 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
        kernel_initializer=tf.keras.initializers.glorot_uniform(),
        activation="relu")(p3)
    p4 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
        kernel_initializer=tf.keras.initializers.glorot_uniform(),
        activation="relu")(p4)
    p5 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
        kernel_initializer=tf.keras.initializers.glorot_uniform(),
        activation="relu")(p5)

    return [p3, p4, p5]
 
