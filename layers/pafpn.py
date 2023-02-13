import tensorflow as tf

def build_PAFPN(features, num_fpn_filters):
    c3, c4, c5 = features
    tmp_p4 = tf.concat((tf.image.resize(c5, [tf.shape(c4)[1],tf.shape(c4)[2]]), c4), axis=-1)

    p3 = tf.concat((tf.image.resize(tmp_p4, [tf.shape(c3)[1],tf.shape(c3)[2]]), c3), axis=-1)
    
    tmp_p3 = tf.keras.layers.Conv2D(num_fpn_filters[0], (3, 3), 2, padding="same", kernel_initializer=tf.keras.initializers.glorot_uniform(), activation="relu")(p3)
    
    p4 = tf.concat((tmp_p3, tmp_p4), axis=-1)

    tmp_p4 = tf.keras.layers.Conv2D(num_fpn_filters[1], (3, 3), 2, padding="same", kernel_initializer=tf.keras.initializers.glorot_uniform(), activation="relu")(p4)

    p5 = tf.concat((tmp_p4, c5), axis=-1)

    return [p3, p4, p5]
 
