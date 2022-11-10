"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os

import tensorflow as tf

from data import anchor
from data.parser import Parser

# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)
def prepare_dataloader(config, tfrecord_dir, feature_map_size, batch_size, subset="train"):

    anchorobj = anchor.Anchor(img_size_h=config.IMAGE_SHAPE[0],img_size_w=config.IMAGE_SHAPE[1],
                              feature_map_size=feature_map_size,
                              aspect_ratio=config.ANCHOR_RATIOS,
                              scale=config.ANCHOR_SCALES)

    parser = Parser(config, anchor_instance=anchorobj, mode=subset)
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "*.*"))
    num_shards = tf.cast(tf.shape(files)[0], tf.int64)
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(num_shards)
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=num_shards,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
