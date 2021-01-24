import numpy as np
import tensorflow as tf


def create_record(image: np.ndarray, label: int, bb: np.ndarray):
    image = image
    bb = np.array(bb).astype(np.float32)
    height, width, channels = image.shape
    image_raw = image.tostring()
    bb_raw = bb.tostring()

    feature = {
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "channels": _int64_feature(channels),
        "label": _int64_feature(label),
        "image": _bytes_feature(image_raw),
        "bb": _bytes_feature(bb_raw),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example.SerializeToString()

    return serialized_example


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parse_record(example_proto):
    feature_description = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "bb": tf.io.FixedLenFeature([], tf.string)
    }
    feature = tf.io.parse_single_example(example_proto, feature_description)

    height = tf.cast(feature["height"], tf.int32)
    width = tf.cast(feature["width"], tf.int32)
    channels = tf.cast(feature["channels"], tf.int32)
    label = tf.cast(feature["label"], tf.int32)
    image_raw = tf.io.decode_raw(feature["image"], tf.uint8)
    bb = tf.io.decode_raw(feature["bb"], tf.float32)
    image = tf.reshape(image_raw, tf.stack([height, width, channels]))

    ret = {
        "image": image,
        "bb": bb,
        "label": label
    }

    return ret


def load_tfrecord(path_tfrecord):
    dataset = tf.data.TFRecordDataset(path_tfrecord)
    dataset = dataset.map(_parse_record)

    return dataset



