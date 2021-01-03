import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D


def conv2d(filters, kernel_size, strides, padding, groups):
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  groups=groups,
                  use_bias=False,
                  kernel_initializer=tf.initializers.he_normal)


class ConvBNReLU(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, groups):
        super(ConvBNReLU, self).__init__()
        self._conv = conv2d(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            groups=groups)

        self._bn = BatchNormalization()
        self._relu = ReLU()

    def call(self, x, training=None, mask=None):
        y = self._conv(x)
        y = self._bn(y, training)
        y = self._relu(y)

        return y


def add_residual(x, residual):
    x_channels = x.shape[-1]
    r_channels = residual.shape[-1]
    if x_channels == r_channels:
        out = x + residual
    else:
        padding = [[0, 0], [0, 0], [0, 0], [0, x_channels - r_channels]]
        residual = tf.pad(residual, padding, "CONSTANT")
        out = x + residual

    return out


class ResBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super(ResBlock, self).__init__()
        self._bn1 = BatchNormalization()
        self._conv1 = conv2d(out_channels, kernel_size=kernel_size, strides=strides, padding=padding, groups=1)
        self._relu1 = ReLU()

        self._bn2 = BatchNormalization()
        self._conv2 = conv2d(out_channels, kernel_size=kernel_size, strides=strides, padding=padding, groups=1)
        self._relu2 = ReLU()

    def call(self, x, training=None, mask=None):
        residual = x
        y = self._bn1(x)
        y = self._relu1(y)
        y = self._conv1(y)
        y = self._bn2(y)
        y = self._relu2(y)
        y = self._conv2(y)

        y = add_residual(y, residual)

        return y


class DepthwiseSeparableConv(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self._depthwise = ConvBNReLU(filters=in_channels,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     groups=in_channels)
        self._pointwise = ConvBNReLU(filters=out_channels,
                                     kernel_size=kernel_size,
                                     strides=1,
                                     padding="SAME",
                                     groups=1)

    def call(self, x, training=None, mask=None):
        y = self._depthwise(x)
        y = self._pointwise(y)

        return y


class SpatialConcat(tf.keras.Model):
    def __init__(self, index_to_align_size):
        super(SpatialConcat, self).__init__()
        self._index_to_align_size = index_to_align_size
        self._pooling = tf.keras.layers.AveragePooling2D()

    def _pool(self, x, output_size):
        h_x = tf.shape(x)[1]
        h_target = output_size[0]
        n_pool = tf.experimental.numpy.log2(h_x / h_target)
        xshape = x.get_shape()
        out = x
        for _ in tf.range(n_pool):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(out,
                                  tf.TensorShape([xshape[0],
                                                  None,
                                                  None,
                                                  xshape[-1]]))])
            out = self._pooling(out)

        return out

    def call(self, xs, training=None, mask=None):
        output_size = tf.shape(xs[self._index_to_align_size])[1:3]
        xs = [self._pool(x, output_size) for x in xs]
        out = tf.concat(xs, axis=-1)

        return out


def resblock3x3(in_channels, out_channels):
    layer = ResBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     strides=1,
                     padding="SAME")

    return layer


def conv3x3(in_channels, out_channels, strides, padding, groups):
    layer = conv2d(filters=out_channels,
                   kernel_size=3,
                   strides=strides,
                   padding=padding,
                   groups=groups)

    return layer


def maxpool2x2(in_channels):
    layer = MaxPool2D(pool_size=2)

    return layer


def gap(in_channels):
    layer = tf.keras.layers.GlobalAveragePooling2D()

    return layer


def dense(in_channels, out_channels):
    layer = tf.keras.layers.Dense(units=out_channels)

    return layer


def spatial_concat(in_channels, out_channels, index_to_align_size):
    layer = SpatialConcat(index_to_align_size)

    return layer