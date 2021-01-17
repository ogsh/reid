import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, DepthwiseConv2D


def conv2d(filters, kernel_size, strides, padding, groups):
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  groups=groups,
                  use_bias=False,
                  kernel_initializer=tf.initializers.he_normal())


def depthise_conv2d(kernel_size, strides, padding):
    return DepthwiseConv2D(kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           use_bias=False,
                           depthwise_initializer=tf.initializers.he_normal())


class SequentialModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(SequentialModel, self).__init__(args, kwargs)
        self._seq = None

    def call(self, x, training=None, mask=None):
        out = self._seq(x, training=training, mask=mask)

        return out


class ConvBNAct(SequentialModel):
    def __init__(self, filters, kernel_size, strides, padding, groups, act):
        super(ConvBNAct, self).__init__()
        self._seq = tf.keras.Sequential(name="ConvBnAct")

        self._seq.add(conv2d(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             groups=groups))

        self._seq.add(BatchNormalization())

        if act is not None:
            self._seq.add(act)


class ConvBNReLU(ConvBNAct):
    def __init__(self, filters, kernel_size, strides, padding, groups):
        super(ConvBNReLU, self).__init__(filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         groups=groups,
                                         act=ReLU())


class DepthwiseConvBNReLU(SequentialModel):
    def __init__(self, kernel_size, strides, padding):
        super(DepthwiseConvBNReLU, self).__init__()
        conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               use_bias=False,
                                               depthwise_initializer=tf.keras.initializers.he_normal())
        bn = BatchNormalization()
        relu = ReLU()
        self._seq = tf.keras.Sequential(layers=[conv, bn, relu], name="DepthwiseConvBNReLU")


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
    def __init__(self, conv1, conv2):
        super(ResBlock, self).__init__()
        self._bn1 = BatchNormalization()
        self._conv1 = conv1
        self._relu1 = ReLU()

        self._bn2 = BatchNormalization()
        self._conv2 = conv2
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


class ResConv(ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size, strides):
        conv1 = conv2d(out_channels, kernel_size=kernel_size, strides=strides, padding="SAME", groups=1)
        conv2 = conv2d(out_channels, kernel_size=kernel_size, strides=strides, padding="SAME", groups=1)

        super(ResConv, self).__init__(conv1=conv1, conv2=conv2)


class ResidualMobile(ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size, strides):
        conv1 = depthise_conv2d(kernel_size=kernel_size, strides=strides, padding="SAME")
        conv2 = conv2d(filters=out_channels, kernel_size=1, strides=1, padding="SAME", groups=1)

        super(ResidualMobile, self).__init__(conv1=conv1, conv2=conv2)


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


class PyramidGAP(tf.keras.Model):
    def __init__(self):
        super(PyramidGAP, self).__init__()
        self._gap = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, xs, training=None, mask=None):
        xs = [self._gap(x) for x in xs]
        out = tf.concat(xs, axis=-1)

        return out


def resblock3x3(in_channels, out_channels):
    layer = ResConv(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    strides=1)

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


def pyramid_gap(in_channels, out_channels):
    layer = PyramidGAP()

    return layer


def resmobile3x3(in_channels, out_channels, strides):
    layer = ResidualMobile(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           strides=strides)

    return layer


def rescale(in_channels, out_channels, scale, offset):
    layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=scale, offset=offset)

    return layer
