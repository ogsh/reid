import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, ReLU, Softmax
from model.layer import ConvBNReLU, DepthwiseSeparableConv

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MobileNet(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(MobileNet, self).__init__()

        self._backbone = tf.keras.Sequential(name="backbone")

        self._backbone.add(DepthwiseSeparableConv(in_channels, 16, 3, 1, "SAME"))
        self._backbone.add(DepthwiseSeparableConv(16, 32, 3, 2, "VALID"))
        self._backbone.add(DepthwiseSeparableConv(32, 64, 3, 1, "SAME"))
        self._backbone.add(DepthwiseSeparableConv(64, 128, 3, 2, "VALID"))
        self._backbone.add(DepthwiseSeparableConv(128, 256, 3, 1, "SAME"))
        self._backbone.add(DepthwiseSeparableConv(256, 512, 3, 2, "SAME"))
        self._backbone.add(DepthwiseSeparableConv(512, 1024, 3, 2, "SAME"))

        self._classifier = tf.keras.Sequential([
            GlobalAveragePooling2D(),
            Dense(units=out_channels),
        ], name="classifier")

    @tf.function
    def call(self, x, training=None, mask=None):
        y = self._backbone(x)
        y = self._classifier(y)

        return y

