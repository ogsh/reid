import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, ReLU, Softmax, Conv2D, MaxPool2D
from model.layer import ConvBNReLU, ResBlock, conv2d

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class ResNet(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()

        self._backbone = tf.keras.Sequential(name="backbone")

        self._backbone.add(conv2d(filters=16, kernel_size=3, strides=1, padding="SAME", groups=1))
        self._backbone.add(MaxPool2D())
        self._backbone.add(ResBlock(in_channels, 32, 3, 1, "SAME"))
        self._backbone.add(MaxPool2D())
        self._backbone.add(ResBlock(64, 128, 3, 1, "SAME"))
        self._backbone.add(MaxPool2D())
        self._backbone.add(ResBlock(128, 256, 3, 1, "SAME"))
        self._backbone.add(MaxPool2D())
        self._backbone.add(ResBlock(256, 512, 3, 1, "SAME"))
        self._backbone.add(ResBlock(512, 1024, 3, 1, "SAME"))

        self._classifier = tf.keras.Sequential([
            GlobalAveragePooling2D(),
            Dense(units=out_channels),
        ], name="classifier")

    def call(self, x, training=None, mask=None):
        y = self._backbone(x)
        y = self._classifier(y)

        return y


