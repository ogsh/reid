import tensorflow as tf
from model.parser import parse_model


class FeedforwardNet(tf.keras.Model):
    def __init__(self, in_channels, config):
        super(FeedforwardNet, self).__init__()

        self._seq, self._out_channels = parse_model(in_channels=in_channels, config=config)

    def call(self, x, training=None, mask=None):
        out = self._seq(x)

        return out
