import tensorflow as tf
from model.parser import parse_model
from config.net import generate_net


class ReIDNet(tf.keras.Model):
    def __init__(self, in_channels, backbone, classifier):
        super(ReIDNet, self).__init__()

        self._backbone, channels = generate_net(in_channels, backbone)
        self._classifier, self._out_channels = generate_net(channels, classifier)

    def call(self, x, training=None, mask=None):
        base_feature = self._backbone(x)

        out = self._classifier(base_feature)

        return out

    @property
    def out_channels(self):
        return self._out_channels