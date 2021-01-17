import tensorflow as tf
from model.parser import parse_model
from config.net import generate_net


class ReIDNet(tf.keras.Model):
    def __init__(self, in_channels, backbone, classifier, name):
        super(ReIDNet, self).__init__(name=name)

        self._backbone = generate_net(in_channels, backbone)
        self._classifier = generate_net(self._backbone.out_channels, classifier)
        self._out_channels = self._classifier.out_channels

    def call(self, x, training=None, mask=None):
        base_feature = self._backbone(x)

        out = self._classifier(base_feature)

        return out

    @property
    def out_channels(self):
        return self._out_channels