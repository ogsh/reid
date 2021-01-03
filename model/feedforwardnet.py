import tensorflow as tf
from model.parser import parse_model


class FeedforwardNet(tf.keras.Model):
    def __init__(self, in_channels, config):
        super(FeedforwardNet, self).__init__()

        self._layer_dict, self._out_channels = parse_model(in_channels=in_channels, config=config)

    @tf.function
    def call(self, x, training=None, mask=None):
        out = x
        out_dict = {}
        for name, layer in self._layer_dict.items():
            if layer["in_layers"] is None:
                out = layer["module"](out)
                out_dict[name] = out
            else:
                if isinstance(layer["in_layers"], list):
                    this_in_layers = [out_dict[l] for l in layer["in_layers"]]
                else:
                    this_in_layers = out_dict[layer["in_layers"]]
                out = layer["module"](this_in_layers)
                out_dict[name] = out

        return out

    @property
    def out_channels(self):
        return self._out_channels
