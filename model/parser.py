import tensorflow as tf
from model.layer import (conv2d, resblock3x3, conv3x3, maxpool2x2, gap, dense,
                         spatial_concat, pyramid_gap)
from collections import OrderedDict


def _get_module(module_name):
    modules = [conv2d, resblock3x3, conv3x3, maxpool2x2,
               gap, dense, spatial_concat, pyramid_gap]
    module_dict = {m.__name__: m for m in modules}
    module = module_dict[module_name]

    return module


def parse_model(in_channels, config):
    layer_dict = OrderedDict()

    channels = in_channels
    for name, layer in config.layers.items():
        layer_dict[name] = {}

        in_layers = layer.in_layers if "in_layers" in layer else None
        layer_dict[name]["in_layers"] = in_layers

        if layer.kwargs is None:
            layer_dict[name]["module"] = _get_module(layer.type)(in_channels=channels)
        else:
            layer_dict[name]["module"] = _get_module(layer.type)(in_channels=channels, **layer.kwargs)
            if "out_channels" in layer.kwargs:
                channels = layer.kwargs.out_channels

    return layer_dict, channels
    # sequential = tf.keras.Sequential()
    # for name, layer in layer_dict.items():
    #     sequential.add(layer)

    # return sequential, channels
