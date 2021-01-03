from util.io import load_yaml
import os
from model.parser import parse_model


def load_net_config(name):
    cd = os.path.dirname(os.path.abspath(__file__))
    path_file = os.path.join(cd, name + ".yaml")

    config = load_yaml(path_file)

    return config


def generate_net(in_channels, name):
    config = load_net_config(name)
    net = parse_model(in_channels, config)

    return net
