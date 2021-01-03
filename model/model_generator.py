from model.resnet import ResNet
from model.reidnet import ReIDNet


def generate_model(type, kwargs):
    model = eval(type)(**kwargs)

    return model