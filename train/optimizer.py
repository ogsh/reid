from tensorflow.keras.optimizers import Adam, SGD


def generate_optimizer(type, kwargs):
    optimizer = eval(type)(**kwargs)

    return optimizer