import tensorflow as tf


def write_loss(loss_dict, iter):
    for key, item in loss_dict:
        if "loss" in key:
            tf.summary.scalar(key, item, step=iter)


def write_model_weight(model, iter):
    for layer in model.layers:
        for weight in layer.weights:
            weight_name = weight.name.replace(':', '_')
            tf.summary.histogram(weight_name, weight, step=iter)
