import pickle as pkl
import tensorflow as tf

def load_pickle(path):
    with open(path, "rb") as f:
        data = pkl.load(f)

    return data


def save_pickle(path, data):
    with open(path, "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_image(path):
    img_file = tf.io.read_file(path)
    img = tf.image.decode_image(img_file, expand_animations=False)

    return img

