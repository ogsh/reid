import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import (RandomContrast,
                                                                RandomRotation)
from functools import partial


class RandomAugmentation(tf.keras.Model):
    def __init__(self, p, augmentation):
        super(RandomAugmentation, self).__init__()
        self._p = p
        self._augmentation = augmentation

    def __call__(self, x, *args, **kwargs):
        randval = tf.random.uniform([])

        if self._p > randval:
            out = self._augmentation(x, *args, **kwargs)
        else:
            out = x

        return out


class RandomBrightness(tf.keras.Model):
    def __init__(self, factor):
        super(RandomBrightness, self).__init__()
        self._func = partial(tf.image.random_brightness, max_delta=factor)

    def call(self, x, training=None, mask=None):
        y = self._func(x)

        return y


class RandomSaturation(tf.keras.Model):
    def __init__(self, factor):
        super(RandomSaturation, self).__init__()
        self._func = partial(tf.image.random_saturation, lower=factor[0], upper=factor[1])

    def call(self, x, training=None, mask=None):
        y = self._func(x)

        return y


class ImageAugmentation:
    def __init__(self,
                 p_rotation,
                 rotation,
                 p_brightness,
                 brightness,
                 p_contrast,
                 contrast,
                 p_saturation,
                 saturation):
        random_rotation = RandomAugmentation(p_rotation,
                                             RandomRotation(factor=rotation,
                                                            fill_mode="nearest"))
        random_contrast = RandomAugmentation(p_contrast,
                                             RandomContrast(factor=contrast))

        random_brightness = RandomAugmentation(p_brightness,
                                               RandomBrightness(factor=brightness))

        random_saturation = RandomAugmentation(p_saturation,
                                               RandomSaturation(factor=saturation))

        self._augmentations = tf.keras.Sequential([random_rotation,
                                                   random_contrast,
                                                   random_brightness,
                                                   random_saturation])

    def __call__(self, data):
        data["img"] = self._augmentations(data["img"])

        return data
