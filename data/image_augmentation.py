import tensorflow as tf
import tensorflow_addons as tfa
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


class RandomCutout(tf.keras.Model):
    def __init__(self, mask_area_rate, aspect_rate):
        super(RandomCutout, self).__init__()
        self._mask_area_rate = mask_area_rate
        self._aspect_rate = aspect_rate

    def call(self, x, training=None, mask=None):
        batch, height, width, channels = tf.shape(x)
        image_area = float(height * width)
        area_rate = tf.random.uniform(shape=[], minval=self._mask_area_rate[0], maxval=self._mask_area_rate[1])
        mask_area = image_area * area_rate
        aspect = tf.random.uniform(shape=[], minval=self._aspect_rate[0], maxval=self._aspect_rate[1])
        mask_height = tf.sqrt(mask_area) * aspect
        mask_width = tf.sqrt(mask_area) / aspect
        mask_size = 0.5 * tf.concat([mask_height, mask_width], axis=0)
        fill_value = tf.random.uniform(shape=[channels], minval=0, maxval=255)
        offset_x = tf.random.uniform(shape=[batch, 1], minval=mask_size[1], maxval=float(width-1)-mask_size[1])
        offset_y = tf.random.uniform(shape=[batch, 1], minval=mask_size[0], maxval=float(height-1)-mask_size[0])
        offset = tf.cast(tf.concat([offset_y, offset_x], axis=1), dtype=tf.int32)
        mask_size = 2 * tf.cast(mask_size, dtype=tf.int32)

        y = tfa.image.cutout(images=x, mask_size=mask_size, constant_values=fill_value, offset=offset)

        return y


# class RandomShear(tf.keras.Model):
#     def __init__(self, factor):
#         super(RandomShear, self).__init__()
#         self._factor = factor
#
#     def call(self, x, training=None, mask=None):
#         x = x[0, :, :, :]
#         level_x = tf.random.uniform(shape=[], minval=self._factor[0], maxval=self._factor[1])
#         out = tfa.image.shear_x(x, level=level_x, replace=tf.constant([0]))
#         level_y = tf.random.uniform(shape=[], minval=self._factor[0], maxval=self._factor[1])
#         out = tfa.image.shear_x(out, level=level_y, replace=tf.constant([0]))
#
#         return out






class ImageAugmentation:
    def __init__(self,
                 p_rotation,
                 rotation,
                 p_brightness,
                 brightness,
                 p_contrast,
                 contrast,
                 p_saturation,
                 saturation,
                 p_cutout,
                 cutout_size_rate,
                 cutout_aspect_rate):
        random_rotation = RandomAugmentation(p_rotation,
                                             RandomRotation(factor=rotation,
                                                            fill_mode="nearest"))
        random_contrast = RandomAugmentation(p_contrast,
                                             RandomContrast(factor=contrast))

        random_brightness = RandomAugmentation(p_brightness,
                                               RandomBrightness(factor=brightness))

        random_saturation = RandomAugmentation(p_saturation,
                                               RandomSaturation(factor=saturation))

        random_cutout = RandomAugmentation(p_cutout,
                                           RandomCutout(cutout_size_rate, cutout_aspect_rate))

        self._augmentations = tf.keras.Sequential([random_rotation,
                                                   random_contrast,
                                                   random_brightness,
                                                   random_saturation,
                                                   random_cutout], name="augmentation")

    def __call__(self, data):
        data["img"] = self._augmentations(data["img"])

        return data
