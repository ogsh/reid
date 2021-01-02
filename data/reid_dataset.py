import tensorflow as tf
from util.io import load_pickle
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from util.bb import draw_bb
from util.io import load_image
from data.preprocess import random_crop_and_resize


class ImageLoader:
    def __init__(self, dir_root):
        self._dir_root = tf.constant(str(dir_root), dtype=tf.string)

    def __call__(self, data):
        path_img = tf.strings.join([self._dir_root, "/", data["img_name"]])
        data["path_img"] = path_img
        img = load_image(path_img)
        data["img"] = img
        data["label"] = data["label"]

        return data


class DataWrapper:
    def __init__(self, data):
        self._data = data
        self._len = len(data)

    def __call__(self):
        for d in self._data:
            yield d

    def __len__(self):
        return self._len


class ImagePreprocessor:
    def __init__(self,
                 image_size,
                 scale,
                 shift):
        self._image_size = image_size
        self._scale = scale
        self._shift = shift

    def __call__(self, data):
        img = data["img"]
        bb = data["bb"]

        img, _ = random_crop_and_resize(tf.expand_dims(img, axis=0),
                                        bbox=bb,
                                        scale=self._scale,
                                        shift=self._shift,
                                        out_size=self._image_size)

        img = img / 255.0

        data["img"] = img[0, :]

        return data


class ReIDDataset:
    def __init__(self,
                 dir_root,
                 path_gt,
                 batch_size,
                 image_size,
                 scale,
                 shift,
                 augmentation):
        self.batch_sie = batch_size
        self.image_size = image_size
        row_data = DataWrapper(load_pickle(path_gt))
        self._data_length = len(row_data)

        self._dataset = tf.data.Dataset.from_generator(row_data,
                                                       output_types={"img_name": tf.string,
                                                                     "bb": tf.float32,
                                                                     "label": tf.int64},
                                                       output_shapes={"img_name": (),
                                                                      "bb": (4,),
                                                                      "label": ()})

        image_loader = ImageLoader(dir_root)
        self._dataset = self._dataset.map(image_loader, num_parallel_calls=tf.data.AUTOTUNE)
        image_preprocessor = ImagePreprocessor(image_size=image_size,
                                               scale=scale,
                                               shift=shift)

        self._dataset = self._dataset.map(image_preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

        self._dataset = self._dataset.shuffle(buffer_size=len(self))
        self._dataset = self._dataset.batch(batch_size)

        if augmentation is not None:
           self._dataset = self._dataset.map(augmentation)


    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return self._data_length

