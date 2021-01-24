import tensorflow as tf
from util.io import load_image
from data.preprocess import random_crop_and_resize
from tool.tool_tfrecord import load_tfrecord
from functools import lru_cache


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
        img = data["image"]
        bb = data["bb"]

        img, _ = random_crop_and_resize(tf.expand_dims(img, axis=0),
                                        bbox=bb,
                                        scale=self._scale,
                                        shift=self._shift,
                                        out_size=self._image_size)

        img = img / 255.0

        data["image"] = img[0, :]

        return data


def set_unique_id(datasets):
    count = 0
    new_dataset = []
    for dataset in datasets:
        hash_count = {}
        for data in dataset:
            label = data["label"]
            data["label"] += count
            new_dataset += [data]
            hash_count[int(label)] = 1
        count += len(hash_count)

    for data in new_dataset:
        print(data["label"])

    return new_dataset


@lru_cache(maxsize=None)
def get_dataset_length(dataset):
    length = sum([1 for _ in dataset])
    return length


class ReIDDataset:
    def __init__(self,
                 path_tfrecord,
                 batch_size,
                 image_size,
                 scale,
                 shift,
                 augmentation):
        self.batch_sie = batch_size
        self.image_size = image_size

        raw_dataset = load_tfrecord(path_tfrecord=path_tfrecord)

        self._length_dataset = get_dataset_length(raw_dataset)

        image_preprocessor = ImagePreprocessor(image_size=image_size,
                                               scale=scale,
                                               shift=shift)

        self._dataset = raw_dataset.map(image_preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

        self._dataset = self._dataset.shuffle(buffer_size=len(self))
        self._dataset = self._dataset.batch(batch_size)

        if augmentation is not None:
           self._dataset = self._dataset.map(augmentation)


    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return self._length_dataset

