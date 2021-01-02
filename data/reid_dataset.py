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

        return data


class DataWrapper:
    def __init__(self, data):
        self._data = data

    def __call__(self):
        for d in self._data:
            yield d


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

        self._dataset = tf.data.Dataset.from_generator(row_data,
                                                       output_types={"img_name": tf.string,
                                                                     "bb": tf.float32},
                                                       output_shapes={"img_name": (),
                                                                      "bb": (4,)})

        image_loader = ImageLoader(dir_root)
        self._dataset = self._dataset.map(image_loader, num_parallel_calls=tf.data.AUTOTUNE)
        image_preprocessor = ImagePreprocessor(image_size=image_size,
                                               scale=scale,
                                               shift=shift)

        self._dataset = self._dataset.map(image_preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    @property
    def dataset(self):
        return self._dataset


def test_reid_dataset():
    import numpy as np

    dir_work = "E:/ogsh/10_work/2020/reid"
    dir_root = Path(dir_work, "dataset/v47")
    path = Path(dir_root, "v47.pkl")
    augmentation = None

    dataset = ReIDDataset(dir_root=dir_root,
                          path_gt=path,
                          batch_size=16,
                          image_size=[64, 64],
                          scale=0.1,
                          shift=0.,
                          augmentation=augmentation)

    for d in dataset.dataset:
        path_img = Path(d["path_img"].numpy().decode("utf-8"))
        print(path_img.exists(), path_img)
        img = d["img"].numpy()
        bb = d["bb"]

        bb_img = draw_bb(img, bb, color=(0, 255, 0))

        cv2.imshow("img", bb_img[:, :, ::-1].copy().astype(np.uint8))
        cv2.waitKey()

        # plt.imshow(img)
        # plt.show()


if __name__ == "__main__":
    test_reid_dataset()
