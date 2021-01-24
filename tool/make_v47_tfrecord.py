from pathlib import Path
from util.io import save_pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from tool.make_dataset_pickle import make_v47_gt
from util.io import load_image
from tool.tool_tfrecord import create_record
from tqdm import tqdm


def make_record(all_gts, dir_root, path_dst):

    with tf.io.TFRecordWriter(str(path_dst)) as writer:
        for gt in tqdm(all_gts):
            img_name = gt["img_name"]
            path_image = Path(dir_root, img_name)
            if not path_image.exists():
                raise FileNotFoundError(str(path_image))
            image = load_image(str(path_image)).numpy()
            bb = gt["bb"]
            label = gt["label"]
            record = create_record(image, label, bb)
            writer.write(record)


def make_v47_tfrecord(dir_root, path_out):
    dir_root = Path(dir_root)
    path_gts = list(dir_root.glob("**/*.txt"))

    all_gts = []
    for path_gt in path_gts:
        gts = make_v47_gt(dir_root, path_gt)
        all_gts += gts

    make_record(all_gts, dir_root, path_out)


if __name__ == "__main__":
    dir_root = "../dataset/v47"
    path_dst = Path(dir_root, "v47.tfrecords")
    make_v47_tfrecord(dir_root, path_dst)
