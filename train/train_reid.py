import argparse
import tensorflow as tf
import os
from data.reid_dataset import ReIDDataset
from model.mobilenet import MobileNet
from model.resnet import ResNet
from pathlib import Path
from train.trainer import Trainer
from train.summary_writer_reid import SummaryWriterReID
from functools import partial
from data.image_augmentation import ImageAugmentation
from pathlib import Path
from util.io import load_yaml
from train.optimizer import generate_optimizer
from model.model_generator import generate_model
from train.checkpoint_manager import CheckpointManager


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_log", type=str)
    parser.add_argument("--dir_chkpt", type=str)
    parser.add_argument("--path_config", type=Path)
    parser.add_argument("--path_model_config", type=Path)

    return parser


def train_reid(args):
    cfg = load_yaml(args.path_config)
    dir_log = args.dir_log
    dir_chkpt = args.dir_chkpt
    custom = True
    os.makedirs(dir_log, exist_ok=True)
    os.makedirs(dir_chkpt, exist_ok=True)

    augmentation = ImageAugmentation(**cfg.augmentation)

    dataset = ReIDDataset(dir_root=cfg.dataset.data[0].dir_root,
                          path_gt=cfg.dataset.data[0].path,
                          batch_size=cfg.dataset.batch_size,
                          image_size=cfg.dataset.img_size,
                          scale=cfg.dataset.scale,
                          shift=cfg.dataset.shift,
                          augmentation=augmentation).dataset

    config_model = load_yaml(args.path_model_config)
    model = generate_model(config_model.model.type, config_model.model.kwargs)

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    summary_writer = SummaryWriterReID(dir_log=dir_log,
                                       interval_to_write_image=cfg.log.interval_to_write_image)

    optimizer = generate_optimizer(cfg.optimizer.type, cfg.optimizer.kwargs)
    chkpt_manager = CheckpointManager(dir_chkpt=dir_chkpt,
                                      model=model,
                                      optimizer=optimizer,
                                      **cfg.chkpt_manager.kwargs)
    n_iter = chkpt_manager.restore_or_initialize()

    if custom:
        trainer = Trainer(epochs=cfg.train.epochs,
                          dataset=dataset,
                          model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          summary_writer=summary_writer,
                          chkpt_manager=chkpt_manager,
                          start_iter=n_iter)

        trainer.train()
    else:
        model.compile(optimizer=optimizer,
                      loss=criterion,
                      metrics=["accuracy"])

        tb_writer = tf.keras.callbacks.TensorBoard(log_dir=dir_log,
                                                   histogram_freq=1,
                                                   write_images=True,
                                                   write_graph=True)
        model.fit(dataset,
                  epochs=cfg.train.epochs,
                  callbacks=[tb_writer])


if __name__ == "__main__":
    args = generate_parser().parse_args()
    train_reid(args)