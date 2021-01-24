import tensorflow as tf
from train.summary_writer import write_model_weight


class SummaryWriterReID:
    def __init__(self, dir_log, interval_to_write_image=1):
        self._writer = tf.summary.create_file_writer(logdir=dir_log)
        self._interval_to_write_image = interval_to_write_image

    def _write_summary(self,
                       iter: int,
                       model: tf.keras.Model,
                       data,
                       loss,
                       save_image=False):
        with self._writer.as_default():
            tf.summary.scalar("loss", float(loss), step=iter)

            if save_image:
                img = data["image"]
                tf.summary.image("image", img, step=iter, max_outputs=32)
                write_model_weight(model, iter)

    def write(self, iter, model, data, loss):
        save_img = iter % self._interval_to_write_image == 0
        self._write_summary(iter, model, data, loss, save_image=save_img)
