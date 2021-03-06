import tensorflow as tf
from tqdm import tqdm
from util.number import MovingAverage
from train.checkpoint_manager import CheckpointManager


def train_each_iteration(data, model, criterion):
    with tf.GradientTape() as tape:
        inference = model(data["image"])
        loss_value = criterion(data["label"], inference)

    grads = tape.gradient(loss_value, model.trainable_weights)

    return loss_value, grads


class Trainer:
    def __init__(self,
                 epochs: int,
                 dataset: tf.data.Dataset,
                 model: tf.keras.Model,
                 criterion,
                 optimizer: tf.keras.optimizers.Optimizer,
                 summary_writer,
                 chkpt_manager: CheckpointManager,
                 start_iter):
        self._dataset = dataset
        self._model = model
        self._criterion = criterion
        self._epochs = epochs
        self._optimizer = optimizer
        self._summary_writer = summary_writer
        self._chkpt_manager = chkpt_manager
        self._start_iter = start_iter

    def train(self):
        loss_average = MovingAverage()
        current_iter = self._start_iter + 1
        for epoch in range(self._epochs):
            with tqdm(enumerate(self._dataset)) as loop_status:
                for step, data in loop_status:
                    loss_value, grads = train_each_iteration(data, self._model, self._criterion)

                    self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

                    loss_average.update(loss_value)
                    loop_status.postfix = "loss:" + str(float(loss_average.val))

                    self._chkpt_manager.save(current_iter)
                    self._summary_writer.write(current_iter, self._model, data, loss_value)
                    current_iter += 1
