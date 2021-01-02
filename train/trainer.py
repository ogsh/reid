import tensorflow as tf
from tqdm import tqdm
from util.number import MovingAverage


def train_each_iteration(data, model, criterion):
    with tf.GradientTape() as tape:
        inference = model(data["img"])
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
                 summary_writer):
        self._dataset = dataset
        self._model = model
        self._criterion = criterion
        self._epochs = epochs
        self._optimizer = optimizer
        self._summary_writer = summary_writer

    def train(self):
        loss_average = MovingAverage()
        iter = 0
        for epoch in range(self._epochs):
            with tqdm(enumerate(self._dataset)) as loop_status:
                for step, data in loop_status:
                    loss_value, grads = train_each_iteration(data, self._model, self._criterion)

                    self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

                    loss_average.update(loss_value)
                    loop_status.postfix = "loss:" + str(float(loss_average.val))

                    self._summary_writer.write(iter, self._model, data, loss_value)
                    iter += 1
