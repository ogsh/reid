import tensorflow as tf


class CheckpointManager:
    def __init__(self, dir_chkpt, model, optimizer=None, max_to_keep=None, interval_to_save=1):
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self._manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                   directory=dir_chkpt,
                                                   max_to_keep=max_to_keep,
                                                   checkpoint_name=model.name)
        self._interval_to_save = interval_to_save

    def save(self, n_iter):
        if n_iter % self._interval_to_save == 0:
            self._manager.save(checkpoint_number=n_iter)

    def restore_or_initialize(self):
        self._manager.restore_or_initialize()
        latest_chkpt = self._manager.latest_checkpoint
        n_iter = 0 if latest_chkpt is None else int(latest_chkpt.split("-")[-1])

        return n_iter

    def restore_checkpoints(self):
        checkpoints = [self._manager.checkpoint.restore(chkpt) for chkpt in self._manager.checkpoints]

        return checkpoints
