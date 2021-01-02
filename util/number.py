

class MovingAverage:
    def __init__(self, val=None, lr=0.1):
        self._val = val
        self._lr = lr

    def update(self, new_val):
        if self._val is None:
            self._val = new_val
        else:
            self._val = self._lr * new_val + (1 - self._lr) * self._val

        return self._val

    @property
    def val(self):
        return self._val