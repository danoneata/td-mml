import math
class RunningMeter(object):
    """ running meteor of a scalar value
        (useful for monitoring training loss)
    """
    def __init__(self, name, val=None, smooth=0.99):
        self._name = name
        self._sm = smooth
        self._val = val

    def __call__(self, value):
        val = (value if self._val is None
               else value*(1-self._sm) + self._val*self._sm)
        if not math.isnan(val) and not math.isinf(val):
            self._val = val
        else:
            print(f'Inf/Nan in {self._name}')

    def __str__(self):
        return f'{self._name}: {self._val:.4f}'

    @property
    def val(self):
        return self._val

    @property
    def name(self):
        return self._name