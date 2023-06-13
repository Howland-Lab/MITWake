import numpy as np


class Linear:
    def __call__(self, deficits, ddeficitdCts, ddeficitdyaws, windfarm):
        U = 1 - np.sum(deficits, axis=0)
        dUdCts = -ddeficitdCts
        dUdyaws = -ddeficitdyaws

        return U, dUdCts, dUdyaws


class Quadratic:
    pass


class NiayifarLinear:
    pass


class NiayifarQuadratic:
    pass


class Zong:
    pass
