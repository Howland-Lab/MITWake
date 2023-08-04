import numpy as np
from .BaseClasses import WindfieldBase


class Uniform(WindfieldBase):
    def wsp(self, x, y, z):
        return np.ones_like(x), np.zeros_like(x)

    def wdir(self, x, y, z):
        return np.zeros_like(x)
