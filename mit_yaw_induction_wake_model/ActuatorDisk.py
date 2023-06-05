import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz
from tqdm import tqdm


def limitedcase(Ctprime, yaw, Uamb=1):
    a = Ctprime * np.cos(yaw) ** 2 / (4 + Ctprime * np.cos(yaw) ** 2)
    u4 = Uamb * (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
    v4 = Uamb * (
        -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
        / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
    )

    return a, u4, v4


def subiteration(Ctprime, yaw, Uamb, a, u4, v4):
    _a = 1 - np.sqrt(Uamb**2 - u4**2 - v4**2) / (
        np.sqrt(Ctprime) * Uamb * np.cos(yaw)
    )

    _u4 = Uamb * (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2)

    _v4 = -Uamb * 0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2

    return _a, _u4, _v4


def fullcase(Ctprime, yaw, Uamb=1, eps=0.000001):
    a_guess, u4_guess, v4_guess = limitedcase(Ctprime, yaw, Uamb)

    a, u4, v4 = subiteration(Ctprime, yaw, Uamb, a_guess, u4_guess, v4_guess)

    if hasattr(a, "__iter__"):
        a_error = np.linalg.norm((a - a_guess), ord=np.inf)
        u4_error = np.linalg.norm((u4 - u4_guess), ord=np.inf)
        v4_error = np.linalg.norm((v4 - v4_guess), ord=np.inf)
    else:
        a_error = np.abs(a - a_guess)
        u4_error = np.abs(u4 - u4_guess)
        v4_error = np.abs(v4 - v4_guess)

    a_guess, u4_guess, v4_guess = a, u4, v4

    niter = 1
    while any(x > eps for x in [a_error, u4_error, v4_error]):
        a, u4, v4 = subiteration(Ctprime, yaw, Uamb, a_guess, u4_guess, v4_guess)

        if hasattr(a, "__iter__"):
            a_error = np.linalg.norm((a - a_guess), ord=np.inf)
            u4_error = np.linalg.norm((u4 - u4_guess), ord=np.inf)
            v4_error = np.linalg.norm((v4 - v4_guess), ord=np.inf)
        else:
            a_error = np.abs(a - a_guess)
            u4_error = np.abs(u4 - u4_guess)
            v4_error = np.abs(v4 - v4_guess)

        a_guess, u4_guess, v4_guess = a, u4, v4
        niter += 1

    return a, u4, v4


class MITWake:
    def __init__(self, Ctprime, yaw, sigma=0.25, kw=0.07):
        # Default values from paper
        self.Ctprime = Ctprime
        self.yaw = yaw
        self.sigma = sigma
        self.kw = kw
        self.a, self.u4, self.v4 = fullcase(Ctprime, yaw)

        self.D = 1
        self.Uamb = 1

    def wake_diameter(self, x):
        return 1 + self.kw * np.log(1 + np.exp(2 * x / self.D - 1))

    def _du(self, x):
        return (
            0.5
            * (self.Uamb - self.u4)
            / self.wake_diameter(x) ** 2
            * (1 + erf(x / (np.sqrt(2) * self.D / 2)))
        )

    def _dv(self, x):
        return (
            -0.5
            * self.v4
            / self.wake_diameter(x) ** 2
            * (1 + erf(x / (np.sqrt(2) * self.D / 2)))
        )

    def centerline(self, x, dx=0.01):
        xmax = np.max(x)
        _x = np.arange(0, xmax, dx)
        dv = self._dv(_x)
        _yc = cumtrapz(-dv, dx=dx, initial=0)

        return np.interp(x, _x, _yc)

    def deficit(self, x, y):
        return (
            self._du(x)
            * self.D**2
            / (8 * self.sigma**2)
            * np.exp(
                -((y - self.centerline(x)) ** 2)
                / (2 * self.sigma**2 * self.wake_diameter(x) ** 2)
            )
        )


if __name__ == "__main__":

    pass
