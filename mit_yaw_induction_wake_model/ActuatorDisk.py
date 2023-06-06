"""
Implementation of the yaw-thrust actuator disk model as described in 'Modelling
the induction, thrust and power of a yaw-misaligned actuator disk' Heck et al.
2023.
"""

import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz, trapz


def limitedcase(Ctprime, yaw, Uamb=1):
    """
    Solves the limiting case when v_4 << u_4. (Eq. 2.19, 2.20)
    """
    a = Ctprime * np.cos(yaw) ** 2 / (4 + Ctprime * np.cos(yaw) ** 2)
    u4 = Uamb * (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
    v4 = Uamb * (
        -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
        / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
    )

    return a, u4, v4


def subiteration(Ctprime, yaw, Uamb, a, u4, v4):
    """
    Subiteration of Eq. 2.15
    """
    _a = 1 - np.sqrt(Uamb**2 - u4**2 - v4**2) / (
        np.sqrt(Ctprime) * Uamb * np.cos(yaw)
    )

    _u4 = Uamb * (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2)

    _v4 = -Uamb * 0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2

    return _a, _u4, _v4


def fullcase(Ctprime, yaw, Uamb=1, eps=0.000001):
    """
    Solves Eq. 2.15.
    """
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
        """
        Solves the normalized far-wake diameter (between C1 and C2)
        """
        return 1 + self.kw * np.log(1 + np.exp(2 * x / self.D - 1))

    def _du(self, x):
        """
        Solves Eq. C2
        """
        return (
            0.5
            * (self.Uamb - self.u4)
            / self.wake_diameter(x) ** 2
            * (1 + erf(x / (np.sqrt(2) * self.D / 2)))
        )

    def _dv(self, x):
        """
        Solves Eq. C3.
        """
        return (
            -0.5
            * self.v4
            / self.wake_diameter(x) ** 2
            * (1 + erf(x / (np.sqrt(2) * self.D / 2)))
        )

    def centerline(self, x, dx=0.01):
        """
        Solves Eq. C4.
        """
        xmax = np.max(x)
        _x = np.arange(0, xmax, dx)
        dv = self._dv(_x)
        _yc = cumtrapz(-dv, dx=dx, initial=0)

        return np.interp(x, _x, _yc)

    def deficit(self, x, y):
        """
        Solves Eq. C1
        """
        return (
            self._du(x)
            * self.D**2
            / (8 * self.sigma**2)
            * np.exp(
                -((y - self.centerline(x)) ** 2)
                / (2 * self.sigma**2 * self.wake_diameter(x) ** 2)
            )
        )

    def REWS(self, x, y, R, disc=50):
        """
        Calculates the rotor effective wind speed over a semiaxisymmetric disk
        of radius R located at downstream and lateral location (x, y) relative
        to the wake source. Disk is assumed to be perpendicular to the
        freestream direction (x).
        """
        dys = np.linspace(-R, R, disc)
        ys = y + dys

        deficit = self.deficit(x, ys)
        REWS = 1 / R**2 * trapz(np.abs(dys) * (1 - deficit), dys)

        return REWS


if __name__ == "__main__":
    pass
