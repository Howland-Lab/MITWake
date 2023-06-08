"""
Implementation of the yaw-thrust actuator disk model as described in 'Modelling
the induction, thrust and power of a yaw-misaligned actuator disk' Heck et al.
2023.
"""

import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz, trapz


def calculate_induction_limited(Ctprime, yaw, Uamb=1):
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


def _calculate_induction(Ctprime, yaw, Uamb, a, u4, v4):
    """
    Subiteration of Eq. 2.15
    """
    _a = 1 - np.sqrt(Uamb**2 - u4**2 - v4**2) / (
        np.sqrt(Ctprime) * Uamb * np.cos(yaw)
    )

    _u4 = Uamb * (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2)

    _v4 = -Uamb * 0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2

    return _a, _u4, _v4


def calculate_induction(Ctprime, yaw, Uamb=1, eps=0.000001):
    """
    Solves Eq. 2.15.
    """
    a_guess, u4_guess, v4_guess = calculate_induction_limited(Ctprime, yaw, Uamb)

    niter = 1
    converged = False
    while not converged:
        a, u4, v4 = _calculate_induction(
            Ctprime, yaw, Uamb, a_guess, u4_guess, v4_guess
        )

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
        converged = all(x < eps for x in [a_error, u4_error, v4_error])

    return a, u4, v4


class MITWake:
    def __init__(self, Ctprime, yaw, sigma=0.25, kw=0.07):
        # Default values from paper
        self.Ctprime = Ctprime
        self.yaw = yaw
        self.sigma = sigma
        self.kw = kw
        self._update_induction()

        self.D = 1
        self.Uamb = 1

    def _update_induction(self):
        """
        Calculate a, u4 and v4 for a given thrust and yaw.
        """
        self.a, self.u4, self.v4 = calculate_induction(self.Ctprime, self.yaw)

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

    def deficit(self, x, y, z=0):
        """
        Solves Eq. C1
        """
        return (
            self._du(x)
            * self.D**2
            / (8 * self.sigma**2)
            * np.exp(
                -((y - self.centerline(x)) ** 2 + z**2)
                / (2 * self.sigma**2 * self.wake_diameter(x) ** 2)
            )
        )

    def REWS(self, x0, y0, R=0.5, r_disc=20, theta_disc=50):
        """
        Calculates the rotor effective wind speed over a disk of radius R
        located at downstream and lateral location (x0, y0) relative to the wake
        source. Disk is assumed to be perpendicular to the freestream direction
        (x). The problem is extended from 2 to 3 dimensions to more accurately
        perform the numerical integration.
        """
        # Define points over rotor disk on polar grid
        rs = np.linspace(0, R, r_disc)
        thetas = np.linspace(0, 2 * np.pi, theta_disc)

        r_mesh, theta_mesh = np.meshgrid(rs, thetas)
        ys = r_mesh * np.sin(theta_mesh) + y0
        zs = r_mesh * np.cos(theta_mesh)

        # Evaluate the deficit at points (converted to cartesian).
        deficit = self.deficit(x0, ys, zs)

        # Perform integration over rotor disk in polar coordinates.
        REWS = 1 - np.trapz(np.trapz(r_mesh * deficit, r_mesh, axis=1), thetas)

        return REWS

    def REWS_anal(self, x, y):
        """
        Approximates the rotor effective wind speed at a location downstream and
        lateral location (x, y) relative to the wake source. Disk is assumed to
        be perpendicular to the freestream direction (x).
        """
        yc = self.centerline(x)
        d = self.wake_diameter(x)
        du = self._du(x)

        REWS = 1 - np.sqrt(2 * np.pi) * du * d / (16 * self.sigma) * (
            erf((y + 0.5 - yc) / ((np.sqrt(2) * self.sigma * d)))
            - erf((y - 0.5 - yc) / ((np.sqrt(2) * self.sigma * d)))
        )
        return REWS


if __name__ == "__main__":
    pass
