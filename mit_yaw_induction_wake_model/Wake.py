from typing import Optional, Tuple
import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz


class Gaussian:
    def __init__(self, u4: float, v4: float, sigma=0.25, kw=0.07) -> None:
        self.u4 = u4
        self.v4 = v4
        self.sigma = sigma  # Default values from paper
        self.kw = kw  # Default values from paper

    def centerline(self, x: np.ndarray, dx=0.5) -> np.ndarray:
        """
        Solves Eq. C4.
        """
        xmax = np.max(x)
        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self.wake_diameter(_x)

        dv = -0.5 / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
        _yc_temp = cumtrapz(-dv, dx=dx, initial=0)

        yc_temp = np.interp(x, _x, _yc_temp, left=0)

        return yc_temp * self.v4

    def wake_diameter(self, x: np.ndarray) -> np.ndarray:
        """
        Solves the normalized far-wake diameter (between C1 and C2)
        """
        return 1 + self.kw * np.log(1 + np.exp(2 * x - 1))

    def du(self, x: np.ndarray, wake_diameter: Optional[float] = None) -> np.ndarray:
        """
        Solves Eq. C2
        """
        d = self.wake_diameter(x) if wake_diameter is None else wake_diameter

        du = 0.5 * (1 - self.u4) / d**2 * (1 + erf(x / (np.sqrt(2) / 2)))
        return du

    def deficit(self, x: np.ndarray, y: np.ndarray, z=0) -> np.ndarray:
        """
        Solves Eq. C1
        """
        d = self.wake_diameter(x)
        yc = self.centerline(x)
        du = self.du(x, wake_diameter=d)
        deficit_ = (
            1
            / (8 * self.sigma**2)
            * np.exp(-(((y - yc) ** 2 + z**2) / (2 * self.sigma**2 * d**2)))
        )

        return deficit_ * du

    def line_deficit(self, x: np.array, y: np.array):
        """
        Returns the deficit at hub height averaged along a lateral line of
        length 1, centered at (x, y).
        """

        d = self.wake_diameter(x)
        yc = self.centerline(x)
        du = self.du(x, wake_diameter=d)

        erf_plus = erf((y + 0.5 - yc) / (np.sqrt(2) * self.sigma * d))
        erf_minus = erf((y - 0.5 - yc) / (np.sqrt(2) * self.sigma * d))

        deficit_ = np.sqrt(2 * np.pi) * d / (16 * self.sigma) * (erf_plus - erf_minus)

        return deficit_ * du


class GradGaussian(Gaussian):
    def __init__(
        self,
        u4: float,
        v4: float,
        dudCt: float,
        dudyaw: float,
        dvdCt: float,
        dvdyaw: float,
        sigma=0.25,
        kw=0.07,
    ) -> None:
        self.u4 = u4
        self.v4 = v4
        self.dudCt = dudCt
        self.dudyaw = dudyaw
        self.dvdCt = dvdCt
        self.dvdyaw = dvdyaw

        self.sigma = sigma  # Default values from paper
        self.kw = kw  # Default values from paper

    def centerline(self, x: np.array, dx=0.5) -> Tuple[np.array, np.array, np.array]:
        xmax = np.max(x)
        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self.wake_diameter(_x)

        dv = -0.5 / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
        _yc_temp = cumtrapz(-dv, dx=dx, initial=0)

        yc_temp = np.interp(x, _x, _yc_temp, left=0)

        dycdCt = yc_temp * self.dvdCt
        dycdyaw = yc_temp * self.dvdyaw

        return yc_temp * self.v4, dycdCt, dycdyaw

    def du(self, x: np.array, wake_diameter=None):
        d = self.wake_diameter(x) if wake_diameter is None else wake_diameter

        du = 0.5 * (1 - self.u4) / d**2 * (1 + erf(x / (np.sqrt(2) / 2)))
        dudCt = -self.dudCt * du / (1 - self.u4)
        dudyaw = -self.dudyaw * du / (1 - self.u4)
        return du, dudCt, dudyaw

    def deficit(
        self, x: np.array, y: np.array, z=0
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Solves Eq. C1
        """
        d = self.wake_diameter(x)
        yc, dycdCt, dycdyaw = self.centerline(x)
        du, dudCt, dudyaw = self.du(x, wake_diameter=d)
        deficit_ = (
            1
            / (8 * self.sigma**2)
            * np.exp(-(((y - yc) ** 2 + z**2) / (2 * self.sigma**2 * d**2)))
        )

        ddeficitdCt = deficit_ * dudCt + du * deficit_ * (
            (y - yc) / (self.sigma**2 * d**2) * dycdCt
        )
        ddeficitdyaw = deficit_ * dudyaw + du * deficit_ * (
            (y - yc) / (self.sigma**2 * d**2) * dycdyaw
        )

        return deficit_ * du, ddeficitdCt, ddeficitdyaw

    def line_deficit(
        self, x: np.array, y: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Returns the deficit at hub height averaged along a lateral line of
        length 1, centered at (x, y).
        """

        d = self.wake_diameter(x)
        yc, dycdCt, dycdyaw = self.centerline(x)
        du, dudCt, dudyaw = self.du(x, wake_diameter=d)

        erf_plus = erf((y + 0.5 - yc) / (np.sqrt(2) * self.sigma * d))
        erf_minus = erf((y - 0.5 - yc) / (np.sqrt(2) * self.sigma * d))

        exp_plus = -np.exp(-((y + 0.5 - yc) ** 2) / (2 * self.sigma**2 * d**2))
        exp_minus = -np.exp(-((y - 0.5 - yc) ** 2) / (2 * self.sigma**2 * d**2))

        deficit_ = np.sqrt(2 * np.pi) * d / (16 * self.sigma) * (erf_plus - erf_minus)

        ddeficitdCt = deficit_ * dudCt + du / (8 * self.sigma**2) * dycdCt * (
            exp_plus - exp_minus
        )
        ddeficitdyaw = deficit_ * dudyaw + du / (8 * self.sigma**2) * dycdyaw * (
            exp_plus - exp_minus
        )

        return deficit_ * du, ddeficitdCt, ddeficitdyaw
