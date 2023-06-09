import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz


class Gaussian:
    def __init__(self, u4, v4, sigma=0.25, kw=0.07):
        self.u4 = u4
        self.v4 = v4
        self.sigma = sigma  # Default values from paper
        self.kw = kw  # Default values from paper

    def wake_diameter(self, x):
        """
        Solves the normalized far-wake diameter (between C1 and C2)
        """
        return 1 + self.kw * np.log(1 + np.exp(2 * x - 1))

    def _du(self, x):
        """
        Solves Eq. C2
        """
        return (
            0.5
            * (self.Uamb - self.u4)
            / self.wake_diameter(x) ** 2
            * (1 + erf(x / (np.sqrt(2) / 2)))
        )

    def _dv(self, x):
        """
        Solves Eq. C3.
        """
        return (
            -0.5
            * self.v4
            / self.wake_diameter(x) ** 2
            * (1 + erf(x / (np.sqrt(2) / 2)))
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


class GradGaussian(Gaussian):
    def __init__(self, u4, v4, dudCt, dudyaw, dvdCt, dvdyaw, sigma=0.25, kw=0.07):
        self.u4 = u4
        self.v4 = v4
        self.dudCt = dudCt
        self.dudyaw = dudyaw
        self.dvdCt = dvdCt
        self.dvdyaw = dvdyaw

        self.sigma = sigma  # Default values from paper
        self.kw = kw  # Default values from paper

    def centerline(self, x, dx=0.01):
        xmax = np.max(x)
        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self.wake_diameter(_x)

        dv = -0.5 / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
        _yc_temp = cumtrapz(-dv, dx=dx, initial=0)

        yc_temp = np.interp(x, _x, _yc_temp, left=0)

        dycdCt = yc_temp * self.dvdCt
        dycdyaw = yc_temp * self.dvdyaw

        return yc_temp * self.v4, dycdCt, dycdyaw

    def du(self, x):
        d = self.wake_diameter(x)

        du = 0.5 * (1 - self.u4) / d**2 * (1 + erf(x / (np.sqrt(2) / 2)))
        dudCt = -self.dudCt * du / (1 - self.u4)
        dudyaw = -self.dudyaw * du / (1 - self.u4)
        return du, dudCt, dudyaw

    def deficit(self, x, y, z=0):
        """
        Solves Eq. C1
        """
        yc, dycdCt, dycdyaw = self.centerline(x)
        du, dudCt, dudyaw = self.du(x)
        d = self.wake_diameter(x)
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
