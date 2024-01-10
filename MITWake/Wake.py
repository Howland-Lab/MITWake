from typing import Optional, Tuple
import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz
from MITWake.Rotor import yawthrust

class Gaussian:
    def __init__(self, u4: float=None, v4: float=None, 
                 ctp=None, yaw=None, 
                 sigma=0.25, kw=0.07, 
                 x0=1., 
                 astar=2.32, bstar=0.154, 
                 TI=0.05, 
                 smooth=True, 
                ) -> None:
        """
        Args: 
            u4, v4 (float): from MITWake.Rotor.yawthrust
            ctp (float): C_T' value
            yaw (float): yaw (radians)
        """
        if u4 is None and v4 is None: 
            a, u4, v4 = yawthrust(Ctprime=ctp, yaw=yaw)  # may throw error
            ct1 = ctp * np.cos(yaw)**2 * (1 - a)**2
            self.ctp = ctp
            self.yaw = yaw
            self.ct = ct1  # "traditional" thrust coefficient
        self.u4 = u4
        self.v4 = v4
        self.sigma = sigma  # Default values from paper
        self.kw = kw  # Default values from paper
        self.x0 = x0  # uses \Delta_w = 0.5 if True
        self.astar, self.bstar = astar, bstar
        self.TI = TI
        self.smooth = smooth

    def centerline(self, x: np.ndarray, dx=0.05) -> np.ndarray:
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
        if self.x0 < 0: 
            # uses near-wake length from Bastankhah and Porte-Agel (2016)
            ct = self.ct * np.cos(self.yaw)**2  # BP16 definition of C_T
            x0 = np.cos(self.yaw) * (1 + np.sqrt(1 - ct)) / \
                (np.sqrt(2) * (self.astar * self.TI + self.bstar * (1 - np.sqrt(1 - ct))))
        else: 
            x0 = self.x0

        if self.smooth: 
            return 1 + self.kw * np.log(1 + np.exp(2 * (x - x0)))

        diam = 1 + self.kw * 2 * (x - x0)
        diam[diam < 1] = 1
        return diam

        # Heck, et al. (2023) paper uses: 
        # return 1 + self.kw * np.log(1 + np.exp(2 * (x - 1)))

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

        return np.squeeze(deficit_ * du)

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

    def centerline(self, x: np.array, dx=0.05) -> Tuple[np.array, np.array, np.array]:
        xmax = np.max(x)
        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self.wake_diameter(_x)

        dv = -0.5 / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
        _yc_temp = cumtrapz(-dv, _x, initial=0)

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


class GaussianBP():
    """
    Yawed Gaussian wake model described in Bastankhah and Porte-Agel (2016).
    """

    def __init__(
        self,
        ct: float,
        yaw: float = 0.0,
        ky: float = 0.035,
        kz: Optional[float] = None,
        TI: float = 0.05,
        astar: float = 2.32,
        bstar: float = 0.154,
        d: float = 1,
        u_inf: float = 1,
        alpha_in: np.ndarray = None,
        alpha_z: np.ndarray = None,  
    ):
        """
        Args:
            ct (float): Rotor thrust coefficient, non-dimensionalized to
                pi/8 d**2 rho u_h^2.
            yaw (float): Rotor yaw angle (radians).
            ky (float): Wake spreading parameter. Defaults to 0.07.
            kz (float, optional): Wake spreading parameter. Defaults to None.
            TI (float): Turbulence intensity. Defaults to 0.05.
            astar (float, optional): alpha^* tuning parameter. Defaults to 2.32.
            bstar (float, optional): beta^* tuning parameter. Defaults to 0.154.
            d (float, optional): non-dimensionalizing value for diameter. Defaults to 1.
            u_inf (float, optional): hub height velocity. Defaults to 1.
            alpha_in (float or ndarray, optional): inflow angles as a function of z
            alpha_z (float or ndarray, optional): z-locations corresponding to indices of alpha_in
        """
        self.ct = ct
        self.yaw = -yaw  # BP2016 uses CW positive sign convention for yaw
        self.ky = ky
        if kz is None:
            self.kz = ky
        else:
            self.kz = kz
        self.TI = TI
        self.astar = astar
        self.bstar = bstar
        self.d = d
        self.x0 = self.calc_x0()
        self.theta0 = self.calc_theta0()
        self.u_inf = u_inf
        self.alpha_in = alpha_in
        self.alpha_in_z = alpha_z

    def calc_theta0(self):
        """
        Solves eq. 6.12
        """
        theta_0 = (
            0.3
            * self.yaw
            / np.cos(self.yaw)
            * (1 - np.sqrt(1 - self.ct * np.cos(self.yaw)))
        )
        return theta_0

    def calc_x0(self):
        """
        Solves eq. 7.3
        """
        x0 = self.d * np.cos(self.yaw) * (1 + np.sqrt(1 - self.ct)) / \
            (np.sqrt(2) * (self.astar * self.TI + self.bstar * (1 - np.sqrt(1 - self.ct))))
        return x0

    def sigma_y(
        self,
        x: np.array,
    ):
        """
        Solves eq. 7.2a
        """
        x = np.atleast_1d(x)
        sigma_y0 = self.d * np.cos(self.yaw) / np.sqrt(8)
        sigma_y = self.ky * (x - self.x0) + sigma_y0
        sigma_y[x < self.x0] = sigma_y0
        return sigma_y

    def sigma_z(
        self,
        x: np.array,
    ):
        """
        Solves eq. 7.2b
        """
        x = np.atleast_1d(x)
        sigma_z0 = self.d / np.sqrt(8)
        sigma_z = self.kz * (x - self.x0) + sigma_z0
        sigma_z[x < self.x0] = sigma_z0
        return sigma_z

    def centerline(
        self,
        x: np.array,
    ):
        """
        Solves eq. 7.4
        """
        x = np.atleast_1d(x)
        d = self.d
        t0 = self.theta0
        ct = self.ct
        cos = np.cos(self.yaw)
        A1 = 1.6 * np.sqrt(
            8 * self.sigma_y(x) * self.sigma_z(x) / d**2 / cos
        )  # tmp variable

        delta = t0 * self.x0 + d * t0 / 14.7 * np.sqrt(cos / self.ky / self.kz / ct) * (
            2.9 + 1.3 * np.sqrt(1 - ct) - ct
        ) * np.log(
            (1.6 + np.sqrt(ct)) * (A1 - np.sqrt(ct))
            / ((1.6 - np.sqrt(ct)) * (A1 + np.sqrt(ct)))
        )

        delta = np.atleast_1d(delta)
        delta[x < self.x0] = t0 * x[x < self.x0]
        delta[x < 0] = 0
        return delta

    def deficit(
        self,
        x: np.array,
        y: np.array,
        z: Optional[np.array] = 0,
    ):
        """
        Computes wake deficit (eq. 7.1)
        """

        sigma_y = self.sigma_y(x)
        sigma_z = self.sigma_z(x)
        delta = self.centerline(x)
        if self.alpha_in is not None: 
            if self.alpha_in_z is None: 
                # assume linear veer, and alpha_in is given by eq. 5 in Abkar et al. (2018)
                alpha_in = alpha_in * z
            else: 
                alpha_in = np.interp(z, self.alpha_in_z, self.alpha_in)
            delta_veer = x * np.tan(alpha_in)
        else: 
            delta_veer = 0  # no deflection due to veer
        
        radical = 1 - self.ct * np.cos(self.yaw) / (8 * sigma_y * sigma_z / self.d**2)
        radical[np.isclose(radical, 0)] = 0  # helps with avoiding nans
        C1 = self.u_inf * (
            1 - np.sqrt(radical)
        )
        C1[x < 0] = 0  # no upstream wakes!
        delta_u = (
            C1
            * np.exp(-0.5 * ((y - delta - delta_veer) / sigma_y) ** 2)
            * np.exp(-0.5 * (z / sigma_z) ** 2)
        )
        
        return np.squeeze(delta_u)
    
