import numpy as np
from mit_yaw_induction_wake_model import Rotor
from scipy.special import erf
from scipy.integrate import cumtrapz, trapz


def _calculate_derivatives_Ct(dadCt, dudCt, dvdCt, Ct, yaw, a, u4, v4):
    """
    Calculate the derivatives of the variables dadCt, dudCt, and dvdCt.

    Args:
        dadCt: Current value of dadCt.
        dudCt: Current value of dudCt.
        dvdCt: Current value of dvdCt.
        Ct: Value of Ct.
        yaw: Value of yaw.
        a: Value of a.
        u4: Value of u4.
        v4: Value of v4.

    Returns:
        Tuple: The new values of dadCt, dudCt, and dvdCt.
    """
    dadCt_new = ((1 - a) ** 2 * np.cos(yaw) ** 2 + 2 * u4 * dudCt + 2 * v4 * dvdCt) / (
        2 * Ct * (1 - a) * np.cos(yaw) ** 2
    )
    dudCt_new = 0.5 * Ct * dadCt * np.cos(yaw) ** 2 - 0.5 * (1 - a) * np.cos(yaw) ** 2
    dvdCt_new = (
        0.5 * Ct * (1 - a) * dadCt * np.sin(yaw) * np.cos(yaw) ** 2
        - 0.25 * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2
    )

    return dadCt_new, dudCt_new, dvdCt_new


def calculate_ddCT(Ct, yaw, a, u4, v4, eps=0.000001):
    """
    Calculate the derivatives of dadCt, dudCt, and dvdCt iteratively until convergence.

    Args:
        Ct: Value of Ct.
        yaw: Value of yaw.
        eps: Convergence threshold (default: 0.000001).

    Returns:
        Tuple: The values of dadCt, dudCt, and dvdCt.
    """
    dadCt, dudCt, dvdCt = 0, 0, 0

    niter = 0
    converged = False
    while not converged:
        dadCt_new, dudCt_new, dvdCt_new = _calculate_derivatives_Ct(
            dadCt, dudCt, dvdCt, Ct, yaw, a, u4, v4
        )

        dadCt_error = np.abs(dadCt - dadCt_new)
        dudCt_error = np.abs(dudCt - dudCt_new)
        dvdCt_error = np.abs(dvdCt - dvdCt_new)

        dadCt, dudCt, dvdCt = dadCt_new, dudCt_new, dvdCt_new

        niter += 1
        converged = all(x < eps for x in [dadCt_error, dudCt_error, dvdCt_error])

    return dadCt, dudCt, dvdCt


def _calculate_derivatives_yaw(dadyaw, dudyaw, dvdyaw, Ct, yaw, a, u4, v4):
    """
    Calculate the derivatives of the variables dadyaw, dudyaw, and dvdyaw.

    Args:
        dadyaw: Current value of dadyaw.
        dudyaw: Current value of dudyaw.
        dvdyaw: Current value of dvdyaw.
        Ct: Value of Ct.
        yaw: Value of yaw.
        a: Value of a.
        u4: Value of u4.
        v4: Value of v4.

    Returns:
        Tuple: The new values of dadyaw, dudyaw, and dvdyaw.
    """
    dadyaw_new = (
        u4 * dudyaw + v4 * dvdyaw - Ct * (1 - a) ** 2 * np.cos(yaw) * np.sin(yaw)
    ) / (Ct * (1 - a) * np.cos(yaw) ** 2)
    dudyaw_new = Ct * np.cos(yaw) * (0.5 * np.cos(yaw) * dadyaw + (1 - a) * np.sin(yaw))
    dvdyaw_new = (
        0.5
        * Ct
        * (1 - a)
        * (
            dadyaw * np.sin(yaw) * np.cos(yaw) ** 2
            - 0.5 * (1 - a) * np.cos(yaw) ** 3
            + (1 - a) * np.sin(yaw) ** 2 * np.cos(yaw)
        )
    )

    return dadyaw_new, dudyaw_new, dvdyaw_new


def calculate_ddyaw(Ct, yaw, a, u4, v4, eps=0.000001):
    """
    Calculate the derivatives of dayaw, duyaw, and dvyaw iteratively until convergence.

    Args:
        Ct: Value of Ct.
        yaw: Value of yaw.
        eps: Convergence threshold (default: 0.000001).

    Returns:
        Tuple: The values of dayaw, duyaw, and dvyaw.
    """
    dayaw, duyaw, dvyaw = 0, 0, 0

    niter = 0
    converged = False
    while not converged:
        dayaw_new, duyaw_new, dvyaw_new = _calculate_derivatives_yaw(
            dayaw, duyaw, dvyaw, Ct, yaw, a, u4, v4
        )

        dayaw_error = np.abs(dayaw - dayaw_new)
        duyaw_error = np.abs(duyaw - duyaw_new)
        dvyaw_error = np.abs(dvyaw - dvyaw_new)

        dayaw, duyaw, dvyaw = dayaw_new, duyaw_new, dvyaw_new

        niter += 1
        converged = all(x < eps for x in [dayaw_error, duyaw_error, dvyaw_error])

    return dayaw, duyaw, dvyaw


def calculate_induction(Ct, yaw):
    a, u4, v4 = Rotor.calculate_induction(Ct, yaw)
    dadCt, dudCt, dvdCt = calculate_ddCT(Ct, yaw, a, u4, v4)
    dadyaw, dudyaw, dvdyaw = calculate_ddyaw(Ct, yaw, a, u4, v4)

    return a, u4, v4, dadCt, dudCt, dvdCt, dadyaw, dudyaw, dvdyaw


def wake_diameter(x, kw):
    """
    Solves the normalized far-wake diameter (between C1 and C2)
    """
    return 1 + kw * np.log(1 + np.exp(2 * x - 1))


def calculate_yc(x, v4, dvdCt, dvdyaw, kw, dx=0.01):

    xmax = np.max(x)
    _x = np.arange(0, xmax, dx)
    d = wake_diameter(_x, kw)
    dv = -0.5  / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
    _yc_temp = cumtrapz(-dv, dx=dx, initial=0)

    yc_temp = np.interp(x, _x, _yc_temp)

    dycdCt = yc_temp  * dvdCt
    dycdyaw = yc_temp * dvdyaw

    return yc_temp * v4, dycdCt, dycdyaw


def calculate_du(x, u4, dudCt, dudyaw, kw):
    d = wake_diameter(x, kw)

    du = 0.5 * (1 - u4) / d**2 * (1 + erf(x / (np.sqrt(2) / 2)))
    dudCt = -dudCt * du / (1 - u4)
    dudyaw = -dudyaw * du / (1 - u4)

    return du, dudCt, dudyaw


def calculate_deficit(x, y, z, yc, du, dycdCt, dycdyaw, dudCt, dudyaw, kw, sigma):
    d = wake_diameter(x, kw)
    deficit = (
        du
        / (8 * sigma**2)
        * np.exp(-(((y - yc) ** 2 + z**2) / (2 * sigma**2 * d**2)))
    )

    ddeficitdCt = deficit / du * dudCt + deficit * (
        (y - yc) / (sigma**2 * d**2) * dycdCt
    )
    ddeficitdyaw = deficit / du * dudyaw + deficit * (
        (y - yc) / (sigma**2 * d**2) * dycdyaw
    )

    return deficit, ddeficitdCt, ddeficitdyaw


def calculate_Cp1(Ctprime, yaw, REWS, a, dREWSdCt, dREWSdyaw, dadCt, dadyaw):
    """
    Power coefficient of a turbine turbine affected by its own operation
    """
    Cp = Ctprime * ((1 - a) * np.cos(yaw) * REWS) ** 3

    dCpdCt = Cp / Ctprime + 3 * Cp * (dREWSdCt / REWS - dadCt / (1 - a))
    dCpdyaw = (
        3
        * Cp
        / ((1 - a) * REWS)
        * ((1 - a) * dREWSdyaw - dadyaw * REWS - (1 - a) * np.tan(yaw) * REWS)
    )

    return Cp, dCpdCt, dCpdyaw


def calculate_Cp2(Ctprime2, yaw2, REWS, a2, dREWSdCt, dREWSdyaw, dadCt, dadyaw):
    """
    Power coefficient of a downstream turbine affected by an upstream wake
    THIS GRADIENT DOESNT WORK!
    """
    Cp2 = Ctprime2 * ((1 - a2) * np.cos(yaw2) * REWS) ** 3

    dCp2dCt = 3 * Cp2 * dREWSdCt / REWS
    dCp2dyaw = 3 * Cp2 * dREWSdyaw / REWS

    return Cp2, dCp2dCt, dCp2dyaw


class MITWakeGrad:
    def __init__(self, Ctprime, yaw, sigma=0.25, kw=0.07):
        # Default values from paper
        self.Ctprime = Ctprime
        self.yaw = yaw
        self.sigma = sigma
        self.kw = kw
        self._update_induction()

    def _update_induction(self):
        """
        Calculate a, u4 and v4 for a given thrust and yaw.
        """
        (
            self.a,
            self.u4,
            self.v4,
            self.dadCt,
            self.dudCt,
            self.dvdCt,
            self.dadyaw,
            self.dudyaw,
            self.dvdyaw,
        ) = calculate_induction(self.Ctprime, self.yaw)

    def calculate_yc(self, x):
        # Gradients verified
        yc, dycdCt, dycdyaw = calculate_yc(x, self.v4, self.dvdCt, self.dvdyaw, self.kw)
        return yc, dycdCt, dycdyaw

    def calculate_du(self, x):
        du, dudCt, dudyaw = calculate_du(x, self.u4, self.dudCt, self.dudyaw, self.kw)
        return du, dudCt, dudyaw

    def deficit(self, x, y, z=0):
        """
        Solves Eq. C1
        """
        yc, dycdCt, dycdyaw = calculate_yc(x, self.v4, self.dvdCt, self.dvdyaw, self.kw)
        du, dudCt, dudyaw = calculate_du(x, self.u4, self.dudCt, self.dudyaw, self.kw)
        deficit, ddeficitdCt, ddeficitdyaw = calculate_deficit(
            x, y, z, yc, du, dycdCt, dycdyaw, dudCt, dudyaw, self.kw, self.sigma
        )

        return deficit, ddeficitdCt, ddeficitdyaw

    def REWS(self, x0, y0, r_disc=20, theta_disc=50):
        """
        Calculates the rotor effective wind speed over a disk of radius R
        located at downstream and lateral location (x0, y0) relative to the wake
        source. Disk is assumed to be perpendicular to the freestream direction
        (x). The problem is extended from 2 to 3 dimensions to more accurately
        perform the numerical integration.
        """
        # Define points over rotor disk on polar grid
        rs = np.linspace(0, 0.5, r_disc)
        thetas = np.linspace(0, 2 * np.pi, theta_disc)

        r_mesh, theta_mesh = np.meshgrid(rs, thetas)
        ys = r_mesh * np.sin(theta_mesh) + y0
        zs = r_mesh * np.cos(theta_mesh)

        # Evaluate the deficit at points (converted to cartesian).
        deficit, ddeficitdCt, ddeficitdyaw = self.deficit(x0, ys, zs)

        # Perform integration over rotor disk in polar coordinates.
        REWS = 1 - np.trapz(np.trapz(r_mesh * deficit, r_mesh, axis=1), thetas)
        dREWSdCt = -np.trapz(np.trapz(r_mesh * ddeficitdCt, r_mesh, axis=1), thetas)
        dREWSdyaw = -np.trapz(np.trapz(r_mesh * ddeficitdyaw, r_mesh, axis=1), thetas)
        return REWS, dREWSdCt, dREWSdyaw

    def Cp1(self):
        # currently assumes freestream. Todo: add custon REWS
        Cp1, dCp1dCt, dCp1dyaw = calculate_Cp1(
            self.Ctprime, self.yaw, 1, self.a, 0, 0, self.dadCt, self.dadyaw
        )

        return Cp1, dCp1dCt, dCp1dyaw

    def Cp2(self, x0, y0, Ct2=2, yaw2=0):
        REWS, dREWSdCt, dREWSdyaw = self.REWS(x0, y0)
        a2, *_ = calculate_induction(Ct2, yaw2)

        Cp2, dCp2dCt, dCp2dyaw = calculate_Cp2(
            Ct2, yaw2, REWS, a2, dREWSdCt, dREWSdyaw, self.dadCt, self.dadyaw
        )

        return Cp2, dCp2dCt, dCp2dyaw
