"""
Implementation of the yaw-thrust actuator disk model as described in 'Modelling
the induction, thrust and power of a yaw-misaligned actuator disk' Heck et al.
2023.
"""
from typing import Tuple
import numpy as np
from ..BaseClasses import RotorBase
from ..Utilities import fixedpointiteration
from ..REWS import Point


class ActuatorDisk(RotorBase):
    def __init__(self, x=0.0, y=0.0, z=0.0, REWS_method=Point):
        """_summary_

        Args:
            x (float): diameter-normalized longitudinal location of rotor center. Defaults to 0.
            y (float): diameter-normalized lateral location of rotor center. Defaults to 0.
            z (float): diameter-normalized vertical location of rotor center. Defaults to 0.
            REWS_method (_type_, optional): _description_. Defaults to Point.
        """
        self._x, self._y, self._z = x, y, z
        self.REWS_method = REWS_method

    def gridpoints(self, _Ctprime, yaw):
        """
        Returns grid points over rotor in meteorological coordinates normalized
        by rotor diameter.
        """
        X, Y, Z = self.REWS_method.grid_points(self.x, self.y)

        return X + self.x, Y + self.y, Z + self.z

    def initialize(self, Ctprime, yaw, windfield=None, eps=0.000001):
        """Solves yawed-actuator disk model in Eq. 2.15.

        Args:
            Windfield (ndarray): Longitudinal wind speed sampled at grid points.
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).
            Uamb (float): Ambient wind velocity. Defaults to 1.0.
            eps (float): Convergence tolerance. Defaults to 0.000001.

        Returns:
            Tuple[float, float, float]: induction and outlet velocities.
        """
        self._Ctprime, self._yaw = Ctprime, yaw

        if windfield:
            self._REWS = self.REWS_method.integrate(windfield)
        else:
            self._REWS = 1

        _a, _u4, _v4 = yawthrustlimited(Ctprime, yaw)

        converged, (self._a, self._u4, self._v4) = fixedpointiteration(
            _yawthrust_residual,
            np.array([_a, _u4, _v4]),
            args=(Ctprime, yaw),
            eps=eps,
        )

    def REWS(self):
        return self._REWS

    def Cp(self):
        Cp = self._Ctprime * ((1 - self._a) * np.cos(self._yaw) * self._REWS) ** 3
        return Cp

    def Ct(self):
        Ct = (1 - self._a) ** 2 * np.cos(self._yaw) ** 2 * self._Ctprime
        return Ct

    def Ctprime(self):
        return self._Ctprime

    def a(self):
        return self._a

    def u4(self):
        return self._u4

    def v4(self):
        return self._v4


def yawthrustlimited(Ctprime: float, yaw: float) -> Tuple[float, float, float]:
    """
    Solves the limiting case when v_4 << u_4. (Eq. 2.19, 2.20). Also takes Numpy
    array arguments.

    Args:
        Ctprime (float): Rotor thrust coefficient.
        yaw (float): Rotor yaw angle (radians).
        Uamb (float): Ambient wind velocity. Defaults to 1.0.

    Returns:
        Tuple[float, float, float]: induction and outlet velocities.
    """
    a = Ctprime * np.cos(yaw) ** 2 / (4 + Ctprime * np.cos(yaw) ** 2)
    u4 = (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
    v4 = (
        -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
        / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
    )

    return a, u4, v4


def _yawthrust_residual(x: np.ndarray, Ctprime: float, yaw: float) -> np.ndarray:
    """
    Residual function of yawed-actuator disk model in Eq. 2.15.

    Args:
        x (np.ndarray): (a, u4, v4)
        Ctprime (float): Rotor thrust coefficient.
        yaw (float): Rotor yaw angle (radians).
        Uamb (float): Ambient wind velocity. Defaults to 1.0.

    Returns:
        np.ndarray: residuals of induction and outlet velocities.
    """

    a, u4, v4 = x
    e_a = 1 - np.sqrt(1 - u4**2 - v4**2) / (np.sqrt(Ctprime) * np.cos(yaw)) - a

    e_u4 = (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2) - u4

    e_v4 = -0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2 - v4

    return np.array([e_a, e_u4, e_v4])
