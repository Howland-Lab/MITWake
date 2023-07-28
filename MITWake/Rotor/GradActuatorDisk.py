"""
Implementation of the yaw-thrust actuator disk model as described in 'Modelling
the induction, thrust and power of a yaw-misaligned actuator disk' Heck et al.
2023.
"""
from typing import Tuple
import numpy as np
from BaseClass import RotorBase
from Utilities import fixedpointiteration


class ActuatorDisk(RotorBase):
    def __init__(self, REWS_method):
        self.REWS_method = REWS_method

    def gridpoints(self, yaw):
        return self.REWS_method.grid_points(0, 0)

    def initialize(self, windfield, Ctprime, yaw, eps=0.000001):
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

        self._REWS = self.REWS_method.integrate(windfield)

        self._a, self._u4, self._v4 = None  # !!!!?!?!?!?

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


def _yawthrust_ddyaw_residual(
    x: np.ndarray, Ct: float, yaw: float, a: float, u4: float, v4: float
) -> np.ndarray:
    """
    Calculate the derivatives of the variables dadyaw, dudyaw, and dvdyaw.

    Args:
        x (np.ndarray): (dadyaw, dudyaw, dvdyaw)
        Ctprime (float): Rotor thrust coefficient.
        yaw (float): Rotor yaw angle (radians).
        a (float): Rotor induction.
        u4 (float): Outlet longitudinal velocity.
        v4 (float): Outlet lateral velocity.

    Returns:
        np.ndarray: residuals of derivatives of induction and outlet velocities.
    """

    dadyaw, dudyaw, dvdyaw = x
    cosy, siny = np.cos(yaw), np.sin(yaw)
    e_dadyaw = (u4 * dudyaw + v4 * dvdyaw - Ct * (1 - a) ** 2 * cosy * siny) / (
        Ct * (1 - a) * cosy**2
    ) - dadyaw
    e_dudyaw = Ct * cosy * (0.5 * cosy * dadyaw + (1 - a) * siny) - dudyaw
    e_dvdyaw = (
        0.5
        * Ct
        * (1 - a)
        * (
            dadyaw * siny * cosy**2
            - 0.5 * (1 - a) * cosy**3
            + (1 - a) * siny**2 * cosy
        )
    ) - dvdyaw

    return np.array([e_dadyaw, e_dudyaw, e_dvdyaw])


def _yawthrust_ddCt_residual(
    x: np.ndarray, Ct: float, yaw: float, a: float, u4: float, v4: float
) -> np.ndarray:
    """
    Calculate the derivatives of the variables dadCt, dudCt, and dvdCt.

    Args:
        x (np.ndarray): (dadCt, dudCt, dvdCt)
        Ctprime (float): Rotor thrust coefficient.
        yaw (float): Rotor yaw angle (radians).
        a (float): Rotor induction.
        u4 (float): Outlet longitudinal velocity.
        v4 (float): Outlet lateral velocity.

    Returns:
        np.ndarray: residuals of derivatives of induction and outlet velocities.
    """
    dadCt, dudCt, dvdCt = x
    cosy, siny = np.cos(yaw), np.sin(yaw)
    cosy2 = cosy**2

    e_dadCt = ((1 - a) ** 2 * cosy2 + 2 * u4 * dudCt + 2 * v4 * dvdCt) / (
        2 * Ct * (1 - a) * cosy2
    ) - dadCt
    e_dudCt = 0.5 * Ct * dadCt * cosy2 - 0.5 * (1 - a) * cosy2 - dudCt
    e_dvdCt = (
        0.5 * Ct * (1 - a) * dadCt * siny * cosy2 - 0.25 * (1 - a) ** 2 * siny * cosy2
    ) - dvdCt

    return np.array([e_dadCt, e_dudCt, e_dvdCt])


def yawthrust_ddCt(
    Ct: float, yaw: float, a: float, u4: float, v4: float, eps=0.000001
) -> Tuple[float, float, float]:
    """
    Solves the yaw-actuator disk model derivative with respect to C_T'.

    Args:
        Ct (float): Rotor thrust coefficient.
        yaw (float): Rotor yaw angle (radians).
        a (float): Rotor axial induction.
        u4 (float): Longitudinal outlet velocity.
        v4 (float): Lateral outlet velocity
        eps (float, optional): Convergence tolerance. Defaults to 0.000001.

    Returns:
        Tuple[float, float, float]: induction and outlet velocity derivatives.
    """
    x0 = np.array([0.0, 0.0, 0.0])

    dadCt, dudCt, dvdCt = fixedpointiteration(
        _yawthrust_ddCt_residual, x0, args=(Ct, yaw, a, u4, v4), eps=eps
    )
    return dadCt, dudCt, dvdCt


def yawthrust_ddyaw(
    Ct: float, yaw: float, a: float, u4: float, v4: float, eps=0.000001
) -> Tuple[float, float, float]:
    """
    Solves the yaw-actuator disk model derivative with respect to yaw.

    Args:
        Ct (float): Rotor thrust coefficient.
        yaw (float): Rotor yaw angle (radians).
        a (float): Rotor axial induction.
        u4 (float): Longitudinal outlet velocity.
        v4 (float): Lateral outlet velocity
        eps (float, optional): Convergence tolerance. Defaults to 0.000001.

    Returns:
        Tuple[float, float, float]: induction and outlet velocity derivatives.
    """
    x0 = np.array([0.0, 0.0, 0.0])

    dadyaw, dudyaw, dvdyaw = fixedpointiteration(
        _yawthrust_ddyaw_residual, x0, args=(Ct, yaw, a, u4, v4), eps=eps
    )
    return dadyaw, dudyaw, dvdyaw


def gradyawthrust(
    Ct: float, yaw: float, eps=0.000001
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Solves the yawed actuator disck model and its derivatives with respect to
    C_T' and yaw.

    Args:
        Ct (float): Rotor thrust coefficient. yaw (float): Rotor yaw angle
        (radians).

        eps (float, optional): Convergence tolerance. Defaults to 0.000001.

    Returns:
        Tuple[float, float, float, float, float, float, float, float, float]:
        induction and outlet velocity and its derivatives: a, u4, v4, dadCt,
        dudCt, dvdCt, dadyaw, dudyaw, dvdyaw
    """
    a, u4, v4 = yawthrust(Ct, yaw, eps=eps)

    dadCt, dudCt, dvdCt = yawthrust_ddCt(Ct, yaw, a, u4, v4, eps=eps)
    dadyaw, dudyaw, dvdyaw = yawthrust_ddyaw(Ct, yaw, a, u4, v4, eps=eps)

    return a, u4, v4, dadCt, dudCt, dvdCt, dadyaw, dudyaw, dvdyaw
