"""
Implementation of the yaw-thrust actuator disk model as described in 'Modelling
the induction, thrust and power of a yaw-misaligned actuator disk' Heck et al.
2023.
"""
from typing import Callable, Tuple
import numpy as np


def fixedpointiteration(
    f: Callable[[np.ndarray, any], np.ndarray],
    x0: np.ndarray,
    args=(),
    eps=0.000001,
    maxiter=100,
) -> np.ndarray:
    """
    Performs fixed-point iteration on function f until residuals converge or max
    iterations is reached.

    Args:
        f (Callable): residual function of form f(x, *args) -> np.ndarray
        x0 (np.ndarray): Initial guess
        args (tuple): arguments to pass to residual function. Defaults to ().
        eps (float): Convergence tolerance. Defaults to 0.000001.
        maxiter (int): Maximum number of iterations. Defaults to 100.

    Raises:
        ValueError: Max iterations reached.

    Returns:
        np.ndarray: Solution to residual function.
    """
    for _ in range(maxiter):
        residuals = f(x0, *args)

        x0 += residuals
        if np.abs(residuals).max() < eps:
            break
    else:
        raise ValueError("max iterations reached.")

    return x0


def yawthrustlimited(
    Ctprime: float, yaw: float, Uamb=1.0
) -> Tuple[float, float, float]:
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
    u4 = Uamb * (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
    v4 = Uamb * (
        -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
        / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
    )

    return a, u4, v4


def _yawthrust_residual(
    x: np.ndarray, Ctprime: float, yaw: float, Uamb=1.0
) -> np.ndarray:
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
    e_a = (
        1
        - np.sqrt((Uamb**2 - u4**2 - v4**2) / Ctprime)
        / (Uamb * np.cos(yaw))
        - a
    )

    e_u4 = Uamb * (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2) - u4

    e_v4 = -Uamb * 0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2 - v4

    return np.array([e_a, e_u4, e_v4])


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


def yawthrust(
    Ctprime: float, yaw: float, Uamb=1.0, eps=0.000001
) -> Tuple[float, float, float]:
    """Solves yawed-actuator disk model in Eq. 2.15.

    Args:
        Ctprime (float): Rotor thrust coefficient.
        yaw (float): Rotor yaw angle (radians).
        Uamb (float): Ambient wind velocity. Defaults to 1.0.
        eps (float): Convergence tolerance. Defaults to 0.000001.

    Returns:
        Tuple[float, float, float]: induction and outlet velocities.
    """

    _a, _u4, _v4 = yawthrustlimited(Ctprime, yaw, Uamb)

    a, u4, v4 = fixedpointiteration(
        _yawthrust_residual,
        np.array([_a, _u4, _v4]),
        args=(Ctprime, yaw, Uamb),
        eps=eps,
    )
    return a, u4, v4


def model_Cp(
        Ctprime: float, yaw: float, eps=0.000001, 
) -> float: 
    """
    Computes C_p = (1 - an(yaw))^3 * C_T' * cos^3(yaw)
    """
    a, u4, v4 = yawthrust(Ctprime, yaw, eps=eps)
    return (1 - a)**3 * Ctprime * np.cos(yaw)**3 


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
