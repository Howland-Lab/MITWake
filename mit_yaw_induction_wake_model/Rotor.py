"""
Implementation of the yaw-thrust actuator disk model as described in 'Modelling
the induction, thrust and power of a yaw-misaligned actuator disk' Heck et al.
2023.
"""

import numpy as np


def fixedpointiteration(f, x0, args=(), eps=0.000001, maxiter=100):
    for _ in range(maxiter):
        residuals = f(x0, *args)

        x0 += residuals
        if np.abs(residuals).max() < eps:
            break
    else:
        raise ValueError("max iterations reached.")

    return x0


def yawthrustlimited(Ctprime, yaw, Uamb=1):
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


def _yawthrust_residual(x, Ctprime, yaw, Uamb):
    """
    residual of Eq. 2.15
    """
    a, u4, v4 = x
    e_a = (
        1
        - np.sqrt(Uamb**2 - u4**2 - v4**2)
        / (np.sqrt(Ctprime) * Uamb * np.cos(yaw))
        - a
    )

    e_u4 = Uamb * (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2) - u4

    e_v4 = -Uamb * 0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2 - v4

    return np.array([e_a, e_u4, e_v4])


def _yawthrust_ddyaw_residual(x, Ct, yaw, a, u4, v4):
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


def _yawthrust_ddCt_residual(x, Ct, yaw, a, u4, v4):
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


def yawthrust(Ctprime, yaw, Uamb=1, eps=0.000001):
    """
    Solves Eq. 2.15.
    """
    _a, _u4, _v4 = yawthrustlimited(Ctprime, yaw, Uamb)

    a, u4, v4 = fixedpointiteration(
        _yawthrust_residual,
        np.array([_a, _u4, _v4]),
        args=(Ctprime, yaw, Uamb),
        eps=eps,
    )
    return a, u4, v4


def yawthrust_ddCt(Ct, yaw, a, u4, v4, eps=0.000001):
    """
    Calculate the derivatives of dadCt, dudCt, and dvdCt iteratively until convergence.

    Args:
        Ct: Value of Ct.
        yaw: Value of yaw.
        eps: Convergence threshold (default: 0.000001).

    Returns:
        Tuple: The values of dadCt, dudCt, and dvdCt.
    """
    x0 = np.array([0.0, 0.0, 0.0])

    dadCt, dudCt, dvdCt = fixedpointiteration(
        _yawthrust_ddCt_residual, x0, args=(Ct, yaw, a, u4, v4), eps=eps
    )
    return dadCt, dudCt, dvdCt


def yawthrust_ddyaw(Ct, yaw, a, u4, v4, eps=0.000001):
    """
    Calculate the derivatives of dadyaw, dudyaw, and dvdyaw iteratively until convergence.

    Args:
        yaw: Value of yaw.
        yaw: Value of yaw.
        eps: Convergence threshold (default: 0.000001).

    Returns:
        Tuple: The values of dadyaw, dudyaw, and dvdyaw.
    """
    x0 = np.array([0.0, 0.0, 0.0])

    dadyaw, dudyaw, dvdyaw = fixedpointiteration(
        _yawthrust_ddyaw_residual, x0, args=(Ct, yaw, a, u4, v4), eps=eps
    )
    return dadyaw, dudyaw, dvdyaw


def gradyawthrust(Ct, yaw, eps=0.000001):
    a, u4, v4 = yawthrust(Ct, yaw, eps=eps)

    dadCt, dudCt, dvdCt = yawthrust_ddCt(Ct, yaw, a, u4, v4, eps=eps)
    dadyaw, dudyaw, dvdyaw = yawthrust_ddyaw(Ct, yaw, a, u4, v4, eps=eps)

    return a, u4, v4, dadCt, dudCt, dvdCt, dadyaw, dudyaw, dvdyaw
