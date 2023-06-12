"""
Implementation of the yaw-thrust actuator disk model as described in 'Modelling
the induction, thrust and power of a yaw-misaligned actuator disk' Heck et al.
2023.
"""

import numpy as np


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


def _yawthrustsubiteration(Ctprime, yaw, Uamb, a, u4, v4):
    """
    Subiteration of Eq. 2.15
    """
    _a = 1 - np.sqrt(Uamb**2 - u4**2 - v4**2) / (
        np.sqrt(Ctprime) * Uamb * np.cos(yaw)
    )

    _u4 = Uamb * (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2)

    _v4 = -Uamb * 0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2

    return _a, _u4, _v4


def yawthrust(Ctprime, yaw, Uamb=1, eps=0.000001):
    """
    Solves Eq. 2.15.
    """
    a_guess, u4_guess, v4_guess = yawthrustlimited(Ctprime, yaw, Uamb)

    niter = 1
    converged = False
    while not converged:
        a, u4, v4 = _yawthrustsubiteration(
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


def _yawthrust_ddCt_subiteration(dadCt, dudCt, dvdCt, Ct, yaw, a, u4, v4):
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
    cosy, siny = np.cos(yaw), np.sin(yaw)
    cosy2 = cosy**2

    dadCt_new = ((1 - a) ** 2 * cosy2 + 2 * u4 * dudCt + 2 * v4 * dvdCt) / (
        2 * Ct * (1 - a) * cosy2
    )
    dudCt_new = 0.5 * Ct * dadCt * cosy2 - 0.5 * (1 - a) * cosy2
    dvdCt_new = (
        0.5 * Ct * (1 - a) * dadCt * siny * cosy2 - 0.25 * (1 - a) ** 2 * siny * cosy2
    )

    return dadCt_new, dudCt_new, dvdCt_new


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
    dadCt, dudCt, dvdCt = 0, 0, 0

    niter = 0
    converged = False
    while not converged:
        dadCt_new, dudCt_new, dvdCt_new = _yawthrust_ddCt_subiteration(
            dadCt, dudCt, dvdCt, Ct, yaw, a, u4, v4
        )

        dadCt_error = np.abs(dadCt - dadCt_new)
        dudCt_error = np.abs(dudCt - dudCt_new)
        dvdCt_error = np.abs(dvdCt - dvdCt_new)

        dadCt, dudCt, dvdCt = dadCt_new, dudCt_new, dvdCt_new

        niter += 1
        converged = all(x < eps for x in [dadCt_error, dudCt_error, dvdCt_error])

    return dadCt, dudCt, dvdCt


def _yawthrust_ddyaw_subiteration(dadyaw, dudyaw, dvdyaw, Ct, yaw, a, u4, v4):
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
    cosy, siny = np.cos(yaw), np.sin(yaw)
    dadyaw_new = (u4 * dudyaw + v4 * dvdyaw - Ct * (1 - a) ** 2 * cosy * siny) / (
        Ct * (1 - a) * cosy**2
    )
    dudyaw_new = Ct * cosy * (0.5 * cosy * dadyaw + (1 - a) * siny)
    dvdyaw_new = (
        0.5
        * Ct
        * (1 - a)
        * (
            dadyaw * siny * cosy**2
            - 0.5 * (1 - a) * cosy**3
            + (1 - a) * siny**2 * cosy
        )
    )

    return dadyaw_new, dudyaw_new, dvdyaw_new


def yawthrust_ddyaw(Ct, yaw, a, u4, v4, eps=0.000001):
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
        dayaw_new, duyaw_new, dvyaw_new = _yawthrust_ddyaw_subiteration(
            dayaw, duyaw, dvyaw, Ct, yaw, a, u4, v4
        )

        dayaw_error = np.abs(dayaw - dayaw_new)
        duyaw_error = np.abs(duyaw - duyaw_new)
        dvyaw_error = np.abs(dvyaw - dvyaw_new)

        dayaw, duyaw, dvyaw = dayaw_new, duyaw_new, dvyaw_new

        niter += 1
        converged = all(x < eps for x in [dayaw_error, duyaw_error, dvyaw_error])

    return dayaw, duyaw, dvyaw


def gradyawthrust(Ct, yaw, eps=0.000001):
    a, u4, v4 = yawthrust(Ct, yaw, eps=eps)

    dadCt, dudCt, dvdCt = yawthrust_ddCt(Ct, yaw, a, u4, v4, eps=eps)
    dadyaw, dudyaw, dvdyaw = yawthrust_ddyaw(Ct, yaw, a, u4, v4, eps=eps)

    return a, u4, v4, dadCt, dudCt, dvdCt, dadyaw, dudyaw, dvdyaw


if __name__ == "__main__":
    pass
