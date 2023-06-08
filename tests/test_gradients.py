import numpy as np
from mit_yaw_induction_wake_model import ActuatorDisk, Gradients


def calculate_ddCT_FD(Ct, yaw, dx=1e-10):
    """
    Calculate the derivative of a function using finite difference approximation.

    Args:
        Ct: Value of Ct.
        yaw: Value of yaw.
        dx: Finite difference step size.

    Returns:
        Tuple: The values of dadCt, dudCt, and dvdCt
    """
    a_plus, u_plus, v_plus = ActuatorDisk.calculate_induction(Ct + dx, yaw)
    a_minus, u_minus, v_minus = ActuatorDisk.calculate_induction(Ct - dx, yaw)
    dadCt = (a_plus - a_minus) / (2 * dx)
    dudCt = (u_plus - u_minus) / (2 * dx)
    dvdCt = (v_plus - v_minus) / (2 * dx)
    return dadCt, dudCt, dvdCt


def calculate_ddyaw_FD(Ct, yaw, dx=1e-10):
    """
    Calculate the derivative of a function using finite difference approximation.

    Args:
        Ct: Value of Ct.
        yaw: Value of yaw.
        dx: Finite difference step size.

    Returns:
        Tuple: The values of dadyaw, dudyaw, and dvdyaw
    """
    a_plus, u_plus, v_plus = ActuatorDisk.calculate_induction(Ct, yaw + dx)
    a_minus, u_minus, v_minus = ActuatorDisk.calculate_induction(Ct, yaw - dx)
    dadyaw = (a_plus - a_minus) / (2 * dx)
    dudyaw = (u_plus - u_minus) / (2 * dx)
    dvdyaw = (v_plus - v_minus) / (2 * dx)
    return dadyaw, dudyaw, dvdyaw


def test_induction_CT_gradients():
    Ct = 2
    yaw = np.deg2rad(-20)

    a, u4, v4 = ActuatorDisk.calculate_induction(Ct, yaw)

    print(a, u4, v4)

    dadCt, dudCt, dvdCt = Gradients.calculate_ddCT(Ct, yaw)
    dadCt_FD, dudCt_FD, dvdCt_FD = calculate_ddCT_FD(Ct, yaw)

    assert np.abs(dadCt - dadCt_FD) < 0.00001
    assert np.abs(dudCt - dudCt_FD) < 0.00001
    assert np.abs(dvdCt - dvdCt_FD) < 0.00001

    dadyaw, dudyaw, dvdyaw = Gradients.calculate_ddyaw(Ct, yaw)
    dadyaw_FD, dudyaw_FD, dvdyaw_FD = calculate_ddyaw_FD(Ct, yaw)

    assert np.abs(dadyaw - dadyaw_FD) < 0.00001
    assert np.abs(dudyaw - dudyaw_FD) < 0.00001
    assert np.abs(dvdyaw - dvdyaw_FD) < 0.00001

    print(dadCt, dudCt, dvdCt)
    print(dadCt_FD, dudCt_FD, dvdCt_FD)
    print()
    print(dadyaw, dudyaw, dvdyaw)
    print(dadyaw_FD, dudyaw_FD, dvdyaw_FD)
