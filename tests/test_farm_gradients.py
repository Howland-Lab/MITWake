import numpy as np
from mit_yaw_induction_wake_model import Windfarm
import pytest

Xs = np.array([0, 8])
Ys = np.array([0, 0.5])
Cts = np.array([2, 2])
yaws = np.array([20, 0])
dx = 1e-10


@pytest.fixture
def farm():
    return Windfarm.GradWindfarm(Xs, Ys, [2, 2], [20, 0])


@pytest.fixture
def farm_ct1_p():
    return Windfarm.GradWindfarm(Xs, Ys, [2 + dx, 2], [20, 0])


@pytest.fixture
def farm_ct1_m():
    return Windfarm.GradWindfarm(Xs, Ys, [2 - dx, 2], [20, 0])


@pytest.fixture
def farm_yaw1_p():
    return Windfarm.GradWindfarm(Xs, Ys, [2, 2], [20 + dx, 0])


@pytest.fixture
def farm_yaw1_m():
    return Windfarm.GradWindfarm(Xs, Ys, [2, 2], [20 - dx, 0])


def test_wsp_dT1(farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m):
    x, y = 2, 0.5
    wsp, dwspdCt, dwspdyaw = farm.wsp(x, y)

    dwspdCt_FD = (farm_ct1_p.wsp(x, y)[0] - farm_ct1_m.wsp(x, y)[0]) / (2 * dx)
    dwspdyaw_FD = (farm_yaw1_p.wsp(x, y)[0] - farm_yaw1_m.wsp(x, y)[0]) / (2 * dx)

    assert np.abs(dwspdCt[0] - dwspdCt_FD) < 0.00001
    assert np.abs(dwspdyaw[0] - dwspdyaw_FD) < 0.00001


def test_REWS_dT1(farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m):
    REWS, dREWSdCt, dREWSdyaw = farm.REWS_at_rotors()

    dREWSdCt_FD = (
        farm_ct1_p.REWS_at_rotors()[0][1] - farm_ct1_m.REWS_at_rotors()[0][1]
    ) / (2 * dx)
    dREWSdyaw_FD = (
        farm_yaw1_p.REWS_at_rotors()[0][1] - farm_yaw1_m.REWS_at_rotors()[0][1]
    ) / (2 * dx)

    assert np.abs(dREWSdCt[0][1] - dREWSdCt_FD) < 0.00001
    assert np.abs(dREWSdyaw[0][1] - dREWSdyaw_FD) < 0.00001


def test_turbine_Cp_dT1(farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m):
    Cp, dCpdCt, dCpdyaw = farm.turbine_Cp()
    dCp1dCt_FD = (farm_ct1_p.turbine_Cp()[0][0] - farm_ct1_m.turbine_Cp()[0][0]) / (
        2 * dx
    )
    dCp1dyaw_FD = (farm_yaw1_p.turbine_Cp()[0][0] - farm_yaw1_m.turbine_Cp()[0][0]) / (
        2 * dx
    )

    assert np.abs(dCpdCt[0][0] - dCp1dCt_FD) < 0.00001
    assert np.abs(dCpdyaw[0][0] - dCp1dyaw_FD) < 0.00001

    dCp2dCt_FD = (farm_ct1_p.turbine_Cp()[0][1] - farm_ct1_m.turbine_Cp()[0][1]) / (
        2 * dx
    )
    dCp2dyaw_FD = (farm_yaw1_p.turbine_Cp()[0][1] - farm_yaw1_m.turbine_Cp()[0][1]) / (
        2 * dx
    )

    assert np.abs(dCpdCt[0][1] - dCp2dCt_FD) < 0.00001
    assert np.abs(dCpdyaw[0][1] - dCp2dyaw_FD) < 0.00001


def test_total_Cp_dT1(farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m):
    Cp, dCpdCt, dCpdyaw = farm.total_Cp()
    dCpdCt_FD = (farm_ct1_p.total_Cp()[0] - farm_ct1_m.total_Cp()[0]) / (2 * dx)
    dCpdyaw_FD = (farm_yaw1_p.total_Cp()[0] - farm_yaw1_m.total_Cp()[0]) / (2 * dx)

    assert np.abs(dCpdCt[0] - dCpdCt_FD) < 0.00001
    assert np.abs(dCpdyaw[0] - dCpdyaw_FD) < 0.00001
