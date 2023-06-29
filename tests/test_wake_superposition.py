import numpy as np
import pytest

from MITWake import Windfarm

X = [0, 4, 8]
Y = [0, -0.5, 0.5]
Cts = [2.11, 1, 2]
yaws = np.deg2rad([24, 10, 0])


def test_windfarm_point_linear():
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws, REWS="point")

    REWS, _dREWSdCt, _dREWSdyaw = windfarm._REWS_at_rotors()

    Cp, _dCpdCt, _dCpdyaw = windfarm.turbine_Cp()

    np.testing.assert_array_almost_equal(REWS, [1.0, 0.537983, 0.92791])
    np.testing.assert_array_almost_equal(Cp, [0.528875, 0.077317, 0.473449])


def test_windfarm_line_linear():
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws, REWS="line")

    REWS, _dREWSdCt, _dREWSdyaw = windfarm._REWS_at_rotors()

    Cp, _dCpdCt, _dCpdyaw = windfarm.turbine_Cp()

    np.testing.assert_array_almost_equal(REWS, [1.0, 0.60569, 0.89368], decimal=5)
    np.testing.assert_array_almost_equal(Cp, [0.52887, 0.11034, 0.42296], decimal=5)


def test_windfarm_area_linear():
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws, REWS="area")

    REWS, _dREWSdCt, _dREWSdyaw = windfarm._REWS_at_rotors()

    Cp, _dCpdCt, _dCpdyaw = windfarm.turbine_Cp()

    np.testing.assert_array_almost_equal(REWS, [1.0, 0.674466, 0.913121])
    np.testing.assert_array_almost_equal(Cp, [0.528875, 0.152352, 0.45117])
