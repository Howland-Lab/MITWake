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

    # np.testing.assert_array_almost_equal(REWS, [1.0, 0.60510138, 0.91372961])
    np.testing.assert_array_almost_equal(REWS, [1.0, 0.58803785, 0.91372961])
    # np.testing.assert_array_almost_equal(Cp, [0.52887482, 0.11001517, 0.45207377])
    np.testing.assert_array_almost_equal(Cp, [0.52887482, 0.10096806, 0.45207377])


def test_windfarm_line_linear():
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws, REWS="line")

    REWS, _dREWSdCt, _dREWSdyaw = windfarm._REWS_at_rotors()

    Cp, _dCpdCt, _dCpdyaw = windfarm.turbine_Cp()

    np.testing.assert_array_almost_equal(REWS, [1.0, 0.64027361, 0.88168824], decimal=5)
    np.testing.assert_array_almost_equal(
        Cp, [0.52887482, 0.13033618, 0.40616394], decimal=5
    )


def test_windfarm_area_linear():
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws, REWS="area")

    REWS, _dREWSdCt, _dREWSdyaw = windfarm._REWS_at_rotors()

    Cp, _dCpdCt, _dCpdyaw = windfarm.turbine_Cp()

    # np.testing.assert_array_almost_equal(REWS, [1.0, 0.63798711, 0.8892935])
    np.testing.assert_array_almost_equal(REWS, [1.0, 0.69865796, 0.90144461])
    # np.testing.assert_array_almost_equal(Cp, [0.52887482, 0.12894481, 0.41676533])
    np.testing.assert_array_almost_equal(Cp, [0.52887482, 0.16934095, 0.43408358])
