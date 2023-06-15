import numpy as np
from mit_yaw_induction_wake_model import Windfarm
import pytest

Xs = np.array([0, 4, 8])
Ys = np.array([0, 4, 0.5])
Cts = np.array([2, 2, 2])
yaws = np.array([20, 1, -1])
dx = 1e-10


def get_perturbed(idx, REWS, summation):
    e = np.zeros_like(Xs, dtype=float)
    e[idx] = dx

    farm = Windfarm.GradWindfarm(Xs, Ys, Cts, yaws, REWS=REWS, summation=summation)
    farm_ct1_p = Windfarm.GradWindfarm(
        Xs, Ys, Cts + e, yaws, REWS=REWS, summation=summation
    )
    farm_ct1_m = Windfarm.GradWindfarm(
        Xs, Ys, Cts - e, yaws, REWS=REWS, summation=summation
    )
    farm_yaw1_p = Windfarm.GradWindfarm(
        Xs, Ys, Cts, yaws + e, REWS=REWS, summation=summation
    )
    farm_yaw1_m = Windfarm.GradWindfarm(
        Xs, Ys, Cts, yaws - e, REWS=REWS, summation=summation
    )

    return farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m


@pytest.mark.parametrize("REWS", ["point", "line", "area"])
@pytest.mark.parametrize("summation", ["linear", "linearniayifar"])
@pytest.mark.parametrize("perturbed_idx", [0, 1, 2])
class TestClass:
    # def test_wsp_dT1(self, perturbed_idx, REWS, summation):
    #     farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m = get_perturbed(
    #         perturbed_idx, REWS, summation
    #     )
    #     x, y = 2, 0.5
    #     wsp, dwspdCt, dwspdyaw = farm.wsp(x, y)

    #     dwspdCt_FD = (farm_ct1_p.wsp(x, y)[0] - farm_ct1_m.wsp(x, y)[0]) / (2 * dx)
    #     dwspdyaw_FD = (farm_yaw1_p.wsp(x, y)[0] - farm_yaw1_m.wsp(x, y)[0]) / (2 * dx)

    #     assert np.abs(dwspdCt[0] - dwspdCt_FD) < 0.00001
    #     assert np.abs(dwspdyaw[0] - dwspdyaw_FD) < 0.00001

    def test_REWS_dT1(self, perturbed_idx, REWS, summation):
        farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m = get_perturbed(
            perturbed_idx, REWS, summation
        )

        REWS, dREWSdCt, dREWSdyaw = farm.REWS_at_rotors()

        dREWSdCt_FD = (
            farm_ct1_p.REWS_at_rotors()[0] - farm_ct1_m.REWS_at_rotors()[0]
        ) / (2 * dx)
        dREWSdyaw_FD = (
            farm_yaw1_p.REWS_at_rotors()[0] - farm_yaw1_m.REWS_at_rotors()[0]
        ) / (2 * dx)

        np.testing.assert_array_almost_equal(
            dREWSdCt[perturbed_idx], dREWSdCt_FD, decimal=5
        )
        np.testing.assert_array_almost_equal(
            dREWSdyaw[perturbed_idx], dREWSdyaw_FD, decimal=5
        )

    def test_turbine_Cp_dT1(self, perturbed_idx, REWS, summation):
        farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m = get_perturbed(
            perturbed_idx, REWS, summation
        )
        Cp, dCpdCt, dCpdyaw = farm.turbine_Cp()
        dCpdCt_FD = (farm_ct1_p.turbine_Cp()[0] - farm_ct1_m.turbine_Cp()[0]) / (2 * dx)
        dCpdyaw_FD = (farm_yaw1_p.turbine_Cp()[0] - farm_yaw1_m.turbine_Cp()[0]) / (
            2 * dx
        )

        np.testing.assert_array_almost_equal(
            dCpdCt[perturbed_idx], dCpdCt_FD, decimal=5
        )
        np.testing.assert_array_almost_equal(
            dCpdyaw[perturbed_idx], dCpdyaw_FD, decimal=5
        )

    def test_total_Cp_dT1(self, perturbed_idx, REWS, summation):
        farm, farm_ct1_p, farm_ct1_m, farm_yaw1_p, farm_yaw1_m = get_perturbed(
            perturbed_idx, REWS, summation
        )
        Cp, dCpdCt, dCpdyaw = farm.total_Cp()
        dCpdCt_FD = (farm_ct1_p.total_Cp()[0] - farm_ct1_m.total_Cp()[0]) / (2 * dx)
        dCpdyaw_FD = (farm_yaw1_p.total_Cp()[0] - farm_yaw1_m.total_Cp()[0]) / (2 * dx)

        np.testing.assert_array_almost_equal(
            dCpdCt[perturbed_idx], dCpdCt_FD, decimal=5
        )
        np.testing.assert_array_almost_equal(
            dCpdyaw[perturbed_idx], dCpdyaw_FD, decimal=5
        )
