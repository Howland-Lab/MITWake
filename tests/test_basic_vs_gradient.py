import numpy as np
from mit_yaw_induction_wake_model import Windfarm
import pytest

Xs = np.array([0, 4, 8])
Ys = np.array([0, 0, 0])
Cts = np.array([2, 2, 2])
yaws = np.array([20, 1, -1])
dx = 1e-10


@pytest.mark.parametrize("summation", ["linear", "linearniayifar"])
@pytest.mark.parametrize("numerical", [True, False])
class TestClass:
    def test_analytical_REWS(self, summation, numerical):
        windfarm_basic = Windfarm.Windfarm(
            Xs, Ys, Cts, yaws, REWS="line", summation=summation, numerical=numerical
        )
        windfarm_grad = Windfarm.GradWindfarm(
            Xs, Ys, Cts, yaws, REWS="line", summation=summation, numerical=numerical
        )

        np.testing.assert_array_almost_equal(
            windfarm_basic.REWS, windfarm_grad.REWS, decimal=5
        )
        np.testing.assert_array_almost_equal(
            windfarm_basic.total_Cp(), windfarm_grad.total_Cp()[0], decimal=5
        )
        np.testing.assert_array_almost_equal(
            windfarm_basic.turbine_Cp(), windfarm_grad.turbine_Cp()[0], decimal=5
        )

        x, y = np.linspace(Xs.min() - 1, Xs.max() + 1, 100), np.linspace(
            Ys.min() - 1, Ys.max() + 1, 100
        )
        xmesh, ymesh = np.meshgrid(x, y)

        np.testing.assert_array_almost_equal(
            windfarm_basic.wsp(xmesh, ymesh),
            windfarm_grad.wsp(xmesh, ymesh)[0],
            decimal=5,
        )
