import numpy as np
from mit_yaw_induction_wake_model import Windfarm
import pytest

Xs = np.array([0, 4, 8])
Ys = np.array([0, 0, 0])
Cts = np.array([2, 2, 2])
yaws = np.array([20, 1, -1])
dx = 1e-10


@pytest.mark.parametrize("summation", ["linear", "linearniayifar"])
class TestClass:
    def test_analytical_flag(self, summation):
        windfarm_anal = Windfarm.GradWindfarm(
            Xs, Ys, Cts, yaws, REWS="line", summation=summation
        )
        windfarm_num = Windfarm.GradWindfarm(
            Xs, Ys, Cts, yaws, REWS="line", summation=summation, numerical=True
        )

        assert windfarm_anal.analytical == True
        assert windfarm_num.analytical == False
    
    
    def test_analytical_REWS(self, summation):
        windfarm_anal = Windfarm.GradWindfarm(
            Xs, Ys, Cts, yaws, REWS="line", summation=summation
        )
        windfarm_num = Windfarm.GradWindfarm(
            Xs, Ys, Cts, yaws, REWS="line", summation=summation, numerical=True
        )

        np.testing.assert_array_almost_equal(windfarm_anal.REWS, windfarm_num.REWS, decimal=5)
        np.testing.assert_array_almost_equal(windfarm_anal.dREWSdCt, windfarm_num.dREWSdCt, decimal=5)
        np.testing.assert_array_almost_equal(windfarm_anal.dREWSdyaw, windfarm_num.dREWSdyaw, decimal=5)
