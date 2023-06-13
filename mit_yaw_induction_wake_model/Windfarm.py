import numpy as np
from mit_yaw_induction_wake_model import Turbine
from mit_yaw_induction_wake_model import REWS as REWS_methods
from mit_yaw_induction_wake_model import Superposition


class Windfarm:
    def __init__(self, xs, ys, Cts, yaws, sigmas=None, kwts=None):
        assert all(len(Cts) == len(x) for x in [yaws, xs, ys])

    def wsp(self, x, y, z=0, ignore=[]):
        raise NotImplementedError

    def REWS_at_rotors(self, r_disc=20, theta_disc=50):
        raise NotImplementedError

    def turbine_Cp(self):
        raise NotImplementedError

    def total_Cp(self):
        raise NotImplementedError


class GradWindfarm:
    def __init__(
        self,
        xs,
        ys,
        Cts,
        yaws,
        REWS="area",
        summation="linear",
        sigmas=None,
        kwts=None,
        induction_eps=0.000001,
        REWS_params={},
    ):
        assert all(len(Cts) == len(x) for x in [yaws, xs, ys])

        self.turbines = []
        for Ct, yaw, x, y in zip(Cts, yaws, xs, ys):
            self.turbines.append(
                Turbine.GradientTurbine(Ct, yaw, x, y, induction_eps=induction_eps)
            )

        if REWS == "area":
            self.REWS_method = REWS_methods.Area(**REWS_params)
        elif REWS == "line":
            self.REWS_method = REWS_methods.Line(**REWS_params)
        elif REWS == "point":
            self.REWS_method = REWS_methods.Point(**REWS_params)
        else:
            raise ValueError(f"REWS {REWS} not found.")

        if summation == "linear":
            self.summation_method = Superposition.Linear()
        elif summation == "quadratic":
            raise NotImplementedError
        elif summation == "linearniayifar":
            raise NotImplementedError
        elif summation == "quadraticniayifar":
            raise NotImplementedError
        elif summation == "zong":
            raise NotImplementedError
        else:
            raise ValueError(f"Wake summation method {summation} not found.")

    def wsp(self, x, y, z=0, ignore=[]):
        dus, ddudCts, ddudyaws = [], [], []
        for turbine in self.turbines:
            du, ddudCt, ddudyaw = turbine.deficit(x, y, FOR="met")
            dus.append(du)
            ddudCts.append(ddudCt)
            ddudyaws.append(ddudyaw)
        for idx in ignore:
            dus[idx] = np.zeros_like(dus[idx])
            ddudCts[idx] = np.zeros_like(ddudCts[idx])
            ddudyaws[idx] = np.zeros_like(ddudyaws[idx])

        # Linear summation
        U = 1 - np.sum(dus, axis=0)
        dUdCt = -np.array(ddudCts)
        dUdyaw = -np.array(ddudyaws)

        return U, dUdCt, dUdyaw

    def REWS_at_rotors(self):
        # Get turbine locations
        N = len(self.turbines)
        X_t = np.array([turbine.x for turbine in self.turbines])
        Y_t = np.array([turbine.y for turbine in self.turbines])

        # Define gridpoints to sample based on REWS method.
        Xs, Ys, Zs = self.REWS_method.grid_points(X_t, Y_t)

        deficits, ddeficitdCts, ddeficitdyaws = (
            np.zeros((N, *Xs.shape)),
            np.zeros((N, *Xs.shape)),
            np.zeros((N, *Xs.shape)),
        )
        for i, turbine in enumerate(self.turbines):
            (
                deficits[i, :],
                ddeficitdCts[i, :],
                ddeficitdyaws[i, :],
            ) = turbine.deficit(Xs, Ys, Zs)

        # ignore effect of own wake on self
        for i in range(N):
            deficits[i, i, :] = 0
            ddeficitdCts[i, i, :] = 0
            ddeficitdyaws[i, i, :] = 0

        # Perform summation
        U, dUdCts, dUdyaws = self.summation_method(
            deficits, ddeficitdCts, ddeficitdyaws, self
        )

        # integrate
        U = self.REWS_method.integrate(U)
        dUdCts = self.REWS_method.integrate(dUdCts)
        dUdyaws = self.REWS_method.integrate(dUdyaws)

        return U, dUdCts, dUdyaws

    def turbine_Cp(self):
        REWS, dREWSdCt, dREWSdyaw = self.REWS_at_rotors()

        a = np.array([x.a for x in self.turbines])
        Ct = np.array([x.Ct for x in self.turbines])

        dadCt = np.diag([x.dadCt for x in self.turbines])
        dadyaw = np.diag([x.dadyaw for x in self.turbines])

        yaw = np.array([x.yaw for x in self.turbines])

        Cp = Ct * ((1 - a) * np.cos(yaw) * REWS) ** 3

        temp = 3 * Ct * ((1 - a) * np.cos(yaw) * REWS) ** 2
        dCpdCt = (
            np.diag(((1 - a) * np.cos(yaw) * REWS) ** 3)
            + temp * (1 - a) * np.cos(yaw) * dREWSdCt
            - temp * np.cos(yaw) * REWS * dadCt
        )
        dCpdyaw = (
            temp * (1 - a) * np.cos(yaw) * dREWSdyaw
            - temp * np.cos(yaw) * REWS * dadyaw
            - temp * (1 - a) * np.sin(yaw) * np.diag(REWS)
        )
        return Cp, dCpdCt, dCpdyaw

    def total_Cp(self):
        Cp, dCpdCt, dCpdyaw = self.turbine_Cp()

        return np.mean(Cp), np.mean(dCpdCt, axis=1), np.mean(dCpdyaw, axis=1)
