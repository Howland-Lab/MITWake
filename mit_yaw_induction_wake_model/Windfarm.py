import numpy as np
from mit_yaw_induction_wake_model import Turbine
from mit_yaw_induction_wake_model import REWS as REWS_methods
from mit_yaw_induction_wake_model import Superposition


class Windfarm:
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
        numerical=False,
        induction_eps=0.000001,
        REWS_params={},
    ):
        N = len(xs)
        assert all(N == len(x) for x in [Cts, yaws, ys])

        # Convert kwts ans sigmas to list if single value is given
        if kwts is None or type(kwts) in [int, float]:
            kwts = N * [kwts]

        if sigmas is None or type(sigmas) in [int, float]:
            sigmas = N * [sigmas]

        # Iteratively instantiate turbines
        self.turbines = []
        for Ct, yaw, x, y, kw in zip(Cts, yaws, xs, ys, kwts):
            self.turbines.append(
                Turbine.BasicTurbine(Ct, yaw, x, y, induction_eps=induction_eps, kw=kw)
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
            self.summation_method = Superposition.LinearNiayifar()
        elif summation == "quadraticniayifar":
            raise NotImplementedError
        elif summation == "zong":
            raise NotImplementedError
        else:
            raise ValueError(f"Wake summation method {summation} not found.")

        self.analytical = (
            REWS == "line"
            and summation in ["linear", "linearniayifar"]
            and not numerical
        )

        self.REWS = self._REWS_at_rotors()

    def wsp(self, x, y, z=0):
        N = len(self.turbines)
        x, y, z = np.array(x), np.array(y), np.array(z)
        deficits = np.zeros((N, *x.shape))

        for i, turbine in enumerate(self.turbines):
            deficits[i, :] = turbine.deficit(x, y, z)

        U = self.summation_method.summation(deficits, self)

        return U

    def _REWS_at_rotors(self):
        if self.analytical:
            return self._REWS_at_rotors_analytical()
        else:
            return self._REWS_at_rotors_numerical()

    def _REWS_at_rotors_analytical(self):
        N = len(self.turbines)
        Xs = np.array([turbine.x for turbine in self.turbines])
        Ys = np.array([turbine.y for turbine in self.turbines])

        deficits = np.zeros((N, N))
        for i, turbine in enumerate(self.turbines):
            deficits[i, :] = turbine.wake.line_deficit(Xs - turbine.x, Ys - turbine.y)

        # ignore effect of own wake on self
        np.fill_diagonal(deficits, 0)

        # Wake summation
        U = self.summation_method.analytical_REWS(deficits, self)

        return U

    def _REWS_at_rotors_numerical(self):
        # Get turbine locations
        N = len(self.turbines)
        X_t = np.array([turbine.x for turbine in self.turbines])
        Y_t = np.array([turbine.y for turbine in self.turbines])

        # Define gridpoints to sample based on REWS method.
        Xs, Ys, Zs = self.REWS_method.grid_points(X_t, Y_t)

        deficits = np.zeros((N, *Xs.shape))
        for i, turbine in enumerate(self.turbines):
            deficits[i, :] = turbine.deficit(Xs, Ys, Zs)

        # ignore effect of own wake on self
        for i in range(N):
            deficits[i, i, :] = 0

        # Perform summation
        REWS = self.summation_method.calculate_REWS(deficits, self.REWS_method, self)

        return REWS

    def turbine_Cp(self):
        a = np.array([x.a for x in self.turbines])
        Ct = np.array([x.Ct for x in self.turbines])

        yaw = np.array([x.yaw for x in self.turbines])

        Cp = Ct * ((1 - a) * np.cos(yaw) * self.REWS) ** 3

        temp = 3 * Ct * ((1 - a) * np.cos(yaw) * self.REWS) ** 2
        return Cp

    def total_Cp(self):
        Cp = self.turbine_Cp()

        return np.mean(Cp)


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
        numerical=False,
        induction_eps=0.000001,
        REWS_params={},
    ):
        N = len(xs)
        assert all(N == len(x) for x in [Cts, yaws, ys])

        # Convert kwts ans sigmas to list if single value is given
        if kwts is None or type(kwts) in [int, float]:
            kwts = N * [kwts]

        if sigmas is None or type(sigmas) in [int, float]:
            sigmas = N * [sigmas]

        # Iteratively instantiate turbines
        self.turbines = []
        for Ct, yaw, x, y, kw in zip(Cts, yaws, xs, ys, kwts):
            self.turbines.append(
                Turbine.GradientTurbine(
                    Ct, yaw, x, y, induction_eps=induction_eps, kw=kw
                )
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
            self.summation_method = Superposition.LinearNiayifar()
        elif summation == "quadraticniayifar":
            raise NotImplementedError
        elif summation == "zong":
            raise NotImplementedError
        else:
            raise ValueError(f"Wake summation method {summation} not found.")

        self.analytical = (
            REWS == "line"
            and summation in ["linear", "linearniayifar"]
            and not numerical
        )

        self.REWS, self.dREWSdCt, self.dREWSdyaw = self._REWS_at_rotors()

    def wsp(self, x, y, z=0):
        N = len(self.turbines)
        x, y, z = np.array(x), np.array(y), np.array(z)
        deficits, ddeficitdCts, ddeficitdyaws = (
            np.zeros((N, *x.shape)),
            np.zeros((N, *x.shape)),
            np.zeros((N, *x.shape)),
        )
        for i, turbine in enumerate(self.turbines):
            (
                deficits[i, :],
                ddeficitdCts[i, :],
                ddeficitdyaws[i, :],
            ) = turbine.deficit(x, y, z)

        U, dUdCt, dUdyaw = self.summation_method.summation_grad(
            deficits, ddeficitdCts, ddeficitdyaws, self
        )

        return U, dUdCt, dUdyaw

    def _REWS_at_rotors(self):
        if self.analytical:
            return self._REWS_at_rotors_analytical()
        else:
            return self._REWS_at_rotors_numerical()

    def _REWS_at_rotors_analytical(self):
        N = len(self.turbines)
        Xs = np.array([turbine.x for turbine in self.turbines])
        Ys = np.array([turbine.y for turbine in self.turbines])

        deficits, ddeficitdCts, ddeficitdyaws = (
            np.zeros((N, N)),
            np.zeros((N, N)),
            np.zeros((N, N)),
        )
        for i, turbine in enumerate(self.turbines):
            (
                deficits[i, :],
                ddeficitdCts[i, :],
                ddeficitdyaws[i, :],
            ) = turbine.wake.line_deficit(Xs - turbine.x, Ys - turbine.y)

        # ignore effect of own wake on self
        np.fill_diagonal(deficits, 0)
        np.fill_diagonal(ddeficitdCts, 0)
        np.fill_diagonal(ddeficitdyaws, 0)

        # Wake summation
        U, dUdCts, dUdyaws = self.summation_method.analytical_REWS_grad(
            deficits, ddeficitdCts, ddeficitdyaws, self
        )

        return U, dUdCts, dUdyaws

    def _REWS_at_rotors_numerical(self):
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
        REWS, dREWSdCts, dREWSdyaws = self.summation_method.calculate_REWS_grad(
            deficits, ddeficitdCts, ddeficitdyaws, self.REWS_method, self
        )

        return REWS, dREWSdCts, dREWSdyaws

    def turbine_Cp(self):
        a = np.array([x.a for x in self.turbines])
        Ct = np.array([x.Ct for x in self.turbines])

        dadCt = np.diag([x.dadCt for x in self.turbines])
        dadyaw = np.diag([x.dadyaw for x in self.turbines])

        yaw = np.array([x.yaw for x in self.turbines])

        Cp = Ct * ((1 - a) * np.cos(yaw) * self.REWS) ** 3

        temp = 3 * Ct * ((1 - a) * np.cos(yaw) * self.REWS) ** 2
        dCpdCt = (
            np.diag(((1 - a) * np.cos(yaw) * self.REWS) ** 3)
            + temp * (1 - a) * np.cos(yaw) * self.dREWSdCt
            - temp * np.cos(yaw) * self.REWS * dadCt
        )
        dCpdyaw = (
            temp * (1 - a) * np.cos(yaw) * self.dREWSdyaw
            - temp * np.cos(yaw) * self.REWS * dadyaw
            - temp * (1 - a) * np.sin(yaw) * np.diag(self.REWS)
        )
        return Cp, dCpdCt, dCpdyaw

    def total_Cp(self):
        Cp, dCpdCt, dCpdyaw = self.turbine_Cp()

        return np.mean(Cp), np.mean(dCpdCt, axis=1), np.mean(dCpdyaw, axis=1)
