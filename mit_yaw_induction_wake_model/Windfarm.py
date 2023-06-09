import numpy as np
from mit_yaw_induction_wake_model import Turbine


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
    def __init__(self, xs, ys, Cts, yaws, sigmas=None, kwts=None):
        assert all(len(Cts) == len(x) for x in [yaws, xs, ys])

        self.turbines = []
        for Ct, yaw, x, y in zip(Cts, yaws, xs, ys):
            self.turbines.append(Turbine.GradientTurbine(Ct, yaw, x, y))

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

    def REWS_at_rotors(self, r_disc=20, theta_disc=50):
        """
        Calculates the rotor effective wind speed over a disk of radius R
        located at downstream and lateral location (x0, y0) relative to the wake
        source. Disk is assumed to be perpendicular to the freestream direction
        (x). The problem is extended from 2 to 3 dimensions to more accurately
        perform the numerical integration.
        """
        # Define points over rotor disk on polar grid
        N = len(self.turbines)

        rs = np.linspace(0, 0.5, r_disc)
        thetas = np.linspace(0, 2 * np.pi, theta_disc)

        r_mesh, theta_mesh = np.meshgrid(rs, thetas)

        Xs = np.ones((N, theta_disc, r_disc))
        Ys = np.zeros((N, theta_disc, r_disc))
        Zs = np.zeros((N, theta_disc, r_disc))
        for i, turbine in enumerate(self.turbines):
            Xs[i, :, :] = turbine.x
            Ys[i, :, :] = r_mesh * np.sin(theta_mesh) + turbine.y
            Zs[i, :, :] = r_mesh * np.cos(theta_mesh)

        # Evaluate the deficit at points (converted to cartesian). Ignore self
        U = np.zeros((N, theta_disc, r_disc))
        dUdCt = np.zeros((N, N, theta_disc, r_disc))
        dUdyaw = np.zeros((N, N, theta_disc, r_disc))
        for i in range(N):
            U[i, :, :], dUdCt[i, :, :], dUdyaw[i, :, :] = self.wsp(
                Xs[i, :, :], Ys[i, :, :], Zs[i, :, :], ignore=[i]
            )

        # Perform integration over rotor disk in polar coordinates.
        REWS, dREWSdCt, dREWSdyaw = np.zeros(N), np.zeros((N, N)), np.zeros((N, N))
        for i, (_U, _dUdCt, _dUdyaw) in enumerate(zip(U, dUdCt, dUdyaw)):
            REWS[i] = np.trapz(np.trapz(r_mesh * _U, r_mesh, axis=1), thetas)
            dREWSdCt[:, i] = np.trapz(
                np.trapz(r_mesh * _dUdCt, r_mesh, axis=-1), thetas, axis=-1
            )
            dREWSdyaw[:, i] = np.trapz(
                np.trapz(r_mesh * _dUdyaw, r_mesh, axis=-1), thetas, axis=-1
            )

        return 4 / np.pi * REWS, 4 / np.pi * dREWSdCt, 4 / np.pi * dREWSdyaw

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
