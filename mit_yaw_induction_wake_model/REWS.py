import numpy as np


class Point:
    def grad_REWS_at_rotors(self, windfarm):
        N = len(windfarm.turbines)
        Xs = [turbine.x for turbine in windfarm.turbines]
        Ys = [turbine.y for turbine in windfarm.turbines]

        U = np.zeros((N))
        dUdCt = np.zeros((N, N))
        dUdyaw = np.zeros((N, N))
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            U[i], dUdCt[:, i], dUdyaw[:, i] = windfarm.wsp(
                np.array(x), np.array(y), ignore=[i]
            )

        return U, dUdCt, dUdyaw


class Line:
    def grad_REWS_at_rotors(self, windfarm):
        N = len(windfarm.turbines)
        Xs = np.array([turbine.x for turbine in windfarm.turbines])
        Ys = np.array([turbine.y for turbine in windfarm.turbines])

        deficits, ddeficitdCts, ddeficitdyaws = [], [], []
        for turbine in windfarm.turbines:
            _deficit, _ddeficitdCt, _ddeficitdyaw = turbine.wake.line_deficit(
                Xs - turbine.x, Ys - turbine.y
            )
            deficits.append(_deficit)
            ddeficitdCts.append(_ddeficitdCt)
            ddeficitdyaws.append(_ddeficitdyaw)
        deficits = np.array(deficits)
        ddeficitdCts = np.array(ddeficitdCts)
        ddeficitdyaws = np.array(ddeficitdyaws)

        # ignore effect of own wake on self
        np.fill_diagonal(deficits, 0)
        np.fill_diagonal(ddeficitdCts, 0)
        np.fill_diagonal(ddeficitdyaws, 0)

        # linear sum
        U = 1 - np.sum(deficits, axis=0)
        dUdCts = -ddeficitdCts
        dUdyaws = -ddeficitdyaws

        return U, dUdCts, dUdyaws


class Area:
    def __init__(self, r_disc=10, theta_disc=10):
        # predefine polar grid for performing REWS
        self.r_disc, self.theta_disc = r_disc, theta_disc
        rs = np.linspace(0, 0.5, r_disc)
        self.thetas = np.linspace(0, 2 * np.pi, theta_disc)

        self.r_mesh, self.theta_mesh = np.meshgrid(rs, self.thetas)

    def grad_REWS_at_rotors(self, windfarm):
        """
        Calculates the rotor effective wind speed over a disk of radius R
        located at downstream and lateral location (x0, y0) relative to the wake
        source. Disk is assumed to be perpendicular to the freestream direction
        (x). The problem is extended from 2 to 3 dimensions to more accurately
        perform the numerical integration.
        """
        # Define points over rotor disk on polar grid
        N = len(windfarm.turbines)

        Xs = np.ones((N, self.theta_disc, self.r_disc))
        Ys = np.zeros((N, self.theta_disc, self.r_disc))
        Zs = np.zeros((N, self.theta_disc, self.r_disc))
        for i, turbine in enumerate(windfarm.turbines):
            Xs[i, :, :] = turbine.x
            Ys[i, :, :] = self.r_mesh * np.sin(self.theta_mesh) + turbine.y
            Zs[i, :, :] = self.r_mesh * np.cos(self.theta_mesh)

        # Evaluate the deficit at points (converted to cartesian). Ignore self
        U = np.zeros((N, self.theta_disc, self.r_disc))
        dUdCt = np.zeros((N, N, self.theta_disc, self.r_disc))
        dUdyaw = np.zeros((N, N, self.theta_disc, self.r_disc))
        for i in range(N):
            U[i, :, :], dUdCt[i, :, :], dUdyaw[i, :, :] = windfarm.wsp(
                Xs[i, :, :], Ys[i, :, :], Zs[i, :, :], ignore=[i]
            )

        # Perform integration over rotor disk in polar coordinates.
        REWS, dREWSdCt, dREWSdyaw = np.zeros(N), np.zeros((N, N)), np.zeros((N, N))
        for i, (_U, _dUdCt, _dUdyaw) in enumerate(zip(U, dUdCt, dUdyaw)):
            REWS[i] = np.trapz(
                np.trapz(self.r_mesh * _U, self.r_mesh, axis=1), self.thetas
            )
            dREWSdCt[:, i] = np.trapz(
                np.trapz(self.r_mesh * _dUdCt, self.r_mesh, axis=-1),
                self.thetas,
                axis=-1,
            )
            dREWSdyaw[:, i] = np.trapz(
                np.trapz(self.r_mesh * _dUdyaw, self.r_mesh, axis=-1),
                self.thetas,
                axis=-1,
            )

        return 4 / np.pi * REWS, 4 / np.pi * dREWSdCt, 4 / np.pi * dREWSdyaw
