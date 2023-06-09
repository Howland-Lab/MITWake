import numpy as np
from mit_yaw_induction_wake_model import Turbine


class Windfarm:
    def __init__(self, xs, ys, Cts, yaws, sigmas=None, kwts=None):
        assert all(len(Cts) == len(x) for x in [yaws, xs, ys])


class GradWindfarm:
    def __init__(self, xs, ys, Cts, yaws, sigmas=None, kwts=None):
        assert all(len(Cts) == len(x) for x in [yaws, xs, ys])

        self.turbines = []
        for Ct, yaw, x, y in zip(Cts, yaws, xs, ys):
            self.turbines.append(Turbine.GradientTurbine(Ct, yaw, x, y))

    def wsp(self, x, y, z=0):
        dus, ddudCts, ddudyaws = [], [], []
        for turbine in self.turbines:
            du, ddudCt, ddudyaw = turbine.deficit(x, y, FOR="met")
            dus.append(du)
            ddudCts.append(ddudCt)
            ddudyaws.append(ddudyaw)

        # Linear summation
        U = 1 - np.sum(dus, axis=0)
        dUdCt = -np.sum(ddudCts, axis=0) # THIS IS WRONG! SHOULD REMAIN SEPARATE!
        dUdyaw = -np.sum(ddudyaws, axis=0) # THIS TOO

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
        rs = np.linspace(0, 0.5, r_disc)
        thetas = np.linspace(0, 2 * np.pi, theta_disc)

        r_mesh, theta_mesh = np.meshgrid(rs, thetas)

        Xs = np.ones((len(self.turbines), theta_disc, r_disc))
        Ys = np.zeros((len(self.turbines), theta_disc, r_disc))
        Zs = np.zeros((len(self.turbines), theta_disc, r_disc))
        for i, turbine in enumerate(self.turbines):
            Xs[i, :, :] = turbine.x
            Ys[i, :, :] = r_mesh * np.sin(theta_mesh) + turbine.y
            Zs[i, :, :] = r_mesh * np.cos(theta_mesh)

        # Evaluate the deficit at points (converted to cartesian).
        deficit, ddeficitdCt, ddeficitdyaw = self.wsp(Xs, Ys, Zs)
        # Perform integration over rotor disk in polar coordinates.
        REWS, dREWSdCt, dREWSdyaw = [], [], []
        for _deficit, _ddeficitdCt, _ddeficitdyaw in zip(
            deficit, ddeficitdCt, ddeficitdyaw
        ):
            REWS.append(
                1 - np.trapz(np.trapz(r_mesh * _deficit, r_mesh, axis=1), thetas)
            )
            dREWSdCt.append(
                -np.trapz(np.trapz(r_mesh * _ddeficitdCt, r_mesh, axis=1), thetas)
            ) # ALSO WRONG! SHOULD HAVE 2 GRADIENTS PER TURBINE
            dREWSdyaw.append(
                -np.trapz(np.trapz(r_mesh * _ddeficitdyaw, r_mesh, axis=1), thetas)
            )# THIS TOO
        breakpoint()
        return REWS, dREWSdCt, dREWSdyaw
