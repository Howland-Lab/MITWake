import numpy as np


class Linear:
    def summation(self, deficits, ddeficitdCts, ddeficitdyaws, windfarm):
        U = 1 - np.sum(deficits, axis=0)
        dUdCts = -ddeficitdCts
        dUdyaws = -ddeficitdyaws

        return U, dUdCts, dUdyaws

    def calculate_REWS(
        self, deficits, ddeficitdCts, ddeficitdyaws, REWS_method, windfarm
    ):
        U = 1 - np.sum(deficits, axis=0)
        dUdCts = -ddeficitdCts
        dUdyaws = -ddeficitdyaws

        REWS = REWS_method.integrate(U)
        dREWSdCt = REWS_method.integrate(dUdCts)
        dREWSdyaw = REWS_method.integrate(dUdyaws)

        return REWS, dREWSdCt, dREWSdyaw

    def analytical_REWS(self, *args):
        return self.summation(*args)


class Quadratic:
    pass


class LinearNiayifar:
    def summation(self, deficits, ddeficitdCts, ddeficitdyaws, windfarm):
        REWS, dREWSdCt, dREWSdyaw = windfarm.REWS, windfarm.dREWSdCt, windfarm.dREWSdyaw
        # broadcast REWS to the shape of deficits
        REWS = REWS[(...,) + (np.newaxis,) * (deficits.ndim - 1)]
        dREWSdCt = dREWSdCt[(...,) + (np.newaxis,) * (deficits.ndim - 1)]
        dREWSdyaw = dREWSdyaw[(...,) + (np.newaxis,) * (deficits.ndim - 1)]

        U = 1 - np.sum(REWS * deficits, axis=0)
        dUdCts = -(np.sum(dREWSdCt * deficits, axis=1) + REWS * ddeficitdCts)
        dUdyaws = -(np.sum(dREWSdyaw * deficits, axis=1) + REWS * ddeficitdyaws)

        return U, dUdCts, dUdyaws

    def calculate_REWS(
        self, deficits, ddeficitdCts, ddeficitdyaws, REWS_method, windfarm
    ):
        # Preintegrate the deficits
        deficits = REWS_method.integrate(deficits)
        ddeficitdCts = REWS_method.integrate(ddeficitdCts)
        ddeficitdyaws = REWS_method.integrate(ddeficitdyaws)

        return self.analytical_REWS(deficits, ddeficitdCts, ddeficitdyaws, windfarm)

    def analytical_REWS(self, deficits, ddeficitdCts, ddeficitdyaws, windfarm):
        Xs = [turbine.x for turbine in windfarm.turbines]
        N = len(Xs)
        REWS = np.zeros_like(Xs, dtype=float)
        dREWSdCts = np.zeros((N, N))
        dREWSdyaws = np.zeros((N, N))

        upstream_idx = []
        # Iterate through turbines from upstream to downstream
        for idx in np.argsort(Xs):
            REWS[idx] = 1 - np.sum(REWS * deficits[:, idx])

            # Determine the effect of each upstream turbine on current turbine
            for idx_u in upstream_idx:
                dREWSdCts[idx_u, idx] = -REWS[idx_u] * ddeficitdCts[
                    idx_u, idx
                ] - np.sum(dREWSdCts[idx_u, :] * deficits[:, idx])
                dREWSdyaws[idx_u, idx] = -REWS[idx_u] * ddeficitdyaws[
                    idx_u, idx
                ] - np.sum(dREWSdyaws[idx_u, :] * deficits[:, idx])

            upstream_idx.append(idx)

        return REWS, dREWSdCts, dREWSdyaws


class QuadraticNiayifar:
    pass


class Zong:
    pass
