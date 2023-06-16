import numpy as np


class Linear:
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


class Quadratic:
    pass


class LinearNiayifar:
    def calculate_REWS(
        self, deficits, ddeficitdCts, ddeficitdyaws, REWS_method, windfarm
    ):
        # Preintegrate the deficits
        deficits = REWS_method.integrate(deficits)
        ddeficitdCts = REWS_method.integrate(ddeficitdCts)
        ddeficitdyaws = REWS_method.integrate(ddeficitdyaws)

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
