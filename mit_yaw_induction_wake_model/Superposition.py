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
        Xs = [turbine.x for turbine in windfarm.turbines]
        N = len(Xs)
        REWS = np.zeros_like(Xs, dtype=float)
        dREWSdCts = np.zeros((N, N))
        dREWSdyaws = np.zeros((N, N))

        upstream_idx = []
        # Iterate through turbines from upstream to downstream
        for idx in np.argsort(Xs):
            # Calculate REWS of current turbine
            U_rotor = np.ones_like(deficits[0, 0, :])
            for idx_u in upstream_idx:
                U_rotor -= REWS[idx_u] * deficits[idx_u, idx, :]
            REWS[idx] = REWS_method.integrate(U_rotor)

            # Determine the effect of each upstream turbine on current turbine
            for idx_u in upstream_idx:
                dU_rotor_dCt = REWS[idx_u] * ddeficitdCts[idx_u, idx, ...]
                dU_rotor_dyaw = REWS[idx_u] * ddeficitdyaws[idx_u, idx, ...]
                for idx_uu in upstream_idx:
                    dU_rotor_dCt += dREWSdCts[idx_u, idx_uu] * deficits[idx_uu, idx]
                    dU_rotor_dyaw += dREWSdyaws[idx_u, idx_uu] * deficits[idx_uu, idx]

                dREWSdCts[idx_u, idx] = -REWS_method.integrate(dU_rotor_dCt)
                dREWSdyaws[idx_u, idx] = -REWS_method.integrate(dU_rotor_dyaw)

            upstream_idx.append(idx)
        return REWS, dREWSdCts, dREWSdyaws


class QuadraticNiayifar:
    pass


class Zong:
    pass
