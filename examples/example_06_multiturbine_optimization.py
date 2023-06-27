from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from MITWake import Windfarm

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


N = 100


def objective_func(x):
    Cts, yaws = x[:N], x[N:]
    farm = Windfarm.GradWindfarm(X, Y, Cts, yaws)
    Cp, dCpdCt, dCpdyaw = farm.total_Cp()
    return -Cp, -np.concatenate([dCpdCt, dCpdyaw])


def find_optimal_setpoints():
    res = optimize.minimize(
        objective_func,
        N * [1] + N * [0],
        bounds=N * [(0.00001, 5)] + N * [(-np.pi / 2 * 0.9, np.pi / 2 * 0.9)],
        jac=True,
    )

    return res


def plot_field(
    X, Y, field, turb_xs, turb_ys, yaws, ax, cmap=None, vmin=None, vmax=None, title=None
):
    ax.imshow(
        field,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    for turb_x, turb_y, yaw in zip(turb_xs, turb_ys, yaws):
        # Draw turbine
        R = 0.5
        p = np.array([[0, 0], [+R, -R]])
        rotmat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

        p = rotmat @ p + np.array([[turb_x], [turb_y]])

        ax.plot(p[0, :], p[1, :], "k", lw=1)

    ax.text(
        0.5,
        0.98,
        title,
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )


# https://stackoverflow.com/a/72226595
def sunflower(n: int, R: float, alpha: float) -> np.ndarray:
    # Number of points respectively on the boundary and inside the cirlce.
    n_exterior = np.round(alpha * np.sqrt(n)).astype(int)
    n_interior = n - n_exterior

    # Ensure there are still some points in the inside...
    if n_interior < 1:
        raise RuntimeError(
            f"Parameter 'alpha' is too large ({alpha}), all "
            f"points would end-up on the boundary."
        )
    # Generate the angles. The factor k_theta corresponds to 2*pi/phi^2.
    k_theta = np.pi * (3 - np.sqrt(5))
    angles = np.linspace(k_theta, k_theta * n, n)

    # Generate the radii.
    r_interior = np.sqrt(np.linspace(0, R**2, n_interior))
    r_exterior = R * np.ones((n_exterior,))
    r = np.concatenate((r_interior, r_exterior))

    return r * np.stack((np.cos(angles), np.sin(angles)))


if __name__ == "__main__":
    X, Y = sunflower(N, 20, 2)
    res = find_optimal_setpoints()

    Cts, yaws = res.x[:N], res.x[N:]

    print("Ct", Cts)
    print("yaw", yaws)

    Cp_ref, *_ = Windfarm.GradWindfarm(X, Y, N * [2], N * [0]).total_Cp()
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws)

    xs = np.linspace(X.min() - 2, X.max() + 2, 300)
    ys = np.linspace(Y.min() - 2, Y.max() + 2, 300)

    xmesh, ymesh = np.meshgrid(xs, ys)

    U, dUdCt, dUdyaw = windfarm.wsp(xmesh, ymesh)

    Cp, dCpdCt, dCpdyaw = windfarm.total_Cp()

    print("dCpdCt:", dCpdCt)
    print("dCpdyaw:", dCpdyaw)

    plt.figure()
    plot_field(xmesh, ymesh, U, X, Y, yaws, plt.gca(), "YlGnBu_r", title="U")
    plt.title(f"power increase: {100 * (Cp/Cp_ref - 1):2.2f}%")
    plt.savefig(
        FIGDIR / "example_06_multiturbine_optimization", dpi=300, bbox_inches="tight"
    )
    plt.close()
