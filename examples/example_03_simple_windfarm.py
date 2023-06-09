from pathlib import Path
import numpy as np
from mit_yaw_induction_wake_model import Windfarm

import matplotlib.pyplot as plt


FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


X = [0, 8]
Y = [0, 0.5]
Cts = [2.11, 2]
yaws = np.deg2rad([24, 0])

TITLES = ["$U$", "$dU/dC_t$", "$dU/d\gamma$"]
cmaps = ["YlGnBu_r", "RdYlGn", "RdYlGn"]
vmins = [None, -0.5, -2]
vmaxs = [None, 0.5, 2]
if __name__ == "__main__":
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws)

    xs = np.linspace(-1, 12, 300)
    ys = np.linspace(-1.5, 1.5, 300)

    xmesh, ymesh = np.meshgrid(xs, ys)

    U, dUdCt, dUdyaw = windfarm.wsp(xmesh, ymesh)

    # REWS = windfarm.REWS_at_rotors()

    fig, axes = plt.subplots(3, 1, sharex=True)

    for ax, field, title, cmap, vmin, vmax in zip(
        axes, [U, dUdCt, dUdyaw], TITLES, cmaps, vmins, vmaxs
    ):
        ax.imshow(
            field,
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        for turbine in windfarm.turbines:
            # Draw turbine
            turb_x, turb_y, yaw = turbine.x, turbine.y, turbine.yaw
            R = 0.5
            p = np.array([[turb_x, turb_x], [turb_y + R, turb_y - R]])
            rotmat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

            p = rotmat @ p

            ax.plot(p[0, :], p[1, :], "k", lw=5)

        ax.text(
            0.5,
            0.98,
            title,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
        )
    plt.savefig(FIGDIR / "example_03_simple_windfarm.png", dpi=300, bbox_inches="tight")
    plt.show()
