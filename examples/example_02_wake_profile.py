from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mit_yaw_induction_wake_model import Turbine

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)

TITLES = ["$\delta U$", "$d\delta U/dC_t$", "$d\delta U/d\gamma$"]
yaw = np.deg2rad(24)
Ct = 2.11
if __name__ == "__main__":
    xs = np.linspace(-1, 8, 300)
    ys = np.linspace(-1.5, 1.5, 300)

    xmesh, ymesh = np.meshgrid(xs, ys)

    turbine = Turbine.GradientTurbine(Ct, yaw)

    deficit, ddeficitdCt, ddeficitdyaw = turbine.deficit(xmesh, ymesh)

    fig, axes = plt.subplots(3, 1, sharex=True)

    for ax, field, title in zip(axes, [deficit, ddeficitdCt, ddeficitdyaw], TITLES):
        ax.imshow(
            field,
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="lower",
            cmap="YlGnBu",
        )

        # Draw turbine
        turb_x, turb_y = 0, 0
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
    plt.savefig(FIGDIR / "example_02_wake_profile.png", dpi=300, bbox_inches="tight")
