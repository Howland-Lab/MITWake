"""
A recreation of Figure 8b with control gradients from:

Heck, Kirby S., Hannah M. Johlas, and Michael F. Howland. "Modelling the
induction, thrust and power of a yaw-misaligned actuator disk." Journal of Fluid
Mechanics 959 (2023): A9.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from MITWake import Turbine

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)

TITLES = ["$du$", "$ddu/dC_t$", "$ddu/d\gamma$"]
cmaps = ["YlGnBu_r", "RdYlGn", "RdYlGn"]
vmins = [None, -0.3, -1]
vmaxs = [None, 0.3, 1]
yaw = np.deg2rad(24)
Ct = 2.11
if __name__ == "__main__":
    xs = np.linspace(-1, 8, 300)
    ys = np.linspace(-1.5, 1.5, 300)

    xmesh, ymesh = np.meshgrid(xs, ys)

    turbine = Turbine.GradientTurbine(Ct, yaw)

    deficit, ddeficitdCt, ddeficitdyaw = turbine.deficit(xmesh, ymesh)

    fig, axes = plt.subplots(3, 1, sharex=True)

    for ax, field, title, cmap, vmin, vmax in zip(
        axes, [deficit, ddeficitdCt, ddeficitdyaw], TITLES, cmaps, vmins, vmaxs
    ):
        ax.imshow(
            field,
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Draw turbine
        turb_x, turb_y, yaw = 0, 0, yaw
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
