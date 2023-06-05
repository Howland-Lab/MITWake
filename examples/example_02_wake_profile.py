from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mit_yaw_induction_wake_model import ActuatorDisk

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


yaw = np.deg2rad(24)
CT = 2.11
if __name__ == "__main__":
    xs = np.linspace(-1, 8, 300)
    ys = np.linspace(-1.5, 1.5, 300)

    xmesh, ymesh = np.meshgrid(xs, ys)

    wake = ActuatorDisk.MITWake(CT, yaw)

    deficit = wake.deficit(xmesh, ymesh)

    plt.figure()
    plt.imshow(
        deficit,
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

    plt.plot(p[0, :], p[1, :], "k", lw=5)
    plt.savefig(FIGDIR / "wake_profile.png", dpi=300, bbox_inches="tight")
