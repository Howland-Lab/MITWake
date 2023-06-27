from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from MITWake import Rotor

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    yaws = np.deg2rad(np.linspace(0, 30, 50))
    Ctprimes = [0.2, 0.4, 0.8, 1.2, 1.6, 2.0]

    fig = plt.figure()
    plt.xlabel("yaw angle [deg]")
    plt.ylabel("P(γ)/P(0) [-]")

    for i, Ct in enumerate(Ctprimes):
        color = plt.cm.viridis(i / len(Ctprimes))
        a, u, v = Rotor.yawthrust(Ct, yaws)
        Pratio = ((1 + 0.25 * Ct) * (1 - a) * np.cos(yaws)) ** 3
        plt.plot(np.rad2deg(yaws), Pratio, c=color, label=f"{Ct}")

    plt.legend()
    plt.grid()
    plt.xlim(0, 30)
    plt.ylim(0.6, 1.05)
    plt.savefig(
        FIGDIR / "example_01_power_yaw_relationship.png", dpi=300, bbox_inches="tight"
    )
