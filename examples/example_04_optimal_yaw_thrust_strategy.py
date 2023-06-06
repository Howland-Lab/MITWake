from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm

from mit_yaw_induction_wake_model import ActuatorDisk

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)

R = 0.5


def Cp(Ctprime, yaw, REWS):
    a, _, _ = ActuatorDisk.fullcase(Ctprime, yaw)
    return Ctprime * ((1 - a) * np.cos(yaw) * REWS) ** 3


def two_turbine_Cp(x, T2_y, T2_x):
    Ct1, yaw = x
    wake = ActuatorDisk.MITWake(Ct1, yaw)

    REWS = wake.REWS(T2_x, T2_y, R=R)

    Cp1 = Cp(Ct1, yaw, 1)
    Cp2 = Cp(2, 0, REWS)

    Cp_total = (Cp1 + Cp2) / 2

    return -Cp_total


def find_optimal_setpoints(T2_x, T2_y):
    res = optimize.minimize(
        two_turbine_Cp,
        [1, 0],
        args=(T2_y, T2_x),
        bounds=[
            (0.00001, 5),
            (-np.pi, np.pi),
        ],
    )

    Ct, yaw = res.x
    return Ct, yaw


if __name__ == "__main__":
    Cts, yaws, T2_xs, T2_ys = [], [], [], []
    for T2_x in tqdm(np.arange(1, 10, 1)):
        for T2_y in tqdm(np.linspace(-3, 3, 200)):
            Ct, yaw = find_optimal_setpoints(T2_x, np.round(T2_y, 3))
            Cts.append(Ct)
            yaws.append(yaw)
            T2_xs.append(T2_x)
            T2_ys.append(T2_y)

    df = pd.DataFrame(
        np.array([Cts, yaws, T2_xs, T2_ys]).T, columns=["Ct", "yaw", "x", "y"]
    )

    plt.figure()
    xs = df.x.unique()
    for i, x in enumerate(xs):
        _df = df[df.x == x]
        plt.plot(
            np.rad2deg(_df.yaw), _df.Ct, c=plt.cm.viridis(i / len(xs)), label=f"x={x}"
        )

    plt.legend()

    plt.xlabel("Optimal yaw")
    plt.ylabel("Optimal Ct")
    plt.savefig(FIGDIR / "optimal_yaw_vs_ct.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    xs = df.x.unique()
    for i, x in enumerate(xs):
        _df = df[df.x == x]
        plt.plot(
            _df.y, np.rad2deg(_df.yaw), c=plt.cm.viridis(i / len(xs)), label=f"x={x}"
        )

    plt.legend()

    plt.xlabel("y")
    plt.ylabel("Optimal yaw")
    plt.savefig(FIGDIR / "optimal_yaw.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    xs = df.x.unique()
    for i, x in enumerate(xs):
        _df = df[df.x == x]
        plt.plot(_df.y, _df.Ct, c=plt.cm.viridis(i / len(xs)), label=f"x={x}")

    plt.legend()

    plt.xlabel("y")
    plt.ylabel("Optimal Ct")
    plt.savefig(FIGDIR / "optimal_Ct.png", dpi=300, bbox_inches="tight")
    plt.close()
