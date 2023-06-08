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

### DEBUG
nit, nfev, njev = [], [], []


def Cp(Ctprime, yaw, REWS):
    a, _, _ = ActuatorDisk.calculate_induction(Ctprime, yaw)
    return Ctprime * ((1 - a) * np.cos(yaw) * REWS) ** 3


def two_turbine_Cp(x, T2_y, T2_x, anal=False):
    Ct1, yaw = x
    wake = ActuatorDisk.MITWake(Ct1, yaw)

    if anal:
        REWS = wake.REWS_anal(T2_x, T2_y)
    else:
        REWS = wake.REWS(T2_x, T2_y, R=R)

    Cp1 = Cp(Ct1, yaw, 1)
    Cp2 = Cp(2, 0, REWS)

    Cp_total = (Cp1 + Cp2) / 2

    return -Cp_total


def find_optimal_setpoints(T2_x, T2_y, anal):
    res = optimize.minimize(
        two_turbine_Cp,
        [1, 0],
        args=(T2_y, T2_x, anal),
        bounds=[
            (0.00001, 5),
            (-np.pi, np.pi),
        ],
    )

    Ct, yaw = res.x
    farm_efficiency = -res.fun

    nit.append(res.nit), nfev.append(res.nfev), njev.append(res.njev)

    return Ct, yaw, farm_efficiency


if __name__ == "__main__":
    Cts, yaws, T2_xs, T2_ys, farm_efficiencies = [], [], [], [], []
    for T2_x in tqdm(np.arange(1, 10, 1)):
        for T2_y in tqdm(np.linspace(-3, 3, 200)):
            Ct, yaw, farm_eff = find_optimal_setpoints(
                T2_x, np.round(T2_y, 3), anal=False
            )
            Cts.append(Ct)
            yaws.append(yaw)
            T2_xs.append(T2_x)
            T2_ys.append(T2_y)
            farm_efficiencies.append(farm_eff)

    print("nit", np.mean(nit))
    print("nfev", np.mean(nfev))
    print("njev", np.mean(njev))

    df = pd.DataFrame(
        np.array([Cts, yaws, T2_xs, T2_ys, farm_efficiencies]).T,
        columns=["Ct", "yaw", "x", "y", "farm_eff"],
    )

    df["yaw"] = np.rad2deg(df["yaw"])
    to_plot_list = [
        ("y", "Ct"),
        ("y", "yaw"),
        ("y", "farm_eff"),
        ("yaw", "Ct"),
    ]

    for to_plot_x, to_plot_y in to_plot_list:
        plt.figure()
        xs = df.x.unique()
        for i, x in enumerate(xs):
            _df = df[df.x == x]
            plt.plot(
                _df[to_plot_x],
                _df[to_plot_y],
                c=plt.cm.viridis(i / len(xs)),
                label=f"x={x}",
            )

        plt.legend(loc="lower right")

        plt.xlabel(to_plot_x)
        plt.ylabel(to_plot_y)
        plt.savefig(
            FIGDIR / f"optimal_{to_plot_y}_vs_{to_plot_x}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
