from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm

from mit_yaw_induction_wake_model import Windfarm

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


### DEBUG
nit, nfev, njev = [], [], []


def objective_func(x, T2_x, T2_y):
    Ct1, yaw1 = x
    farm = Windfarm.GradWindfarm([0, T2_x], [0, T2_y], [Ct1, 2], [yaw1, 0])
    Cp, dCpdCt, dCpdyaw = farm.total_Cp()

    return -Cp, (-dCpdCt[0], -dCpdyaw[0])


def find_optimal_setpoints(T2_x, T2_y):
    res = optimize.minimize(
        objective_func,
        [1, 0.00],
        args=(T2_x, T2_y),
        # bounds=[
        #     (0.00001, 5),
        #     (-np.pi / 2 * 0.9, np.pi / 2 * 0.9),
        # ],
        jac=True,
    )

    Ct, yaw = res.x
    farm_efficiency = -res.fun

    nit.append(res.nit), nfev.append(res.nfev), njev.append(res.njev)

    return Ct, yaw, farm_efficiency


if __name__ == "__main__":
    Cts, yaws, T2_xs, T2_ys, farm_efficiencies = [], [], [], [], []
    for T2_x in tqdm(np.arange(1, 10, 1)):
        for T2_y in tqdm(np.linspace(-3, 3, 200)):
            Ct, yaw, farm_eff = find_optimal_setpoints(T2_x, np.round(T2_y, 3))
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
            FIGDIR / f"example_05_optimal_{to_plot_y}_vs_{to_plot_x}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
