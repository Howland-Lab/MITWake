from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mit_yaw_induction_wake_model import Windfarm

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


X = [0, 8]
Y = [0, 0.5]

Ct2, yaw2 = 2, 0


def plot_Cpmap(
    Cts, yaws, field, yaw_best, Ct_best, title, save, vmin=None, vmax=None, cmap=None
):
    plt.imshow(
        field,
        extent=[
            np.rad2deg(yaws.min()),
            np.rad2deg(yaws.max()),
            Cts.min(),
            Cts.max(),
        ],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        origin="lower",
        aspect="auto",
    )

    plt.xlabel("$\gamma_1$ [deg]")
    plt.ylabel("$C_{T,1}'$")
    plt.plot(yaw_best, Ct_best, "xk")

    plt.text(
        0.5,
        0.98,
        title,
        horizontalalignment="center",
        verticalalignment="top",
        transform=plt.gca().transAxes,
    )

    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close()


def run():
    Ct1s = np.linspace(0.5, 4, 50)
    yaw1s = np.deg2rad(np.linspace(-50, 50, 50))

    yaw_mesh, Ct_mesh = np.meshgrid(yaw1s, Ct1s)

    inputs = list(zip(yaw_mesh.ravel(), Ct_mesh.ravel()))
    Cps, dCpdCts, dCpdyaws = [], [], []

    for yaw1, Ct1 in tqdm(inputs):
        windfarm = Windfarm.GradWindfarm(X, Y, [Ct1, Ct2], [yaw1, yaw2])
        Cp, dCpdCt, dCpdyaw = windfarm.total_Cp()
        # windfarm.REWS_at_rotor(0)

        Cps.append(Cp)
        dCpdCts.append(dCpdCt[0])
        dCpdyaws.append(dCpdyaw[0])

    Cps = np.reshape(Cps, Ct_mesh.shape)
    dCpdCts = np.reshape(dCpdCts, Ct_mesh.shape)
    dCpdyaws = np.reshape(dCpdyaws, Ct_mesh.shape)

    idx_Ct, idx_yaw = np.unravel_index(np.argmax(Cps), Cps.shape)
    yaw_best = np.rad2deg(yaw1s[idx_yaw])
    Ct_best = Ct1s[idx_Ct]

    plot_Cpmap(
        Ct1s,
        yaw1s,
        Cps,
        yaw_best,
        Ct_best,
        "$C_p$",
        FIGDIR / "example_04_2_turbine_optimal_Cp.png",
    )

    vmax = np.abs(dCpdCts).max()
    vmin = -vmax
    plot_Cpmap(
        Ct1s,
        yaw1s,
        dCpdCts,
        yaw_best,
        Ct_best,
        "$dC_p/dC_t$",
        FIGDIR / "example_04_2_turbine_optimal_dCpdCt.png",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )

    vmax = np.abs(dCpdyaws).max()
    vmin = -vmax
    plot_Cpmap(
        Ct1s,
        yaw1s,
        dCpdyaws,
        yaw_best,
        Ct_best,
        "$dC_p/d\gamma$",
        FIGDIR / "example_04_2_turbine_optimal_dCpdyaw.png",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )

    return yaw_best, Ct_best, Cps.max()


if __name__ == "__main__":
    out = run()
