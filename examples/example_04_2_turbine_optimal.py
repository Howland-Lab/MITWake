"""
A recreation of Figure 9a with control gradients from:

Heck, Kirby S., Hannah M. Johlas, and Michael F. Howland. "Modelling the
induction, thrust and power of a yaw-misaligned actuator disk." Journal of Fluid
Mechanics 959 (2023): A9.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from tqdm import tqdm

from MITWake import Windfarm

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


X = [0, 8]
Y = [0, 0.5]

Ct2, yaw2 = 2, 0

path_Ct, path_yaw = [2], [0]


def objective_func(x):
    Cts, yaws = [x[0], 2], [x[1], 0]

    farm = Windfarm.GradWindfarm(X, Y, Cts, yaws)
    Cp, dCpdCt, dCpdyaw = farm.total_Cp()
    return -Cp, -np.array([dCpdCt[0], dCpdyaw[0]])


def find_optimal_setpoints():
    def callback(xk):
        path_Ct.append(xk[0])
        path_yaw.append(np.rad2deg(xk[1]))

    res = optimize.minimize(
        objective_func,
        [2, np.deg2rad(0)],
        bounds=[(0.00001, 5), (-np.pi / 2 * 0.9, np.pi / 2 * 0.9)],
        jac=True,
        callback=callback,
    )
    Ct_opt, yaw_opt = res.x
    Cp_opt = -res.fun

    return Cp_opt, Ct_opt, np.rad2deg(yaw_opt)


def plot_Cpmap(
    Cts, yaws, field, yaw_opt, Ct_best, title, save, vmin=None, vmax=None, cmap=None
):
    plt.contourf(
        np.rad2deg(yaws), Cts, field, levels=10, cmap=cmap, vmin=vmin, vmax=vmax
    )
    plt.contour(np.rad2deg(yaws), Cts, field, levels=[0], colors="k")
    plt.xlabel("$\gamma_1$ [deg]")
    plt.ylabel("$C_{T,1}'$")
    plt.plot(yaw_opt, Ct_best, "*k", ms=10)

    plt.text(
        0.5,
        0.98,
        title,
        horizontalalignment="center",
        verticalalignment="top",
        transform=plt.gca().transAxes,
    )

    plt.plot(path_yaw, path_Ct, "--.k", alpha=0.4)

    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close()


def run():
    Cp_opt, Ct_opt, yaw_opt = find_optimal_setpoints()

    Ct1s = np.linspace(0.5, 4, 50)
    yaw1s = np.deg2rad(np.linspace(0, 50, 50))

    yaw_mesh, Ct_mesh = np.meshgrid(yaw1s, Ct1s)

    inputs = list(zip(yaw_mesh.ravel(), Ct_mesh.ravel()))
    Cptots, dCptotdCts, dCptotdyaws = [], [], []

    for yaw1, Ct1 in tqdm(inputs):
        windfarm = Windfarm.GradWindfarm(X, Y, [Ct1, Ct2], [yaw1, yaw2])
        windfarm.turbine_Cp()
        Cptot, dCptotdCt, dCptotdyaw = windfarm.total_Cp()

        Cptots.append(Cptot)
        dCptotdCts.append(dCptotdCt[0])
        dCptotdyaws.append(dCptotdyaw[0])

    Cptots = np.reshape(Cptots, Ct_mesh.shape)
    dCptotdCts = np.reshape(dCptotdCts, Ct_mesh.shape)
    dCptotdyaws = np.reshape(dCptotdyaws, Ct_mesh.shape)

    plot_Cpmap(
        Ct_mesh,
        yaw_mesh,
        Cptots,
        yaw_opt,
        Ct_opt,
        "$C_p$",
        FIGDIR / "example_04_2_turbine_optimal_Cp.png",
    )

    vmax = np.abs(dCptotdCts).max()
    vmin = -vmax
    plot_Cpmap(
        Ct1s,
        yaw1s,
        dCptotdCts,
        yaw_opt,
        Ct_opt,
        "$dC_p/dC_t$",
        FIGDIR / "example_04_2_turbine_optimal_dCpdCt.png",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )

    vmax = np.abs(dCptotdyaws).max()
    vmin = -vmax
    plot_Cpmap(
        Ct1s,
        yaw1s,
        dCptotdyaws,
        yaw_opt,
        Ct_opt,
        "$dC_p/d\gamma$",
        FIGDIR / "example_04_2_turbine_optimal_dCpdyaw.png",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )

    return yaw_opt, Ct_opt, Cp_opt


if __name__ == "__main__":
    yaw_opt, Ct_opt, Cp_opt = run()
    print("yaw_opt: ", yaw_opt)
    print("Ct_opt: ", Ct_opt)
    print("Cp_opt: ", Cp_opt)
