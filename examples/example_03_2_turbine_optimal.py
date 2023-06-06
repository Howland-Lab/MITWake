from pathlib import Path

import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from tqdm import tqdm

from mit_yaw_induction_wake_model import ActuatorDisk

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


# Downstream turbine location
# T2_x, T2_y = 8, 0.0
R = 0.5  # rotor radius
Ad = np.pi * R**2


def Cp(Ctprime, yaw, REWS):
    a, _, _ = ActuatorDisk.fullcase(Ctprime, yaw)
    return Ctprime * ((1 - a) * np.cos(yaw) * REWS) ** 3


def two_turbine_Cp(Ct1, yaw, T2_y, T2_x):
    wake = ActuatorDisk.MITWake(Ct1, yaw)

    REWS = wake.REWS(T2_x, T2_y, R=R)

    Cp1 = Cp(Ct1, yaw, 1)
    Cp2 = Cp(2, 0, REWS)

    Cp_total = (Cp1 + Cp2) / 2

    return Cp_total


def optimize_and_plot(T2_x, T2_y):
    Cts = np.linspace(0.5, 4, 50)
    yaws = np.deg2rad(np.linspace(-50, 50, 50))

    yaw_mesh, Ct_mesh = np.meshgrid(yaws, Cts)

    inputs = list(zip(yaw_mesh.ravel(), Ct_mesh.ravel()))
    outputs = []

    for yaw, Ct in inputs:
        out = two_turbine_Cp(Ct, yaw, T2_y, T2_x)
        outputs.append(out)

    outputs = np.reshape(outputs, Ct_mesh.shape)
    idx_Ct, idx_yaw = np.unravel_index(np.argmax(outputs), outputs.shape)
    yaw_best = np.rad2deg(yaws[idx_yaw])
    Ct_best = Cts[idx_Ct]
    plt.imshow(
        outputs,
        extent=[np.rad2deg(yaws.min()), np.rad2deg(yaws.max()), Cts.min(), Cts.max()],
        origin="lower",
        aspect="auto",
    )

    plt.xlabel("$\gamma_1$ [deg]")
    plt.ylabel("$C_{T,1}'$")
    plt.title(f"x: {T2_x}    y:{T2_y}")
    plt.plot(yaw_best, Ct_best, "xk")
    plt.savefig(
        FIGDIR / f"2_turbine_optimal_x{T2_x}_y{T2_y}.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    return yaw_best, Ct_best, outputs.max()


if __name__ == "__main__":
    T2_x = 8
    for T2_y in tqdm(np.arange(0, 2, 0.1)):
        out = optimize_and_plot(T2_x, np.round(T2_y, 3))


