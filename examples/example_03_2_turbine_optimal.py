from pathlib import Path

import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from tqdm import tqdm

from mit_yaw_induction_wake_model import ActuatorDisk

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)




# Downstream turbine location
T2_x, T2_y = 8, 0.5
R = 0.5  # rotor radius
Ad = np.pi * R**2


def Cp(Ctprime, yaw, REWS):
    a, _, _ = ActuatorDisk.fullcase(Ctprime, yaw)
    return Ctprime * ((1 - a) * np.cos(yaw) * REWS) ** 3


def two_turbine_Cp(Ct1, yaw):
    dys = np.linspace(-R, R, 100)
    ys = T2_y + dys

    wake = ActuatorDisk.MITWake(Ct1, yaw)

    deficit = wake.deficit(T2_x, ys)
    REWS = 1 / R**2 * trapz(np.abs(dys) * (1 - deficit), dys)

    Cp1 = Cp(Ct1, yaw, 1)
    Cp2 = Cp(2, 0, REWS)

    Cp_total = (Cp1 + Cp2) / 2

    return Cp_total


if __name__ == "__main__":
    Cts = np.linspace(0.5, 4, 50)
    yaws = np.deg2rad(np.linspace(0, 50, 50))

    yaw_mesh, Ct_mesh = np.meshgrid(yaws, Cts)

    inputs = list(zip(yaw_mesh.ravel(), Ct_mesh.ravel()))
    outputs = []

    for yaw, Ct in tqdm(inputs):
        out = two_turbine_Cp(Ct, yaw)
        outputs.append(out)

    idx_max = np.argmax(outputs)
    Ct_best = Ct_mesh.ravel()[idx_max]
    yaw_best = yaw_mesh.ravel()[idx_max]

    outputs = np.reshape(outputs, Ct_mesh.shape)
    plt.imshow(
        outputs,
        # extent=[yaw.min(), yaw.max(), Cts.min(), Cts.max()],
        origin="lower",
    )
    plt.plot(yaw_best, Ct_best, "xk")
    plt.show()
