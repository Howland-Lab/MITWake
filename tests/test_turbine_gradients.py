import numpy as np
from mit_yaw_induction_wake_model import Turbine
import pytest

Ct = 2
yaw = np.deg2rad(24)
dx = 1e-6


@pytest.fixture
def turb():
    return Turbine.GradientTurbine(Ct, yaw)


@pytest.fixture
def turb_ct_p():
    return Turbine.GradientTurbine(Ct + dx, yaw)


@pytest.fixture
def turb_ct_m():
    return Turbine.GradientTurbine(Ct - dx, yaw)


@pytest.fixture
def turb_yaw_p():
    return Turbine.GradientTurbine(Ct, yaw + dx)


@pytest.fixture
def turb_yaw_m():
    return Turbine.GradientTurbine(Ct, yaw - dx)


def test_induction_gradients(turb, turb_ct_p, turb_ct_m, turb_yaw_p, turb_yaw_m):
    dadCt = turb.dadCt
    dadyaw = turb.dadyaw
    dudCt = turb.wake.dudCt
    dudyaw = turb.wake.dudyaw
    dvdCt = turb.wake.dvdCt
    dvdyaw = turb.wake.dvdyaw

    dadCt_FD = (turb_ct_p.a - turb_ct_m.a) / (2 * dx)
    dadyaw_FD = (turb_yaw_p.a - turb_yaw_m.a) / (2 * dx)
    dudCt_FD = (turb_ct_p.wake.u4 - turb_ct_m.wake.u4) / (2 * dx)
    dudyaw_FD = (turb_yaw_p.wake.u4 - turb_yaw_m.wake.u4) / (2 * dx)
    dvdCt_FD = (turb_ct_p.wake.v4 - turb_ct_m.wake.v4) / (2 * dx)
    dvdyaw_FD = (turb_yaw_p.wake.v4 - turb_yaw_m.wake.v4) / (2 * dx)

    assert np.abs(dadCt - dadCt_FD) < 0.00001
    assert np.abs(dadyaw - dadyaw_FD) < 0.00001
    assert np.abs(dudCt - dudCt_FD) < 0.00001
    assert np.abs(dudyaw - dudyaw_FD) < 0.00001
    assert np.abs(dvdCt - dvdCt_FD) < 0.00001
    assert np.abs(dvdyaw - dvdyaw_FD) < 0.00001


def test_centerline(turb, turb_ct_p, turb_ct_m, turb_yaw_p, turb_yaw_m):
    x = 2
    yc, dycdCt, dycdyaw = turb.wake.centerline(x)

    dycdCt_FD = (turb_ct_p.wake.centerline(x)[0] - turb_ct_m.wake.centerline(x)[0]) / (
        2 * dx
    )
    dycdyaw_FD = (
        turb_yaw_p.wake.centerline(x)[0] - turb_yaw_m.wake.centerline(x)[0]
    ) / (2 * dx)

    assert np.abs(dycdCt - dycdCt_FD) < 0.00001
    assert np.abs(dycdyaw - dycdyaw_FD) < 0.00001


def test_du(turb, turb_ct_p, turb_ct_m, turb_yaw_p, turb_yaw_m):
    x = 2
    du, ddudCt, ddudyaw = turb.wake.du(x)

    ddudCt_FD = (turb_ct_p.wake.du(x)[0] - turb_ct_m.wake.du(x)[0]) / (2 * dx)
    ddudyaw_FD = (turb_yaw_p.wake.du(x)[0] - turb_yaw_m.wake.du(x)[0]) / (2 * dx)

    assert np.abs(ddudCt - ddudCt_FD) < 0.00001
    assert np.abs(ddudyaw - ddudyaw_FD) < 0.00001


def test_deficit_gradients(turb, turb_ct_p, turb_ct_m, turb_yaw_p, turb_yaw_m):
    x, y, z = np.array([1, 2, 3]), np.array([0, 0, 0]), np.array([0, 0, 0])
    deficit, ddeficitdCt, ddeficitdyaw = turb.deficit(x, y, z)

    ddeficitdCt_FD = (turb_ct_p.deficit(x, y, z)[0] - turb_ct_m.deficit(x, y, z)[0]) / (
        2 * dx
    )
    ddeficitdyaw_FD = (
        turb_yaw_p.deficit(x, y, z)[0] - turb_yaw_m.deficit(x, y, z)[0]
    ) / (2 * dx)

    assert all(x < 0.00001 for x in np.abs(ddeficitdCt - ddeficitdCt_FD))
    assert all(x < 0.00001 for x in np.abs(ddeficitdyaw - ddeficitdyaw_FD))
