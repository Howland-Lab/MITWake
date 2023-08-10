import numpy as np

from ..BaseClasses import RotorBase
from MITBEM.BEM import BEM as mitbem
from MITBEM import ThrustInduction, TipLoss


class BEM(RotorBase):
    def __init__(
        self, rotordefinition, x=0.0, y=0.0, z=0.0, Cta_method=None, tiploss_method=None
    ):
        """_summary_

        Args:
            rotordefinition (_type_): _description_
            x (float): diameter-normalized longitudinal location of rotor center. Defaults to 0.0.
            y (float): diameter-normalized lateral location of rotor center. Defaults to 0.0.
            z (float): diameter-normalized vertical location of rotor center. Defaults to 0.0.
            Cta_method (_type_, optional): _description_. Defaults to None.
            tiploss_method (_type_, optional): _description_. Defaults to None.
        """
        self._x, self._y, self._z = x, y, z

        if Cta_method is None:
            Cta_method = "mike_corrected"

        if tiploss_method is None:
            tiploss_method = "tiproot"

        self.bem = mitbem(
            rotordefinition, Cta_method=Cta_method, tiploss=tiploss_method
        )

    def initialize(self, pitch, tsr, yaw, U, V):
        wsp = np.sqrt(U**2 + V**2)
        wdir = np.arctan2(V, U)
        
        return self.bem.solve(pitch, tsr, yaw, (wsp, wdir))

    def gridpoints(self, _pitch, _tsr, yaw):
        """
        Returns grid points in cartesian coordinates in meteorological frame of
        reference (origin at hub center) normalized by turbine diameter
        """
        # grid points normalized by turbine RADIUS
        X, Y, Z = self.bem.gridpoints_cart(yaw)

        return X / 2 + self.x, Y / 2 + self.y, Z / 2 + self.z

    def REWS(self):
        return self.bem.REWS()

    def Cp(self):
        return self.bem.Cp()

    def Ct(self):
        return self.bem.Ct()

    def Cq(self):
        return self.bem.Cq()

    def Ctprime(self):
        return self.bem.Ctprime()

    def a(self):
        return self.bem.a()

    def u4(self):
        return self.bem.u4()

    def v4(self):
        return self.bem.v4()
