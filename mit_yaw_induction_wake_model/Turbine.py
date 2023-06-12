from mit_yaw_induction_wake_model import Rotor, Wake


class BasicTurbine:
    def __init__(self, Ct, yaw, x=0, y=0, sigma=0.25, kw=0.07):
        self.x, self.y = x, y
        self.Ct, self.yaw = Ct, yaw
        self.a, u4, v4 = Rotor.yawthrust(Ct, yaw)
        self.wake = Wake.Gaussian(u4, v4, sigma, kw)

    def deficit(self, x, y, z=0):
        return self.wake.deficit(x, y, z)


class GradientTurbine:
    def __init__(self, Ct, yaw, x=0, y=0, sigma=0.25, kw=0.07, induction_eps=0.000001):
        self.x, self.y = x, y
        self.Ct, self.yaw = Ct, yaw
        (
            self.a,
            u4,
            v4,
            self.dadCt,
            dudCt,
            dvdCt,
            self.dadyaw,
            dudyaw,
            dvdyaw,
        ) = Rotor.gradyawthrust(Ct, yaw, eps=induction_eps)
        self.wake = Wake.GradGaussian(u4, v4, dudCt, dudyaw, dvdCt, dvdyaw, sigma, kw)

    def deficit(self, x, y, z=0, FOR="met"):
        if FOR == "met":
            return self.wake.deficit(x - self.x, y - self.y, z)
        elif FOR == "local":
            return self.wake.deficit(x, y, z)
