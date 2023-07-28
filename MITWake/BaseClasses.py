from abc import ABCMeta, abstractmethod


class RotorBase(metaclass=ABCMeta):
    @abstractmethod
    def initialize(self, windfield, *args, **kwargs):
        ...

    @abstractmethod
    def gridpoints(self, yaw):
        ...

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @abstractmethod
    def REWS(self):
        ...

    @abstractmethod
    def Cp(self):
        ...

    @abstractmethod
    def Ct(self):
        ...

    @abstractmethod
    def Ctprime(self):
        ...

    @abstractmethod
    def a(self):
        ...

    @abstractmethod
    def u4(self):
        ...

    @abstractmethod
    def v4(self):
        ...


class WakeBase(metaclass=ABCMeta):
    @abstractmethod
    def deficit(self, x, y, z):
        ...

    @abstractmethod
    def initialize(self, *args, **kwargs):
        ...


class WindfarmBase(metaclass=ABCMeta):
    @abstractmethod
    def deficits(self, indices):
        ...

    @abstractmethod
    def wsp(self, x, y, z):
        ...

    @abstractmethod
    def initialize(self, *args, **kwargs):
        ...


class SuperpositionBase(metaclass=ABCMeta):
    pass


class WindfieldBase(metaclass=ABCMeta):
    pass
