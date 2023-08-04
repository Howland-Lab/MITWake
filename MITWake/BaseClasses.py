from abc import ABCMeta, abstractmethod


class RotorBase(metaclass=ABCMeta):
    @abstractmethod
    def initialize(self, U, V, *args, **kwargs):
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
    def wsp(self, x, y, z):
        ...

    @abstractmethod
    def initialize(self, *args, **kwargs):
        ...

    @abstractmethod
    def REWS(self):
        """
        Returns a list of REWS for each turbine in the wind farm.
        """
        ...

    @abstractmethod
    def Cp(self):
        """
        Returns a list of Cp for each turbine in the wind farm.
        """
        ...

    @abstractmethod
    def Ct(self):
        """
        Returns a list of Ct for each turbine in the wind farm.
        """
        ...

    @abstractmethod
    def Ctprime(self):
        """
        Returns a list of Ctprime for each turbine in the wind farm.
        """
        ...

    @abstractmethod
    def a(self):
        """
        Returns a list of a for each turbine in the wind farm.
        """
        ...

    @abstractmethod
    def u4(self):
        """
        Returns a list of u4 for each turbine in the wind farm.
        """
        ...

    @abstractmethod
    def v4(self):
        """
        Returns a list of v4 for each turbine in the wind farm.
        """
        ...


class SuperpositionBase(metaclass=ABCMeta):
    @abstractmethod
    def summation(self, deficits, baseclass):
        ...


class WindfieldBase(metaclass=ABCMeta):
    @abstractmethod
    def wsp(self, X, Y, Z):
        """
        Returns the normalized wind speed at coordinates (X, Y, Z) in
        meteorological coordinates normalized by rotor diameter.
        """
        ...

    @abstractmethod
    def wdir(self, X, Y, Z):
        """
        Returns the normalized lateral wind direction (positive is clockwise as
        viewed from top??) at coordinates (X, Y, Z) in meteorological
        coordinates normalized by rotor diameter.
        """
        ...
