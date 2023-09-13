"""
Curled Wake model testing for one turbine in isolation. 

#TODO: add more turbines

Kirby Heck
2023 Aug. 23
"""

from typing import Optional 
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import interpn

class CurledWake(): 
    """
    Martinez-Tossas, et al. (2021) implementation of the curled wake model. 
    """

    def __init__(
            self, 
            ct: float,
            yaw: float = 0.0,
            d: float = 1,
            z_hub: float = 1, 
            N: int = 20, 
            sigma: float = 0.2, 
            C: float = 4., 
            lam: float = 0.214, 
            kappa: float = 0.4, 
            integrator: str = 'EF', 
            bkgd: Optional[object] = None, 
            avg_xy: bool = True, 
            dx: float = 0.05, 
            dy: float = 0.1, 
            dz: float = 0.1, 
    ): 
        """
        Args: 
            ct (float): Rotor thrust, non-dimensionalized to
                pi/8 d**2 rho u_h^2 cos(yaw)^2.
            yaw (float): Rotor yaw angle (radians).
            d (float): non-dimensionalizing value for diameter. Defaults to 1.
            z_hub (float): non-dimensional hub height z_hub/d. Defaults to 1. 
            N (int): number of points to discretize Lamb-Oseem vortices. Defaults to 20. 
            sigma (float): width of Lamb-Oseem vortices. Defaults to 0.2. 
            C (float): mixing length constant. Defaults to 4.
            lam (float): free atmosphere mixing length. Defaults to 0.214 (27 m for the NREL 5 MW). 
            kappa (float): von Karman constant. Defaults to 0.4. 
            integrator (str): forward integration method. Defaults to 'EF'.
            bkgd (PadeOpsIO.BudgetIO): object with background flow properties. Defaults to None. 
            avg_xy (bool): average background flow in x, y. Defaults to True. 
        """

        self.ct = ct
        self.yaw = yaw
        self.d = d  # TODO - be more consistent in the dimensional problem
        self.z_hub = z_hub
        self.N = N
        self.sigma = sigma
        self.C = C
        self.lam = lam
        self.kappa = kappa
        self.integrator = get_integrator(integrator)
        
        self.bkgd = bkgd  # TODO - fix this down the line
        self.avg_xy = avg_xy
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.has_result = False  # true when solved for the wake solution
        self.extent = None

        # initialize background flow and grid
        self.init_grid()
        self.init_background()
        self.init_turbulence()

    def deficit(
            self, 
            x: np.array, 
            y: np.array, 
            z: Optional[np.array] = 0, 
            nu_eff: bool = None, 
            field: str = 'u', 
            non_dim: bool = True, 
    ): 
        """
        Compute wake deficit at selected points. 
        """
        if not self.has_result or not self.within_bounds(x, y, z): 
            self.compute_uvw(x, y, z, nu_eff=nu_eff)
        
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        points = (self.xg, self.yg, self.zg)
        if x.ndim > 1 or y.ndim > 1 or z.ndim > 1:  
            # if given ndarray, use those for the grid (assume tuples)
            xG, yG, zG = x, y, z
            if len(xG) == 1:  # TODO: Fix this
                xG = np.ones_like(yG) * xG
            if len(zG) == 1: 
                zG = np.ones_like(yG) * zG
        else: 
            xG, yG, zG = np.meshgrid(x, y, z, indexing='ij')
        shape = xG.shape
        xl = np.ravel(xG)
        yl = np.ravel(yG)
        zl = np.ravel(zG)

        query = np.stack([xl, yl, zl], axis=-1)

        if field == 'u': 
            u = interpn(points, self.u, query)
            u = np.squeeze(np.reshape(u, shape))
            if non_dim: 
                u /= self.get_ud(weighting='hub')  # non-dimensionalize
        else: 
            raise NotImplementedError('deficit(): only Delta-u is implemented.')
        return -u  # flip so to match with other deficit sign conventions  TODO
        
    def compute_uvw(
            self, 
            x: np.array, 
            y: np.array, 
            z: np.array, 
            return_uvw: bool = False, 
            nu_eff=None, 
    )-> None: 
        """
        Computes the wake deficit field via forward marching. 
        """        
        self.init_grid(x, y, z)  # creates a (possibly new) grid
        self._interp_Ui()
        self.init_turbulence(nu_eff=nu_eff)
        # TODO - reinitialize eddy viscosity, if needed ??

        # initialize vortices
        self._compute_vw()
        # compute deficit field
        self._compute_u() 
        self.has_result = True

        if return_uvw: 
            return (self.u, self.v, self.w)

    def init_grid(self, x=0, y=0, z=0, yz_buff: float = 1.,): 
        """
        Creates a grid centered around the wind turbine. 
        """
        def _make_axis(xmin, xmax, dx): 
            """Helper function to create an axis from limits that always includes 0"""
            n_st = np.ceil(np.abs(xmin / dx)) * np.sign(xmin)
            n_end = np.ceil(np.abs(xmax / dx)) * np.sign(xmax)
            return np.arange(n_st, n_end + dx) * dx  # ensure that 0 is included in the grid
        
        # set limits with buffers: 
        xlim = [np.min([-self.dx, np.min(x)]), np.max(x)]
        ylim = [np.min([-yz_buff, np.min(y)]), np.max([yz_buff, np.max(y)])]
        zlim = [np.min([-yz_buff, np.min(z)]), np.max([yz_buff, np.max(z)])]

        self.xg = _make_axis(*xlim, self.dx)
        self.yg = _make_axis(*ylim, self.dy)
        self.zg = _make_axis(*zlim, self.dz)

        self.shape = (len(self.xg), len(self.yg), len(self.zg))

        self.extent = [min(self.xg), max(self.xg), 
                       min(self.yg), max(self.yg), 
                       min(self.zg), max(self.zg)]  # these bounds may differ from inputs
        
    def within_bounds(self, x, y, z): 
        """Check if new axes are within computed bounds"""
        for xi, Xi in zip([x, y, z], [self.xg, self.yg, self.zg]): 
            if np.min(xi) < np.min(Xi) or np.max(xi) > np.max(Xi): 
                return False
            
        return True

    def init_background(self): 
        """
        Initialize background flow fields. 
        """
        if self.bkgd is not None: 
            # LES background flow
            self.bkgd.read_budgets(budget_terms=['ubar', 'vbar', 'wbar']) 
            if self.avg_xy: 
                Ui = self.bkgd.budget
                self._U = np.mean(Ui['ubar'], (0, 1))
                self._V = np.mean(Ui['vbar'], (0, 1))
                try: 
                    self._W = np.mean(Ui['wbar'], (0, 1))
                except KeyError as e: 
                    self._W = np.zeros_like(self._V)  # should have zero subsidence
                self._z = self.bkgd.zLine
                self._interp_Ui()

            else: 
                raise NotImplementedError('init_background(): Currently `avg_xy` must be True.')
        else: 
            self.U = np.ones_like(self.zg)  # uniform inflow
            self.V = 0 
            self.W = 0

    def _interp_Ui(self)-> None: 
        """Interpolate velocity profiles to local grid"""
        if self.bkgd is not None: 
            self.U = np.interp(self.zg, self._z, self._U)
            self.V = np.interp(self.zg, self._z, self._V)
            self.W = np.interp(self.zg, self._z, self._W)
        else: 
            self.U = np.ones_like(self.zg)

    def init_turbulence(self, nu_eff=None)-> None: 
        """
        Initializes eddy viscosity. 
        """
        if nu_eff is not None: 
            self.nu_eff = nu_eff  # override turblence model
            return
        
        if self.bkgd is not None: 
            self.lam = 2.7e-4 * self.bkgd.Ro_f * self.d
        
        # compute mixing length and eddy viscosity
        z = self.zg + self.z_hub  # in unphysical cases, could be below zero? 
        self.lm = self.kappa * z / (1. + self.kappa * z / self.lam)
        dUdz = np.gradient(self.U, self.dz, axis=-1)
        self.nu_eff = self.C * self.lm**2 * abs(dUdz)

    def get_ic(
            self, 
            y: np.array, 
            z: np.array, 
            ud: float, 
            smooth_fact: float = 1.5, 
    )-> np.array: 
        """
        Initial condition for the wake model

        Args: 
            self (CurledWake)
            y (np.array): lateral axis
            z (np.array): vertical axis
            ud (float): disk velocity
            smooth_fact (float): Gaussian convolution standard deviation, 
                equal to smooth_fact * self.dy. Defaults to 1.5. 
        """
        yG, zG = np.meshgrid(y, z, indexing='ij')
        kernel_y = np.arange(-10, 11)[:, None] * self.dy
        kernel_z = np.arange(-10, 11)[None, :] * self.dz

        turb = (yG**2 + zG**2) < (self.d / 2)**2
        # gauss = np.exp(-(yG**2 + zG**2) / (np.sqrt(self.dy * self.dz) * smooth_fact)**2 / 2)
        gauss = np.exp(-(kernel_y**2 + kernel_z**2) / (np.sqrt(self.dy * self.dz) * smooth_fact)**2 / 2)
        gauss /= np.sum(gauss)
        a = 0.5 * (1 - np.sqrt(1 - self.ct * np.cos(self.yaw)**2))
        delta_u = -2 * a * ud

        return convolve2d(turb, gauss, 'same') * delta_u
    
    def get_ud(self, weighting='disk')-> float: 
        """
        Gets background disk velocity by numerically integrating self.U
        """
        if self.U.ndim == 1: 
            r = self.d/2
            zids = abs(self.zg) < r  # within the rotor area
            if weighting == 'disk': 
                # weight based on the area of the "disk"
                A = np.trapz(np.sqrt(r**2 - self.zg[zids]**2)) 
                return np.trapz(self.U[zids] * np.sqrt(r**2 - self.zg[zids]**2)) / A
            
            elif weighting == 'equal': 
                return np.mean(self.U[zids])
            elif weighting == 'hub': 
                return np.interp(0, self.zg, self.U)  # hub height velocity
            else: 
                raise NotImplementedError('get_ud(): `weighting` must be "disk", "equal", or "hub". ')
        else: 
            # TODO - FIX
            raise NotImplementedError('Deal with this later')

    def _compute_u(self, ud=None)-> None: 
        """
        Forward marches Delta_u field. 
        """
        def _dudt(x, _u): 
            """du/dt function"""
            xid = np.argmin(np.abs(x - self.xg))
            # Full velocity fields for advection: 
            u = _u + self.U  
            v = self.V
            w = self.W
            if self.yaw != 0: 
                v = v + self.v[xid, ...]
                w = w + self.w[xid, ...]  # TODO: HOTFIX
            # gradient fields: 
            dudy = np.gradient(_u, self.dy, axis=0)
            dudz = np.gradient(_u, self.dz, axis=1)
            d2udy2 = np.gradient(dudy, self.dy, axis=0)
            d2udz2 = np.gradient(dudz, self.dz, axis=1)
            return (-v * dudy - w * dudz + self.nu_eff * (d2udy2 + d2udz2)) / u
        
        # now integrate! 
        if ud is None:
            ud = self.get_ud()
        ic = self.get_ic(self.yg, self.zg, ud)
        xmin = 0
        xmax = max(self.xg)
        x, delta_u = integrate(ic, _dudt, dt=self.dx, T=[xmin, xmax], f=self.integrator)

        self.u = np.zeros(self.shape)
        xid_st = np.argmin(abs(self.xg-xmin))
        self.u[xid_st:, ...] = delta_u

    def _compute_vw(self)-> None: 
        """
        Use Lamb-Oseen vortices to compute curling
        """
        if self.yaw == 0: 
            self.v = np.zeros(self.shape)
            self.w = np.zeros(self.shape)
            return
        
        u_inf = self.get_ud('hub')
        r_i = np.linspace(-(self.d - self.dz) / 2, (self.d - self.dz) / 2, self.N) 

        Gamma_0 = 0.5 * self.d * u_inf * self.ct * np.sin(self.yaw) * np.cos(self.yaw)**2
        Gamma_i = Gamma_0 * 4 * r_i / (self.N * self.d**2 * np.sqrt(1 - (2 * r_i / self.d)**2))

        # generally, vortices can decay. So sigma should be a vector
        sigma = self.sigma * self.d * np.ones_like(self.xg) 
        
        # now we build the main summation, which is 4D (x, y, z, i)
        yG, zG = np.meshgrid(self.yg, self.zg, indexing='ij')
        yG = yG[None, ..., None]
        zG = zG[None, ..., None]
        r4D = yG**2 + (zG - r_i[None, None, None, :])**2  # 4D grid variable

        # put pieces together: 
        exponent = 1 - np.exp(-r4D / sigma[..., None, None, None]**2)
        summation = exponent / (2 * np.pi * r4D) * Gamma_i[None, None, None, :]

        v = np.sum(summation * (zG - r_i[None, None, None, :]), axis=-1)  # sum all vortices
        w = np.sum(summation * -yG, axis=-1)
        self.v = v * (self.xg >= 0)[:, None, None]
        self.w = w * (self.xg >= 0)[:, None, None]


def get_integrator(integrator): 
    """Return integrator function"""
    if integrator == 'EF': 
        return EF_step
    elif integrator == 'RK4': 
        return rk4_step
    else: 
        raise ValueError(f'"{integrator}" not a valid integration function')


def rk4_step(t_n, u_n, dudt, dt): 
    """
    Computes the next timestep of u_n given the finite difference function du/dt
    with a 4-stage, 4th order accurate Runge-Kutta method. 
    
    Parameters
    ----------
    t_n : float
        time for time step n
    u_n : array-like
        condition at time step n
    dudt : function 
        function du/dt(t, u)
    dt : float
        time step
    
    Returns u_(n+1)
    """    
    k1 = dt * dudt(t_n, u_n)
    k2 = dt * dudt(t_n + dt/2, u_n + k1/2)
    k3 = dt * dudt(t_n + dt/2, u_n + k2/2)
    k4 = dt * dudt(t_n + dt, u_n + k3)

    u_n1 = u_n + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u_n1


def EF_step(t_n, u_n, dudt, dt): 
    """
    Forward Euler stepping scheme
    """
    u_n1 = u_n + dt * dudt(t_n, u_n)
    return u_n1


def integrate(u0, dudt, dt=0.1, T=[0, 1], f=rk4_step): 
    """
    General integration function which calls a step function multiple times depending 
    on the parabolic integration strategy. 

    Parameters
    ----------
    u0 : array-like
        Initial condition of values
    dudt : function 
        Evolution function du/dt(t, u, ...)
    dt : float
        Time step
    T : (2, ) 
        Time range
    f : function
        Integration stepper function (e.g. RK4, EF, etc.)

    Returns
    -------
    t : (Nt, ) vector
        Time vector 
    u(t) : (Nt, ...) array-like
        Solution to the parabolic ODE. 
    """
    t = []
    ut = []

    u_n = u0  # initial condition
    t_n = T[0]
    
    while True: 
        ut.append(u_n)
        t.append(t_n)

        # update timestep
        t_n1 = t_n + dt
        if t_n1 > T[1] + dt/2:  # add some buffer here
            break
        u_n1 = f(t_n, u_n, dudt, dt)

        # update: 
        u_n = u_n1
        t_n = t_n1

    return np.array(t), np.array(ut)