"""
Calibration scripts for wake models
"""

import numpy as np
from MITWake import Wake, CurledWake, Turbine
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calibrate(wake_type, uwake_ref, x=0, y=0, z=0, 
              ctp=2., yaw=0., wake_kwargs=None, 
              order=2, mask_thresh=0., 
              debug=False): 
    """
    Calibration script for different wake models. 

    Parameters
    ----------
    wake_type : str
        Wake type; see Turbine.py. Currently implemented: 
        'Gauss', 'BP16', 'CWM'
    uwake_ref : ndarray
        Array of wake velocities, normalized to u_hub
    x : ndarray
        Array of x-values corresponding to uwake_ref. 
    y : ndarray
        Array of y-values corresponding to uwake_ref. 
    z : ndarray, optional
        Array of z-values. Defaults to 0 (hub plane). 
    ctp : float, optional
        Value for C_T' of the calibration turbine. Defaults to 2.0. 
    yaw : float, optional
        Yaw value (radians) for the calibration turbine. Defaults to 0. 
    wake_kwargs : dict
        Keyword arguments for the wake model, except for the calibration parameter(s). 
    order : int
        Order of taking norms for the optimization function 
    """

    if wake_type == 'Gauss': 
        calib_kwargs = {'kw': 0.07, 'sigma': 0.25}  # default values
        constraints = ([1e-4, 0.5], [1e-3, 0.5])
        if 'sigma' in wake_kwargs.keys():  # hotfix workaround: don't fit sigma
            del calib_kwargs['sigma']
            constraints = (constraints[0], )
    elif wake_type == 'BP16': 
        calib_kwargs = {'ky': 0.03}  # default: symmetric in y, z
        constraints = ([1e-4, 0.5], )
    elif wake_type == 'BP16_yz': 
        wake_type = 'BP16'
        calib_kwargs = {'ky': 0.03, 'kz': 0.03}  # asymmetric gaussian
        constraints = ([1e-4, 0.5], [1e-4, 0.5], )
    elif wake_type == 'CWM': 
        calib_kwargs = {'C': 2.}
        constraints = ([0.1, 8], )
    else: 
        raise ValueError("`wake_type` must be 'Gauss', 'BP16', 'BP16_yz', or 'CWM'. ")

    if wake_kwargs is None: 
        wake_kwargs = {}  # setting a keyword argument to a mutable has issues

    xG, yG, zG = np.meshgrid(x, y, z, indexing='ij')

    counter = [0]

    mask = (uwake_ref > mask_thresh)
    def _minimize(calib_params): 
        """Minimization function"""
        _kwargs = {key: value for key, value in zip(calib_kwargs.keys(), calib_params)}
        wake = Turbine.BasicTurbine(ctp, yaw, 
                                    wake_type=wake_type,
                                    wake_kwargs={**wake_kwargs, **_kwargs}
                                    )
        delta_u = wake.deficit(xG, yG, zG)

        err = np.linalg.norm(((delta_u - uwake_ref) * mask).ravel(), ord=order)

        if debug: 
            counter[0] += 1
            print(f'iteration {counter[0]} error: {err:.4f}')
            print('\tMax delta_U: ', delta_u.max())
            print(f'\tParameter values: {_kwargs}')

        return err
    
    x0 = list(calib_kwargs.values())  # initial guess
    ret = minimize(_minimize, x0, bounds=constraints)
    
    if debug:  # plot stuff
        _kwargs = {key: value for key, value in zip(calib_kwargs.keys(), ret.x)}
        wake = Turbine.BasicTurbine(ctp, yaw, 
                                    wake_type=wake_type,
                                    wake_kwargs={**wake_kwargs, **_kwargs}
                                    )
        xG, yG = np.meshgrid(x, y, indexing='ij')
        delta_u = wake.deficit(xG, yG, 0)  # compute reference wake
        zline = np.atleast_1d(z)
        if len(zline) == 1: 
            ref_plot = uwake_ref
        else: 
            zid = np.argmin(abs(zline))
            ref_plot = uwake_ref[..., zid]

        ext = [x.min(), x.max(), y.min(), y.max()]
        AR = (y.max() - y.min()) / (x.max() - x.min())
        fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(6 * AR, 2))
        ax = axs[0]
        # mask = 1# sl_plot['ubar_deficit'] > 0.02
        im = ax.imshow(ref_plot.T, origin='lower', extent=ext)
        ax.set_title('LES')
        # plt.colorbar(im, ax=ax)

        ax = axs[1]
        ax.imshow(delta_u.T, origin='lower', extent=ext, clim=im.get_clim())
        ax.set_title('Model')
        plt.colorbar(im, ax=axs[0:2])

        ax = axs[2]
        diff = ref_plot-delta_u
        cmax = np.max(abs(diff))
        im2 = ax.imshow(diff.T, origin='lower', extent=ext, cmap='RdBu_r', clim=[-cmax, cmax])
        # ax.set_title('LES - model')
        ax.set_title(_kwargs)
        plt.colorbar(im2, ax=ax)
        # plt.tight_layout()
        plt.show()

    params = {key: value for key, value in zip(calib_kwargs.keys(), ret.x)}
    return params  # return optimized parameters in a dictionary

    
if __name__ == '__main__': 
    import padeopsIO as pio
    import os
    
    # Run debug calibration case
    dir_name = r'G:\Shared drives\howland_lab\current_projects\2022_harrington_urop\LES_data\two_turbine_sweep\export3'
    case = pio.BudgetIO(dir_name, filename='calibration', mat=True)
    pre = pio.BudgetIO(dir_name, filename='precursor', mat=True)

    sl = case.slice(budget_terms=['ubar'], xlim=[4, 8]) 
    slp = pre.slice(budget_terms=['ubar'])
    U_z = np.mean(slp['ubar'], (0, 1))
    if len(slp['z']) > 1: 
        U_hub = np.interp(0, slp['z'], U_z)
    else: 
        U_hub = U_z

    uwake = (U_z - sl['ubar']) / U_hub
    wake_kwargs = {'x0': -1}
    # wake_kwargs = {'x0': 0.5}
    # wake_kwargs = {'bkgd': pre}
    # wake_kwargs = {'x0': 1}

    adm = case.turbineArray.turbines[0]
    x = sl['x'] #- adm.xloc
    y = sl['y'] #- adm.yloc
    z = sl['z'] #- adm.zloc

    ret = calibrate(wake_type='Gauss', uwake_ref=uwake, x=x, y=y, z=z, 
                    ctp=adm.ct, yaw=np.deg2rad(adm.yaw), wake_kwargs=wake_kwargs, 
                    mask_thresh=0.03, order=2, debug=True)
    
    print(ret)
    