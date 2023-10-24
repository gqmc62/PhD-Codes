# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:26:09 2022.

@author: gqmc62.
"""

import numpy as np

## calculate what r0 needed for each layer
## in a 2-layer atm model
## to get the desired total r0 and rytov

def distribute_coherent_length(r0, Rytov, HA, wvl, zenith_angle):
    """
    Calculate coherent length in 2-layer atm to get the given r0 and Rytov.

    Parameters
    ----------
    r0 : float
        (m)
        Desired total turbulence strength.
    Rytov : float
        (unitless)
        Desired Rytov number = sigma_chi^2 = variance of log amplitude.
    HA : float
        (m)
        High Altitude height.
    wvl : float
        (m)
        Propagation wavelength.
    zenith_angle : float#
        (radian)

    Returns
    -------
    r0_GL : float
        Required turbulent strength at the Ground Layer.
    r0_HA : float
        Required turbulent strength at the High Altitude.
    np.NaN : np.NaN
        If the given situation is impossible.
    np.inf : if no turbulence
    """
    if Rytov == 0:
        return r0, np.inf
    k = 2*np.pi/wvl
    c2ndz_tot = r0**(-5./3.)/(0.423*k**2/np.cos(zenith_angle))
    c2ndz_HA = Rytov / (0.5631 * k**(7./6.)
                        / (np.cos(zenith_angle))**(11./6.)
                        * HA**(5./6.))
    if c2ndz_HA >= c2ndz_tot :
        # print('WARNING: Some conditions are impossible. Return np.NaN.')
        return np.NaN, np.NaN
    c2ndz_GL = c2ndz_tot - c2ndz_HA
    r0_GL = (0.423*k**2/np.cos(zenith_angle)*c2ndz_GL)**(-3./5.)
    r0_HA = (0.423*k**2/np.cos(zenith_angle)*c2ndz_HA)**(-3./5.)

    return r0_GL, r0_HA