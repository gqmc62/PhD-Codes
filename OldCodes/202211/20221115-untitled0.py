# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:26:09 2022

@author: gqmc62
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
    rGL : float
        Required turbulent strength at the Ground Layer.
    rHA : float
        Required turbulent strength at the High Altitude.
    np.NaN : np.NaN
        If the given situation is impossible.
    """
    k = 2*np.pi/wvl
    c2ndz_tot = r0**(-5./3.)/(0.423*k**2/np.cos(zenith_angle))
    c2ndz_HA = Rytov / (0.5631 * k**(7./6.)
                        / (np.cos(zenith_angle))**(11./6.)
                        * HA**(5./6.))
    if c2ndz_HA >= c2ndz_tot :
        # print("Impossible to get this high Rytov number.\n"
        #       + "r0={:.2e}, Rytov={:.2e},\n".format(r0, Rytov)
        #       + "HA={:.2e}, wvl={:.2e},\n".format(HA, wvl)
        #       + "zenith_angle={:.2e}".format(zenith_angle))
        return np.NaN, np.NaN
    c2ndz_GL = c2ndz_tot - c2ndz_HA
    rGL = (0.423*k**2/np.cos(zenith_angle)*c2ndz_GL)**(-3./5.)
    rHA = (0.423*k**2/np.cos(zenith_angle)*c2ndz_HA)**(-3./5.)
    
    # print(c2ndz_tot, c2ndz_GL, c2ndz_HA)

    return rGL, rHA