# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:14:14 2024

@author: gqmc62
"""

import numpy as np

def get_cndz_z_from_rytov_displace(wvl,dsub,rytov,displacement):


    A = np.log10(rytov)
    B = np.log10(displacement)
    C1 = np.log10(0.563*(2.0*np.pi/wvl)**(7./6.))
    C2 = 0.5*np.log10(0.182*0.423*(2.*np.pi)**2) - (7./6.)*np.log10(dsub)
    M = np.array([[   1., 5./6.],
                  [1./2.,    1.]])

    solve = np.linalg.inv(M) @ np.array([A - C1, B - C2])
    P = solve[0]
    Q = solve[1]

    cn2dz = 10.0**P
    z = 10.0**Q

    return cn2dz,z

def cn2dz2r0(cn2dz,wvl=500e-9):
    r0 = (0.423*(2.*np.pi/wvl)**2*cn2dz)**(-3./5.)
    return r0

def r02cn2(r0,wvl=500e-9):
    cn2dz = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2)
    return cn2dz

def angle_of_arrival(wvl,D,r0):
    alpha = np.sqrt(0.182 * wvl**2 * D**(-1./3.) * r0**(-5./3.))
    return alpha

def displacement(wvl,dsub,r0,z):
    displace = angle_of_arrival(wvl,dsub,r0) * z/dsub
    return displace

def rytov_number(wvl, r0, z):
    # calculate rytov number from rytov approximation
    k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
    cn2dz = r02cn2(r0, wvl)        # float; Turbulence strength.
    variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z**(5./6.)
    return variance_theory_approx

rytov_list = np.logspace(np.log10(0.1),np.log10(0.6),5)#np.array([0.6,0.5,0.4,0.3,0.2,0.1])#np.array([0.6])#
displace_list = np.logspace(np.log10(0.1),np.log10(1.0),5)#np.array([1.0,0.75,0.5,0.25,0.1])#np.array([1.])#

cn2dz_list = np.zeros((rytov_list.shape[0],displace_list.shape[0]),dtype=float)
z_list = np.zeros((rytov_list.shape[0],displace_list.shape[0]),dtype=float)

for i in np.arange(rytov_list.shape[0]):
  for j in np.arange(displace_list.shape[0]):
    rytov= rytov_list[i]
    displace = displace_list[j]
    temp = get_cndz_z_from_rytov_displace(500e-9,0.05,rytov,displace)
    cn2dz_list[i,j] = temp[0]
    z_list[i,j] = temp[1]
    
# print(cn2dz_list)
# print(z_list)

cn2dz_list_old = np.copy(cn2dz_list)
z_list_old = np.copy(z_list)

cn2dz_list[cn2dz_list_old>r02cn2(2e-2)] = np.nan
z_list[cn2dz_list_old>r02cn2(2e-2)] = np.nan

cn2dz_list[z_list_old>25e3] = np.nan
z_list[z_list_old>25e3] = np.nan

r0_list = cn2dz2r0(cn2dz_list)



print(np.sort(cn2dz2r0(cn2dz_list.flatten())))
print(np.sort(z_list.flatten()))

new_cn2dz_list = np.sort(cn2dz2r0(cn2dz_list.flatten()))
new_z_list = np.sort(z_list.flatten())

new_cn2dz_list=new_cn2dz_list[np.where(~ np.isnan(new_cn2dz_list))]
new_z_list=new_z_list[np.where(~ np.isnan(new_z_list))]

np.savetxt('cn2dz_list.txt',new_cn2dz_list)
np.savetxt('z_list.txt',new_z_list)

AS, RY = np.meshgrid(displace_list,rytov_list)
for i in np.arange(rytov_list.shape[0]):
    for j in np.arange(displace_list.shape[0]):
        p = RY[i,j]
        q = AS[i,j]
        r = r0_list[i,j]
        s = z_list[i,j]
        P = rytov_number(500e-9, r0_list[i,j], z_list[i,j])
        Q = displacement(500e-9,0.05,r0_list[i,j],z_list[i,j])
        if (not np.isnan(s)) and (not np.isnan(r)):
            
            if not np.isclose(p,P,rtol=1e-2):
                print('NO',p,P)
            if not np.isclose(q,Q,rtol=1e-2):
                print('NO',q,Q)
                
            print(p,q,r,s)
            