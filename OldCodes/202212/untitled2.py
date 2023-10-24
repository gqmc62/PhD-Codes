# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:15:52 2022

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt

# 
# 
# 

def solve_side(A,B,C):
    return np.arccos((np.cos(C) + np.cos(A)*np.cos(B))/(np.sin(A)*np.sin(B)))

def solve_angle(a,b,c):
    return np.arccos((np.cos(c) - np.cos(a)*np.cos(b))/(np.sin(a)*np.sin(b)))

def solve_time(dec,lat,alt):
    return 90 - np.degrees(solve_angle(np.radians(90 - dec),
                                       np.radians(90 - lat),
                                       np.radians(90 - alt)))

lat = 55
month = np.arange(12)
dec = np.sin(month*2.*np.pi/12.)*23.5

t_rise = 6 - (solve_time(dec,lat,0))/90*6

alt = np.arange(-8,8)*10
# plt.plot(month,t_rise)