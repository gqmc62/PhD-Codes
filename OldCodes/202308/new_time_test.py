# -*- coding: utf-8 -*-
"""

test how many pixels do i need in an input

"""
import numpy as np
from aotools.turbulence import ft_phase_screen

r0 = 0.3
dor0 = 0.5
N = 2**(np.arange(6) + 1)
d = r0*dor0
L0 = 20
l0 = 2e-10

for n in N:
    delta = d/n
    screen = ft_phase_screen(r0, n, delta, L0, l0)
    