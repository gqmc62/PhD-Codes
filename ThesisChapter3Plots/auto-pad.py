# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:43:35 2025

@author: gqmc62
"""

import matplotlib.pyplot as plt
import numpy as np
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov


N = 128
pixel_scale = 0.05/16
r0 = 0.05
L0 = 20

PS = PhaseScreenKolmogorov(N, pixel_scale, r0, L0)

for i in np.arange(N):
    PS.add_row()
screen = PS.scrn
screen -= screen.mean()
screen = np.angle(np.exp(1j*screen))

temp = screen[N//4:3*N//4,N//4:3*N//4]

screen1 = np.zeros_like(screen)
screen1[N//4:3*N//4,N//4:3*N//4] = temp
screen2 = np.pad(temp,(N//4,N//4),mode='symmetric')
screen3 = np.pad(temp,(N//4,N//4),mode='wrap')


plt.imshow(screen)
plt.colorbar()
plt.hlines([N//4,3*N//4],xmin=N//4,xmax=3*N//4,color='r')
plt.vlines([N//4,3*N//4],ymin=N//4,ymax=3*N//4,color='r')
plt.show()

plt.imshow(screen1)
plt.colorbar()
plt.hlines([N//4,3*N//4],xmin=N//4,xmax=3*N//4,color='r')
plt.vlines([N//4,3*N//4],ymin=N//4,ymax=3*N//4,color='r')
plt.show()

plt.imshow(screen2)
plt.colorbar()
plt.hlines([N//4,3*N//4],xmin=N//4,xmax=3*N//4,color='r')
plt.vlines([N//4,3*N//4],ymin=N//4,ymax=3*N//4,color='r')
plt.show()

plt.imshow(screen3)
plt.colorbar()
plt.hlines([N//4,3*N//4],xmin=N//4,xmax=3*N//4,color='r')
plt.vlines([N//4,3*N//4],ymin=N//4,ymax=3*N//4,color='r')
plt.show()
