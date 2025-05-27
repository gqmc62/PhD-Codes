# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:35:16 2025

@author: gqmc62
"""

import numpy as np
import matplotlib.pyplot as plt

N = 16
x = np.arange(N) - N/2 + 0.5
yy,xx = np.meshgrid(x,x)

TRIAL = 50
center = np.zeros((TRIAL),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)

for trial in range(TRIAL):
    i = sample[trial]
    phase = (yy+xx)*(i*(np.pi/16))
    # plt.imshow(phase)
    # plt.colorbar()
    # plt.show()
    phase -= (yy+xx)*(np.pi/32)
    
    I = np.abs(np.fft.fftshift(np.fft.fft2(np.pad(np.exp(1j*phase),
                                           ((int(N/2),int(N/2)),
                                            (int(N/2),int(N/2))),
                                           mode='constant',constant_values=0))
                               )[int(N/2):int(3*N/2),
                                 int(N/2):int(3*N/2)])**2
    # plt.imshow(I)
    # plt.colorbar()
    # plt.show()
    
    center[trial] = np.sum(I*xx)/np.sum(I)
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# — Plot 1 on the left —
ax1.plot(sample, center)
ax1.set_xlim(-8, 8)
ax1.set_ylim(-8, 8)
ax1.set_xlabel('expected center (pixel)')
ax1.set_ylabel('measured center (pixel)')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')  # square axes

# — Plot 2 on the right —
ax2.plot(sample, center - sample)
ax2.set_xlim(min(sample), max(sample))     # or (-8, 8) if you prefer
ax2.set_xlim(-8, 8)
ax2.set_xlabel('expected center (pixel)')
ax2.set_ylabel('measurement error (pixel)')
ax2.grid(True)

# Tight layout to avoid overlap
plt.tight_layout()
plt.savefig('wfs-response.pdf',bbox_inches='tight')
plt.show()