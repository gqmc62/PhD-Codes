# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:30:43 2025

@author: gqmc62
"""

import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\trash\DumpImages')
reader = imageio.get_reader('pco1_Pupil_20140529-141935_486_289-388-186.mov')
os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes')
# for i, frame in enumerate(reader):
#   imageio.imwrite(f'frame_{i}.png',frame)

# for i, frame in enumerate(reader):
#     xmax = frame.shape[0]
#     ymax = frame.shape[1]
#     step = 100
#     xcount = xmax//step
#     ycount = ymax//step
#     plt.hlines(np.arange(ycount - 1)*step + step,0,xmax,color='r')
#     plt.vlines(np.arange(xcount - 1)*step + step,0,ymax,color='r')
#     plt.imshow(frame)
#     # plt.grid(color='red')
#     plt.title(i)
#     # plt.savefig('frame_{:}.pdf'.format(i))
#     plt.show()
#     plt.clf()
    
# i = 0,2,14,16

# for i in np.arange(43):

plt.clf()

fig, axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(8,8))

xmax = reader.get_data(0).shape[0]
ymax = reader.get_data(0).shape[1]
step = 100
xcount = xmax//step
ycount = ymax//step

ls = '-'

axes[0,0].pcolor(reader.get_data(0))
axes[0,0].hlines(np.arange(ycount - 1)*step + step,0,xmax,color='r',ls=ls)
axes[0,0].vlines(np.arange(xcount - 1)*step + step,0,ymax,color='r',ls=ls)
axes[0,0].set_xticklabels([])
axes[0,0].set_yticklabels([])
axes[0,0].set_title("(a)")

axes[0,1].pcolor(reader.get_data(37))
axes[0,1].hlines(np.arange(ycount - 1)*step + step,0,xmax,color='r',ls=ls)
axes[0,1].vlines(np.arange(xcount - 1)*step + step,0,ymax,color='r',ls=ls)
axes[0,1].set_xticklabels([])
axes[0,1].set_yticklabels([])
axes[0,1].set_title("(b)")

axes[1,0].pcolor(reader.get_data(2))#14
axes[1,0].hlines(np.arange(ycount - 1)*step + step,0,xmax,color='r',ls=ls)
axes[1,0].vlines(np.arange(xcount - 1)*step + step,0,ymax,color='r',ls=ls)
axes[1,0].set_xticklabels([])
axes[1,0].set_yticklabels([])
axes[1,0].set_title("(c)")

axes[1,1].pcolor(reader.get_data(29))#29
axes[1,1].hlines(np.arange(ycount - 1)*step + step,0,xmax,color='r',ls=ls)
axes[1,1].vlines(np.arange(xcount - 1)*step + step,0,ymax,color='r',ls=ls)
axes[1,1].set_xticklabels([])
axes[1,1].set_yticklabels([])
axes[1,1].set_title("(d)")

plt.tight_layout()
# plt.suptitle('{:}'.format(i))
plt.savefig('pupil-distortion.png',dpi=1000,bbox_inches='tight')

plt.show()

