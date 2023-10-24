import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import phasescreen
color_map = plt.cm.get_cmap('Blues')
reversed_color_map = color_map.reversed()

scrnStrengths=20e-2
scrn_size=8192
d_in=0.9e-3
L0=20
random_seed_num=5
l0=1.5e-10
sample_scrn_size=512
wvl=500e-9
z_in=15e3
z_out=0
dz=z_in-z_out
d_out=d_in

x=np.arange(-scrn_size/2.,scrn_size/2.)

xx,yy=np.meshgrid(x,x)

edge1=int((scrn_size-sample_scrn_size)/2)
edge2=int((scrn_size+sample_scrn_size)/2)

phase_scrn=phasescreen.ft_sh_phase_screen(scrnStrengths,
                        scrn_size,d_in,L0,l0,
                        seed=random_seed_num)

print('plotting1')

plt.pcolor(x[edge1:edge2]*d_in,x[edge1:edge2]*d_in,
           phase_scrn[edge1:edge2,edge1:edge2],
           cmap=reversed_color_map,shading='auto')
plt.title('phasescrn')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('phasescrn')
plt.clf()

slopex=np.zeros((scrn_size,scrn_size))
slopey=slopex
for i in np.arange(scrn_size):
    for j in np.arange(scrn_size):
        dx=phase_scrn[i,j-1]-phase_scrn[i-1,j-1]
        dy=phase_scrn[i-1,j]-phase_scrn[i-1,j-1]
        slopex[i,j]=dx*wvl/2./np.pi/d_in
        slopey[i,j]=dy*wvl/2./np.pi/d_in
        
slopesize=np.sqrt(slopex**2+slopey**2)
slopedeviate=slopesize*dz

print(slopedeviate.max())

# plt.quiver(xx[edge1:edge2,edge1:edge2]*d_in,yy[edge1:edge2,edge1:edge2]*d_in,
#            slopex[edge1:edge2,edge1:edge2],slopey[edge1:edge2,edge1:edge2],
#            units='width')

plt.pcolor(x[edge1:edge2]*d_out,x[edge1:edge2]*d_out,
           slopedeviate[edge1:edge2,edge1:edge2],cmap=reversed_color_map,shading='auto')

plt.title('slopesize+{:5.2f}'.format(slopedeviate.max()))
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('phaseslopesize')
plt.clf()