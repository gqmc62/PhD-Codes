# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:41:08 2023

@author: gqmc62

test ffts

"""

import numpy as np
from matplotlib import pyplot as plt

N = int(128)
n = np.arange(-N/2, N/2)

dx = 4*np.pi/N
x = n*dx

y = np.sin(x)
fy = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y)))
dfx = 1/N/dx

plt.plot(n*dx,y)
plt.title('sin(x)')
plt.xlabel('x')
plt.show()

plt.plot(2*np.pi*n*dfx,fy.imag)
plt.title('Im(F(sin(x)))')
plt.xlabel('2$\pi$x')
plt.xlim(-2,2)
plt.show()

iffy = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(fy)))
diffx = 1/N/dfx
plt.plot(n*diffx,iffy)
plt.xlabel('x')
plt.title('F$^{-1}$(F(sin(x)))')
plt.show()

print('numpy fft is')
if (np.isclose(y,iffy)).all() == True:
    print('This Fourier transform is Unitary.')
else:
    print('Further check this.')
if (np.isclose(np.abs(dfx*n[np.abs(fy)==(np.abs(fy)).max()]),1/2/np.pi)).all():
    print('This Fourier transform is using ordinay frequency')
else:
    print('Further check this.')
    

    
import pyfftw

fftw_y = pyfftw.empty_aligned((N), dtype=complex)
fftw_fy = pyfftw.empty_aligned((N), dtype=complex)
fftw_iffy = pyfftw.empty_aligned((N), dtype=complex)
fftw_F = pyfftw.FFTW(fftw_y, fftw_fy, direction='FFTW_FORWARD')
fftw_iF = pyfftw.FFTW(fftw_fy, fftw_iffy, direction='FFTW_BACKWARD')

y = np.sin(x)

plt.plot(n*dx,y)
plt.title('sin(x)')
plt.xlabel('x')
plt.show()


fftw_y[:] = np.fft.fftshift(y)
fy = np.fft.fftshift(fftw_F())

plt.plot(2*np.pi*n*dfx, fy.imag)
plt.title('Im(F(sin(x)))')
plt.xlabel('2$\pi$x')
plt.xlim(-2,2)
plt.show()


iffy = np.fft.ifftshift(fftw_iF())

plt.plot(n*diffx,iffy)
plt.xlabel('x')
plt.title('F$^{-1}$(F(sin(x)))')
plt.show()

print('pyfftw fft is')

if (np.isclose(y,iffy)).all() == True:
    print('This Fourier transform is Unitary.')
else:
    print('Further check this.')
if (np.isclose(np.abs(dfx*n[np.abs(fy)==(np.abs(fy)).max()]),1/2/np.pi)).all():
    print('This Fourier transform is using ordinay frequency')
else:
    print('Further check this.')
    
    
# shift test
N = 8
dx = 4
df = 1/N/dx
x_grid = np.arange(N)*dx - N/2*dx
x_plot = x_grid + 0.5*dx
y = np.zeros((N), dtype=float)
y[int(N/4):int(3*N/4)] = 1
plt.plot(x_plot,y,marker='o',label='y')
plt.grid()
plt.legend()
plt.show()



plt.plot(x_plot,np.abs(np.fft.fft(y)),marker='s',label='|fft(y)|',markersize=10)
plt.plot(x_plot,np.abs(np.fft.fftshift(np.fft.fft(y))),marker='s',markersize=10,label='|fftshift(fft(y))|')
plt.plot(x_plot,np.abs(np.fft.fft(np.fft.fftshift(y))),marker='o',label='|fft(fftshift(y))|')
plt.plot(x_plot,np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(y)))),marker='o',label='|fftshift(fft(fftshift(y)))|')
plt.grid()
plt.legend()
plt.show()

# fy = np.fft.fft(y*np.exp(-1j*0.5*x_grid/8*2*np.pi));plt.plot(x_plot,np.fft.fftshift(np.abs(fy)**2),marker='o')

plt.plot(x_plot,np.abs(np.fft.fftshift(np.fft.fft(y*np.exp(1j*x_grid*2*np.pi*-0.5*df)))),marker='o',label='shift by -0.5px')
plt.plot(x_plot,np.abs(np.fft.fftshift(np.fft.fft(y*np.exp(1j*x_grid*2*np.pi*0*df)))),marker='o',label='shift by 0px')
plt.plot(x_plot,np.abs(np.fft.fftshift(np.fft.fft(y*np.exp(1j*x_grid*2*np.pi*0.5*df)))),marker='o',label='shift by +0.5px')
plt.legend()
plt.grid()
plt.show()


