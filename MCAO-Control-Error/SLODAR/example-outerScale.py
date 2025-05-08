
# Example showing how to generate SLODAR reference functions using C module

import numpy
import slodcov   # SLODAR C routines




# Map of subaperture locations
pupil = numpy.array([
    [0,0,0,1,1,0,0,0],
    [0,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [1,1,1,0,0,1,1,1],
    [1,1,1,0,0,1,1,1],
    [0,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [0,0,0,1,1,0,0,0] ],numpy.int32)

# Subaperture size (in m)
d = 0.0625

# Outer scale (in m)
L0 = 2.

# Azimuthal angle of binary separation
az_ang = 0.  

# Number of subapertures
nsubx = pupil.shape[0]


# Calculate slope covariances
slopeCov = slodcov.slopecovVK(nsubx,8,d,L0,az_ang)[:-2] #nsamp=8 hard-wired

# Apply tip/tilt subtraction to make SIRFs
refFuncs = slodcov.refFuncs(slopeCov,pupil)


# Save output arrays to fits files
from astropy.io import fits
fits.writeto('slopeCovarianceVK.fits', slopeCov, overwrite=True)
fits.writeto('referenceFunctionsVK.fits', refFuncs, overwrite=True)



