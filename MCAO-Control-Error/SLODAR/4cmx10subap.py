
# Example showing how to generate SLODAR reference functions using C module

import numpy
import slodcov   # SLODAR C routines




# Map of subaperture locations
pupil = numpy.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]],numpy.int32)

# Subaperture size (in m)
d = 0.4/10.

# Azimuthal angle of binary separation
az_ang = 0.  

# Number of subapertures
nsubx = pupil.shape[0]

L0 = 4

# Calculate slope covariances
slopeCov = slodcov.slopecovVK(nsubx,8,d,L0,az_ang)[:-2] #nsamp=8 hard-wired

# Apply tip/tilt subtraction to make SIRFs
refFuncs = slodcov.refFuncs(slopeCov,pupil)


# Save output arrays to fits files
from astropy.io import fits
# fits.writeto('slopeCovariance_C.fits', slopeCov, overwrite=True)
fits.writeto('slodarRefFunc4cmx10subapVK4mL0.fits', refFuncs, overwrite=True)



