
# Example showing how to generate raw slope covariances in sub-integer steps

import numpy
import slodcov   # SLODAR C routines


# Pixels/subap to use in numerical integration (also defines step size)
nsamp = 8

# Max offset in subaperture-widths
maxoffset = 10

# Max offset in pixels
nmax = nsamp*maxoffset + 1

# Subaperture size (in m)
d = 0.0625

# Outer scale (in m)
L0 = 2.

# Calculate slope covariances
slopeCov = slodcov.slopecovVKsubint_xx(nsamp, nmax, d, L0)

# Save output arrays to fits files
from astropy.io import fits
fits.writeto('slopeCovarianceVKsubint.fits', slopeCov, overwrite=True)



