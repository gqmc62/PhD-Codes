# slodar-covariance

This is a C module that is imported by Python 3. It contains 4 functions that can be called from Python. The first 3 functions calculate slope covariances between pairs of square subapertures for different power spectra (Kolmogorov, Von Karman and Boreman-Dainty). This is done via (rather inefficient) numerical integration - see equation 7 from Butterley et al (2006). The final function takes an array of slope covariances and applies tip/tilt subtraction i.e. equation 10 from the paper. The result is an array of SLODAR reference functions.

To compile the code on linux you need to have gcc, python 3, numpy and gsl 1.x installed. If you're lucky you will just have to type "python3 setup.py build". If there are no errors, it will create a folder called "build/lib.??" containing "slodar.??.so". Copy that .so file to the working folder and you should be ready to go.

There's an example for that makes Kolmogorov reference functions for an 8x8 WFS - run it with "python3 example.py". It should generate a set of reference functions and, if you have astropy installed, save them to a FITS file for convenient (so you can view them with e.g. ds9). Hopefully the parameters in the example file are self-explanatory.

Things to be aware of:

- The code generates 2D reference functions. If you only want 1D reference functions you just need to take a cut through the middle of the array in the right direction.

- The output is a 3D numpy array. As you iterate over the first index you get alternate x and y reference functions for delta = 0, 1, 2, ..., 7.

- The code assumes the subapertures are all square and completely unvignetted.

- The output units are arcsec^2.

- The scaling of the reference functions is for d/r0 = 1 (where d is the subaperture size). The source code says the wavelength is 500nm but this actually means r0 is defined at 500nm. The angle of arrival variance does not vary with wavelength (to a very good approximation).

- As in the 2006 paper, the code assumes geometrical propagation of light. Your subapertures will be quite small so if you're trying to measure turbulence that is not close to the ground you might want to consider including scintillation effects in the impulse response functions. The method for doing this is described by Goodwin et al (2007).

- Some of the variable names in the C code are not very sensible/logical - apologies for this.

- There's a small bug that means the x and y axes get swapped round between the covariances and reference functions. This isn't a big deal, it just means the reference functions are flipped around the diagonal.

