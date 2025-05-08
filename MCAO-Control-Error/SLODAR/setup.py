#!/usr/bin/env python3
# encoding: utf-8

from distutils.core import setup, Extension
import numpy

slodcov_module = Extension('slodcov', sources = ['slodcov.c'], include_dirs = [numpy.get_include()], libraries = ['gsl','gslcblas'],)

setup(name='slodcov',
      version='0.1.0',
      description='C SLODAR covariance module',
      ext_modules=[slodcov_module])




