# rfpipe

A fast radio interferometric transient search pipeline for use at the VLA. 
Extends on [rtpipe](http://github.com/caseyjlaw/rtpipe).


[![Documentation Status](https://readthedocs.org/projects/rfpipe/badge/?version=latest)](http://rfpipe.readthedocs.io/en/latest/?badge=latest)

[![Build Status](https://travis-ci.org/realfastvla/rfpipe.svg?branch=master)](https://travis-ci.org/realfastvla/rfpipe)

[![codecov](https://codecov.io/gh/realfastvla/rfpipe/branch/master/graph/badge.svg)](https://codecov.io/gh/realfastvla/rfpipe)

## Installation

```
conda config --add channels pkgw-forge
conda config --add channels conda-forge
conda create -n realfast numpy scipy jupyter bokeh cython matplotlib pwkit casa-tools casa-python casa-data numba astropy pandas pyfftw
pip install git+ssh://git@github.com/realfastvla/sdmpy
pip install git+ssh://git@github.com/realfastvla/rfpipe
```
