# rfpipe

A fast radio interferometric transient search library. Extends on [rtpipe](http://github.com/caseyjlaw/rtpipe).

This library primarily supports offline analysis of VLA data on a single workstation. Integration with the real-time VLA and cluster processing is provided by [realfast](http://github.com/realfastvla/realfast).

Planned future development include:
- Supporting other search algorithms.
- Extending support for GPUs.
- Supporting other interferometers by adding data and metadata reading functions.
- Remove dependence on rtpipe.
- Python 3 support.

[![Documentation Status](https://readthedocs.org/projects/rfpipe/badge/?version=latest)](http://rfpipe.readthedocs.io/en/latest/?badge=latest)

[![Build Status](https://travis-ci.org/realfastvla/rfpipe.svg?branch=master)](https://travis-ci.org/realfastvla/rfpipe)

[![codecov](https://codecov.io/gh/realfastvla/rfpipe/branch/master/graph/badge.svg)](https://codecov.io/gh/realfastvla/rfpipe)

## Installation

`rfpipe` requires the [anaconda](http://anaconda.com) installer on Linux and OSX. The most reliable install process adds two custom channels and a new build environment.

```
conda config --add channels pkgw-forge
conda config --add channels conda-forge
conda create -n realfast numpy scipy jupyter bokeh cython matplotlib pwkit casa-tools casa-python casa-data numba astropy pandas pyfftw
source activate realfast
pip install git+ssh://git@github.com/realfastvla/rfpipe
```

## Dependencies

- pwkit casa environment (for quanta and measures)
- numba for multi-core and gpu acceleration
- astropy for time
- sdmpy for reading sdm data
- rtpipe for flagging
- pyfftw
- pyyaml
- attrs
- pycuda and pyfft (optional; for GPU FFTs)
- vys/vysmaw and vysmaw_reader (optional; to read vys data at VLA)
