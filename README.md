# rfpipe

A fast radio interferometric transient search library. Extends on [rtpipe](http://github.com/caseyjlaw/rtpipe).

This library primarily supports offline analysis of VLA data on a single workstation. Integration with the real-time VLA and cluster processing is provided by [realfast](http://github.com/realfastvla/realfast).

Planned future development include:
- Supporting other search algorithms.
- Extending support for GPUs.
- Supporting other interferometers by adding data and metadata reading functions.
- Remove dependence on rtpipe.
- Python 3 support.

[![Docs](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://realfastvla.github.io/rfpipe)
[![Build Status](https://travis-ci.org/realfastvla/rfpipe.svg?branch=master)](https://travis-ci.org/realfastvla/rfpipe)
[![codecov](https://codecov.io/gh/realfastvla/rfpipe/branch/master/graph/badge.svg)](https://codecov.io/gh/realfastvla/rfpipe)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/rfpipe/)
[![ASCL](https://img.shields.io/badge/ascl-1710.002-blue.svg?colorB=262255)](https://ascl.net/1710.002)

## Installation

`rfpipe` requires the [anaconda](http://anaconda.com) installer on Linux and OSX. The most reliable install process adds two custom channels and a new build environment.

```
conda config --add channels pkgw-forge
conda config --add channels conda-forge
conda create -n realfast numpy scipy cython matplotlib pwkit casa-tools casa-python casa-data numba pyfftw
source activate realfast
pip install -e git+git://github.com/realfastvla/rfpipe#egg=rfpipe
```

## Dependencies

- numpy/scipy/matplotlib
- pwkit casa environment (for quanta and measures)
- numba (for multi-core and gpu acceleration)
- rtpipe (for flagging; will be removed soon)
- astropy (<3.0; for Python2 and 3 compatibility)
- sdmpy
- pyfftw
- pyyaml
- attrs
- pycuda and pyfft (optional; for GPU FFTs)
- vys/vysmaw and vysmaw_reader (optional; to read vys data from VLA correlator)

## Citation
If you use rfpipe, please support open software by citing the record on the [Astrophysics Source Code Library](ascl.net) at http://ascl.net/1710.002. In AASTeX, you can do this like so:
```
\software{..., rfpipe \citep{2017ascl.soft10002L}, ...}
```
