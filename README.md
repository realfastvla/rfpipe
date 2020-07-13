# rfpipe

A fast radio interferometric transient search library. Extends on [rtpipe](http://github.com/caseyjlaw/rtpipe).

This library supports real-time analysis for the realfast project and offline analysis of VLA data on a single workstation. Integration with the real-time VLA and cluster processing is provided by [realfast](http://github.com/realfastvla/realfast).

Planned future development include:
- Supporting other search algorithms.
- Supporting other interferometers by adding data and metadata reading functions.

[![Docs](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://realfastvla.github.io/rfpipe)
[![Build Status](https://travis-ci.org/realfastvla/rfpipe.svg?branch=main)](https://travis-ci.org/realfastvla/rfpipe)
[![codecov](https://codecov.io/gh/realfastvla/rfpipe/branch/main/graph/badge.svg)](https://codecov.io/gh/realfastvla/rfpipe)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/rfpipe/)
[![ASCL](https://img.shields.io/badge/ascl-1710.002-blue.svg?colorB=262255)](https://ascl.net/1710.002)

## Installation

`rfpipe` requires the [anaconda](http://anaconda.com) installer on Linux and OSX. The most reliable installation is for Python3.6 and adding conda-forge:

```
conda config --add channels conda-forge
conda create -n realfast python=3.6 numpy scipy cython matplotlib numba pyfftw bokeh
source activate realfast
pip install --extra-index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple casatools
pip install -e git+git://github.com/realfastvla/rfpipe#egg=rfpipe
```

## Dependencies

- numpy/scipy/matplotlib
- casa6 python libraries (for quanta and measures; available on Python 3.6 via pip)
- numba (for multi-core and gpu acceleration)
- rtpipe (for flagging; will be removed soon)
- astropy
- sdmpy
- pyfftw
- pyyaml
- attrs
- rfgpu (optional; for GPU FFTs)
- vys/vysmaw and vysmaw_reader (optional; to read vys data from VLA correlator)

## Citation
If you use rfpipe, please support open software by citing the record on the [Astrophysics Source Code Library](ascl.net) at http://ascl.net/1710.002. In AASTeX, you can do this like so:
```
\software{..., rfpipe \citep{2017ascl.soft10002L}, ...}
```
