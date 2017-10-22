Overview
###############

rfpipe is a library for interferometric radio transient searches. It is designed to run on a multi-core architectures using visibility data as input. Transient searches are typically run by forming images, but other algorithms are allowed. The primary use case is for fast sampled (millisecond scale) visibilities from the Very Large Array.

Installation
####################

Use the anaconda installer to create a new environment::

  conda config --add channels pkgw-forge
  conda config --add channels conda-forge
  conda create -n realfast numpy scipy cython matplotlib pwkit casa-tools casa-python casa-data numba astropy pyfftw
  source activate realfast
  pip install git+ssh://git@github.com/realfastvla/rfpipe


Quick Start
#############

In Python, run a simple search pipeline like::

  import rfpipe
  st = rfpipe.state.State(sdmfile='sdmname', sdmscan=sdmscan)
  candcollection = rfpipe.pipeline.pipeline_scan(st)

.. toctree::
   :maxdepth: 2
