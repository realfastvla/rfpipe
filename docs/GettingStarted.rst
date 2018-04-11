==============
Getting Started
==============

.. _installation

Installation
==============

Rfpipe requires the anaconda installer and two new channels to find its dependencies. To install, try the following::

  conda config --add channels pkgw-forge
  conda config --add channels conda-forge
  conda create -n realfast numpy scipy cython matplotlib pwkit casa-tools casa-python casa-data numba astropy pyfftw
  source activate realfast
  pip install git+ssh://git@github.com/realfastvla/rfpipe

If you intend to contribute to rfpipe, you should instead install rfpipe from source. To do so, replace the "pip install" line with::

  git clone git@github.com:realfastvla/rfpipe.git
  cd rfpipe
  python setup.py install

.. _quickstart

Quick Start
==============

As a quick validation of the installation and search capability, one can search with a mock observation filled with random numbers. To do so, run::

  import rfpipe
  t0 = 0.
  t1 = 10./(24*3600)
  inmeta = rfpipe.metadata.mock_metadata(t0, t1, 27, 16, 512, 4, 100000, datasource='sim')
  st = rfpipe.state.State(inmeta=inmeta, inprefs={'maxdm': 100, 'memory_limit':2, 'fftmode': 'fftw', 'npix_max': 512})
  candcollection = rfpipe.pipeline.pipeline_scan(st)

The inmeta object is a Python dict that defines metadata for an observation that, among other parameters, is set to last 10 seconds sampled at 100000 microsecond per sample. The inprefs definition defines a set of preferences on how the pipeline should be run; in this case, we define a maximum dm and a few other parameters for a modest, CPU-based search.

The first logging shows the parameters of the search as defined in the pipeline "state". The state is used to dynamically calculate parameters of the search based on the input metadata and preferences (see :ref:`state` for details). The logging of the pipeline_scan function will show the reading (in this case simulating) of data and iterating the search over the data.