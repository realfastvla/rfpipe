===============
Getting Started
===============

.. _installation:

Installation
==============

Rfpipe requires the `anaconda installer <https://conda.io/docs/user-guide/install/download.html>`_. The installation usually works best by adding two channels to find dependencies. To install rfpipe, try the following::

  conda config --add channels pkgw-forge
  conda config --add channels conda-forge
  conda create -n realfast numpy scipy cython matplotlib pwkit casa-tools casa-python casa-data numba astropy pyfftw
  source activate realfast
  pip install git+ssh://git@github.com/realfastvla/rfpipe

This creates a conda environment, installs the rfpipe dependencies there, activates it and installs rfpipe. Alternatively, you can use ``conda install`` in place of ``conda create`` if you prefer to install in your default conda environment. 

If you intend to contribute to rfpipe, you should instead install rfpipe from source. To do so, replace the "pip install" line with::

  git clone git@github.com:realfastvla/rfpipe.git
  cd rfpipe
  python setup.py install

Furthermore, you can run the latest test suite with ``pytest``::

  pip install pytest
  pytest

.. _quickstart:

Test Your Installation
=======================

As a quick validation of the installation and search capability, create a mock observation filled with random numbers. To do so, run::

  import rfpipe
  t0 = 0.
  t1 = 10./(24*3600)
  inmeta = rfpipe.metadata.mock_metadata(t0, t1, 27, 16, 512, 4, 100000, datasource='sim')
  st = rfpipe.state.State(inmeta=inmeta, inprefs={'maxdm': 100, 'memory_limit':2, 'fftmode': 'fftw', 'npix_max': 512})
  candcollection = rfpipe.pipeline.pipeline_scan(st)

At the creation of the pipeline ``State`` (see :ref:`state` for details), logging will describe the values in the metadata and configuration of the search pipeline. The inmeta object is a Python dict that defines metadata for an observation that, among other parameters, has a duration of 10 seconds with visibilities sampled at 100000 microsecond cadence. The inprefs definition defines a set of preferences on how the pipeline should be run; in this case, we define a maximum dm and a few other parameters for a modest, CPU-based search.

The ``pipeline_scan`` command will run the search, which will include statements about reading and preparing data, dedispersing, and searching for transient events. It should find none, since it searches simulated (random valued) data with no transient inserted.

If that works, then you may wish to learn more about how the pipeline state is created (see :ref:`state`).