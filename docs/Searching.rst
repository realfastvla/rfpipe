=========================
Transient Search
=========================

A transient search will use a State object and data to produce a set of candidate transients. The state defines the way the search functions operate, including physical parameters, algorithms, and computational limits. The pipeline module (see :ref:`pipeline`) wraps both reading and searching functions for ease of use.

The data can be input from one of three sources:

* SDM file (a standard VLA and realfast data product),
* simulation (numpy random number generator), and
* vysmaw client (a real-time streaming protocol at the VLA).

Given a state object that defines the data source, one can read a single segment with ``rfpipe.source.read_segment``.

.. autofunction:: rfpipe.source.read_segment

There are several functions for preparing the data for a search, but most importantly the data will probably need to be calibrated and flagged. Again, given a state that defines the data preparation steps, one can prepare data with ``rfpipesource.data_prep``.

.. autofunction:: rfpipe.source.data_prep

The transient search is similarly defined by the state and search functions take it as input. A range of data correction and search functions are available in the search module. Many algorithms have been implemented both for CPU (using the FFTW library) and GPU (using CUDA) environments.

.. autofunction:: rfpipe.search.dedisperse_image_fftw

.. autofunction:: rfpipe.search.dedisperse_image_cuda

.. _sourceapi:

Source API
=========================

.. automodule:: rfpipe.source
   :members:
   :undoc-members:

.. _searchapi:

Search API
=========================

.. automodule:: rfpipe.search
   :members:
   :undoc-members:
