=========================
Transient Search
=========================

A transient search will use a State object and data to produce a set of candidate transients. The state defines the way the search functions operate, including physical parameters, algorithms, and computational limits. The pipeline module (see :ref:`pipeline`) wraps both reading and searching functions for ease of use.

The data can be input from one of three sources:

* SDM file (a standard VLA and realfast data product),
* simulation (numpy random number generator), and
* vysmaw client (a real-time streaming protocol at the VLA).

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
