.. _pipeline:

=========================
Using a Search Pipeline
=========================

A transient search is composed of a series of steps to read, prepare, and process data. These are wrapped in functions in the ``pipeline`` module. 

.. automodule:: rfpipe.pipeline
   :members:
   :undoc-members:

To run a search on a single segment, we create a state::

  > import rfpipe
  > st = rfpipe.state.State(sdmfile='16A-459_TEST_1hr.57623.72670021991.cut', sdmscan=6, inprefs={'dmarr': [0, 560], 'npix_max': 1024, 'gainfile': '16A-459_TEST_1hr.57623.72670021991.GN'})

To read, prepare, and search the data for a single segment, run::

  > cc = rfpipe.pipeline.pipeline_seg(st, 0)

To read, prepare, and search the data for all segments in a scan, run::

  > cc = rfpipe.pipeline.pipeline_scan(st)

The ``pipeline_scan`` function simply iterates over calls to ``pipeline_segment`` for each segment. Candcollections from segments can be added together to produce new candcollections with ``array``s that are the concatenation of the inputs.