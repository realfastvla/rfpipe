.. _state:

==============
Search State
==============

The definition of a search in rfpipe is controlled with a ``State`` object. The state object is passed to many of the core functions that run the search, which then defines how the functions operate (e.g., values of DM to search, memory limits of the computer, number of pixels in an image, thresholds to apply, etc.).

A transient search is uniquely defined by the combination of metadata and preferences (see :ref:`metadataapi` :ref:`preferencesapi`). Metadata is fixed, while preferences are defined by the user. Separating these two concepts allows the definition of a ``State`` object that defines the search. Separating these concepts also makes the state portable, so preferences from a search conducted in realtime at the VLA can be reproduced by others offline.

Examples of how preferences can be used to control the search behavior:

* Set memory_limit or maxdm to define time windows for a search (segmenttimes),
* Set dmarr or maxdm to define dispersion measures to search, and
* Set uvoversample and npix_max to calculate the image size optimally with a preferred maximum.

Here is an example of how to create a ``State`` with 1 second of 10 ms-sampled data and default preferences::

  > import rfpipe; from astropy import time
  > t0 = time.Time.now().mjd
  > t1 = t0+1/(24*3600.)
  > inmeta = rfpipe.metadata.mock_metadata(t0, t1, 27, 32, 1024, 4, 10e3, datasource='vys', datasetid='test')
  > st = rfpipe.state.State(inmeta=inmeta)
  2018-04-16 14:02:55,045 - rfpipe.metadata - INFO - Generating mock metadata
  2018-04-16 14:02:55,101 - rfpipe.state - INFO - Metadata summary:
  2018-04-16 14:02:55,101 - rfpipe.state - INFO -      Working directory and fileroot: /lustre/evla/test/realfast, test.1.1
  2018-04-16 14:02:55,101 - rfpipe.state - INFO -      Using scan 1, source testsource
  2018-04-16 14:02:55,102 - rfpipe.state - INFO -      nants, nbl: 27, 351
  2018-04-16 14:02:55,102 - rfpipe.state - INFO -      nchan, nspw: 1024, 32
  2018-04-16 14:02:55,104 - rfpipe.state - INFO -      Freq range: 2.000 -- 4.000
  2018-04-16 14:02:55,104 - rfpipe.state - INFO -      Scan has 100 ints (1.0 s) and inttime 0.010 s
  2018-04-16 14:02:55,105 - rfpipe.state - INFO -      4 polarizations: ['A*A', 'A*B', 'B*A', 'B*B']
  2018-04-16 14:02:55,106 - rfpipe.state - INFO -      Ideal uvgrid npix=(2592, 2304) and res=83 (oversample 1.0)
  2018-04-16 14:02:55,106 - rfpipe.state - INFO - Pipeline summary:
  2018-04-16 14:02:55,106 - rfpipe.state - INFO -      Using 1 segment of 100 ints (1.0 s) with overlap of 0.0 s
  2018-04-16 14:02:55,106 - rfpipe.state - INFO -      Searching 100 of 100 ints in scan
  2018-04-16 14:02:55,106 - rfpipe.state - INFO -      Using pols ['A*A', 'B*B']
  2018-04-16 14:02:55,108 - rfpipe.state - WARNING -   Gainfile preference (/home/mchammer/evladata/telcal/2018/04/test.GN) is not a telcal file
  2018-04-16 14:02:55,108 - rfpipe.state - INFO - 
  2018-04-16 14:02:55,108 - rfpipe.state - INFO -      Using fftw for image1 search at 7 sigma using 1 thread.
  2018-04-16 14:02:55,108 - rfpipe.state - INFO -      Using 1 DMs from 0 to 0 and dts [1].
  2018-04-16 14:02:55,109 - rfpipe.state - INFO -      Using uvgrid npix=(2592, 2304) and res=83 with 100 int chunks.
  2018-04-16 14:02:55,109 - rfpipe.state - INFO -      Expect 0 thermal false positives per scan.
  2018-04-16 14:02:55,110 - rfpipe.state - INFO - 
  2018-04-16 14:02:55,111 - rfpipe.state - INFO -      Visibility/image memory usage is 1.1501568/4.777574400000001 GB/segment when using fftw imaging.

.. _segments:

Scans and Segments
===================

One of the key concepts of rfpipe is the division of data into time windows called segments. A fast transient search is unique in that the integration time is much smaller than typical. In particular, "slow" interferometry typically samples close to the inverse fringe rate of the longest baseline, since that defines the slowest sampling that preserves image quality and sensitivity. "Fast" interferometry instead can assume that it can search many integrations as if the array was stationary.

Rfpipe defines a time window called a ``segment`` that is at most equal to the inverse fringe rate of the longest baseline. The calculation of the segment times is made during the definition of the ``State`` and is kept as a property of that object. The fringe rate is nominally defined by the ``Metadata`` (e.g., by defining the baseline lengths), but the ``Preferences`` also influence the definition of segments by controlling the search algorithm (e.g., by limiting the size of the image or memory usage).

Using the ``State`` generated above, we can see how the segments were defined with properties::

  > st.nsegment
  1
  > st.segmenttimes
  array([[58224.83535931, 58224.83537088]])
