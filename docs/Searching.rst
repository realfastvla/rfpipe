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
  :noindex:

This function will read data according to how the source of data is defined in the ``Metadata`` object. For a given state, you can display the data source and read data like this::

  > st.metadata.datasource
  'sim'
  > data = rfpipe.source.read_segment(st, 0)
  Simulating data with shape (16, 351, 1024, 4)

There are several functions for preparing the data for a search, but most importantly the data will probably need to be calibrated and flagged. Again, given a state that defines the data preparation steps, one can prepare data with ``rfpipesource.data_prep``.

.. autofunction:: rfpipe.source.data_prep
  :noindex:

The transient search is similarly defined by the state and search functions take it as input. A range of data correction and search functions are available in the search module. Many algorithms have been implemented both for CPU (using the FFTW library) and GPU (using CUDA) environments. These are wrapped by ``rfpipe.search.prep_and_search``, which uses the ``Preferences`` to decide which function to use.

.. autofunction:: rfpipe.search.dedisperse_image_fftw
  :noindex:

.. autofunction:: rfpipe.search.dedisperse_image_cuda
  :noindex:

.. autofunction:: rfpipe.search.prep_and_search
  :noindex:

For a given state, you can search a given data array like this::

  > cc = rfpipe.search.prep_and_search(st, 0, data)
  2018-04-16 14:22:59,173 - rfpipe.source - INFO - Not applying online flags.
  2018-04-16 14:22:59,175 - rfpipe.calibration - WARNING - /home/mchammer/evladata/telcal/2018/04/test.GN is not a telcal file. No forward calibration to apply.
  2018-04-16 14:23:01,491 - rfpipe.source - INFO - flag by badchtslide: 0/32 pol-times and 2/2048 pol-chans flagged.
  2018-04-16 14:23:01,653 - rfpipe.source - INFO - flag by blstd: 52 of 32768 total channel/time/pol cells flagged.
  2018-04-16 14:23:01,726 - rfpipe.source - INFO - No visibility subtraction done.
  2018-04-16 14:23:01,821 - rfpipe.search - INFO - Correcting by delay/resampling 0/1 ints in single mode
  2018-04-16 14:23:02,554 - rfpipe.search - INFO - Imaging 16 ints (0-15) in seg 0 at DM/dt 0.0/1 with image 2592x2304 (uvres 83) with fftw
  2018-04-16 14:23:10,902 - rfpipe.search - INFO - 0 candidates returned for (seg, dmind, dtind) = (0, 0, 0)
  2018-04-16 14:23:10,956 - rfpipe.search - INFO - Correcting by delay/resampling 1/1 ints in single mode
  2018-04-16 14:23:11,308 - rfpipe.search - INFO - Imaging 15 ints (0-14) in seg 0 at DM/dt 16.2/1 with image 2592x2304 (uvres 83) with fftw
  2018-04-16 14:23:16,603 - rfpipe.search - INFO - 0 candidates returned for (seg, dmind, dtind) = (0, 1, 0)
  2018-04-16 14:23:16,655 - rfpipe.search - INFO - Correcting by delay/resampling 3/1 ints in single mode
  2018-04-16 14:23:16,949 - rfpipe.search - INFO - Imaging 13 ints (0-12) in seg 0 at DM/dt 32.5/1 with image 2592x2304 (uvres 83) with fftw
  2018-04-16 14:23:21,546 - rfpipe.search - INFO - 0 candidates returned for (seg, dmind, dtind) = (0, 2, 0)
  2018-04-16 14:23:21,597 - rfpipe.search - INFO - Correcting by delay/resampling 4/1 ints in single mode
  2018-04-16 14:23:21,866 - rfpipe.search - INFO - Imaging 12 ints (0-11) in seg 0 at DM/dt 48.8/1 with image 2592x2304 (uvres 83) with fftw
  2018-04-16 14:23:26,196 - rfpipe.search - INFO - 0 candidates returned for (seg, dmind, dtind) = (0, 3, 0)
  2018-04-16 14:23:26,246 - rfpipe.search - INFO - Correcting by delay/resampling 5/1 ints in single mode
  2018-04-16 14:23:26,491 - rfpipe.search - INFO - Imaging 11 ints (0-10) in seg 0 at DM/dt 65.0/1 with image 2592x2304 (uvres 83) with fftw
  2018-04-16 14:23:30,486 - rfpipe.search - INFO - 0 candidates returned for (seg, dmind, dtind) = (0, 4, 0)
  2018-04-16 14:23:30,535 - rfpipe.search - INFO - Correcting by delay/resampling 6/1 ints in single mode
  2018-04-16 14:23:30,757 - rfpipe.search - INFO - Imaging 10 ints (0-9) in seg 0 at DM/dt 81.2/1 with image 2592x2304 (uvres 83) with fftw
  2018-04-16 14:23:34,302 - rfpipe.search - INFO - 0 candidates returned for (seg, dmind, dtind) = (0, 5, 0)
  2018-04-16 14:23:34,349 - rfpipe.search - INFO - Correcting by delay/resampling 8/1 ints in single mode
  2018-04-16 14:23:34,527 - rfpipe.search - INFO - Imaging 8 ints (0-7) in seg 0 at DM/dt 97.5/1 with image 2592x2304 (uvres 83) with fftw
  2018-04-16 14:23:37,369 - rfpipe.search - INFO - 0 candidates returned for (seg, dmind, dtind) = (0, 6, 0)
  2018-04-16 14:23:37,378 - rfpipe.search - INFO - 0 candidates returned for seg 0
  2018-04-16 14:23:37,381 - rfpipe.candidates - INFO - Not saving candidates.
