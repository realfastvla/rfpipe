=========================
Search Preferences
=========================

Preferences are a way for a user to define how a transient search for any kind of data/metadata. Preferences have sensible defaults, so the state can be defined even if no preferences are provided.

The most commonly used preferences define the search parameters (e.g., the range of dispersion measures), the computational limits (e.g., the maximum memory or maximum image size), and the search algorithm (e.g., CUDA or FFTW imaging).

The simplest way to define the search preferences is by passing arguments during the definition of the State object. This call can take a file as input (a preffile) and overload preference values with the inprefs (a dict).

The most common way of modifying ``Preferences`` is during the definition of the ``State``. To modify the preferences with an input dict::

  > import rfpipe; from astropy import time
  > t0 = time.Time.now().mjd
  > t1 = t0+1/(24*3600.)
  > inmeta = rfpipe.metadata.mock_metadata(t0, t1, 27, 32, 1024, 4, 10e3, datasource='vys', datasetid='test')
  > inprefs = {'maxdm': 100, 'memory_limit': 1.}
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
  2018-04-16 14:02:55,106 - rfpipe.state - INFO -      Using 11 segments of 16 ints (0.2 s) with overlap of 0.1 s
  2018-04-16 14:02:55,106 - rfpipe.state - INFO -      Searching 96 of 100 ints in scan
  2018-04-16 14:02:55,106 - rfpipe.state - INFO -      Using pols ['A*A', 'B*B']
  2018-04-16 14:02:55,108 - rfpipe.state - WARNING -   Gainfile preference (/home/mchammer/evladata/telcal/2018/04/test.GN) is not a telcal file
  2018-04-16 14:02:55,108 - rfpipe.state - INFO - 
  2018-04-16 14:02:55,108 - rfpipe.state - INFO -      Using fftw for image1 search at 7 sigma using 1 thread.
  2018-04-16 14:02:55,108 - rfpipe.state - INFO -      Using 7 DMs from 0.0 to 97.5 and dts [1].
  2018-04-16 14:02:55,109 - rfpipe.state - INFO -      Using uvgrid npix=(2592, 2304) and res=83 with 16 int chunks.
  2018-04-16 14:02:55,109 - rfpipe.state - INFO -      Expect 0 thermal false positives per scan.
  2018-04-16 14:02:55,110 - rfpipe.state - INFO - 
  2018-04-16 14:02:55,111 - rfpipe.state - INFO -      Visibility/image memory usage is 0.184025088/0.764411904 GB/segment when using fftw imaging.
