.. _pipeline:

================================
Searching for an FRB in SDM Data
================================

A transient search is composed of a series of steps to read, prepare, and process data. These are wrapped in functions in the ``pipeline`` module. 

.. automodule:: rfpipe.pipeline
   :members:
   :undoc-members:

The ``pipeline_scan`` function simply iterates over calls to ``pipeline_segment`` for each segment. Candcollections from segments can be added together to produce new candcollections with an ``array`` that is the concatenation of the inputs.

As an example, we demonstrate a search of a small SDM of VLA data that contains a bright burst from FRB 121102 (see `this ApJ paper <https://ui.adsabs.harvard.edu/#abs/2017ApJ...850...76L/abstract>`_ for details). The SDM data and calibration tables are available publicly thanks to the `Harvard dataverse <https://doi.org/10.7910/DVN/TLDKXG>`_. To reproduce these steps, download the SDM and calibration files, "16A-459_TEST_1hr_000.57633.66130137732.scan7.cut" and "16A-459_TEST_1hr_000.57633.66130137732.GN", respectively.

To start simply, set up the state by reading metadata from the SDM and defining preferences for a modest search::

  > import rfpipe
  > st = rfpipe.state.State(sdmfile='16A-459_TEST_1hr_000.57633.66130137732.scan7.cut', sdmscan=7, inprefs={'dmarr': [0, 565], 'dtarr': [1,2,4], 'npix_max': 512, 'gainfile': '16A-459_TEST_1hr_000.57633.66130137732.GN', 'savecands': True})

This should produce a summary of the metadata and pipeline state, including lines such as::

  Reading metadata from 16A-459_TEST_1hr_000.57633.66130137732.scan7.cut, scan 7
  Metadata summary:
  ...
     Freq range: 2.488 -- 3.508
     Scan has 800 ints (4.0 s) and inttime 0.005 s
  ...
  Pipeline summary:
     Using 1 segment of 800 ints (4.0 s) with overlap of 0.2 s
     ...
     Found telcal file /lustre/evla/test/realfast/repeater/16A-459_TEST_1hr_000.57633.66130137732.GN
     Using fftw for image1 search at 7 sigma using 1 thread.
     Using 2 DMs from 0 to 565 and dts [1, 2, 4].
     Visibility/image memory usage is 1.1501568/1.6777216000000001 GB/segment when using fftw imaging.

Be sure that the telcal file is available locally and is found here. It is also good to confirm that the memory required for the search is available locally.

Given that the SDM is small (only 800 integrations and about 1 GB in size), it is easy to search the whole scan with the ``rfpipe.pipeline.pipeline.scan`` function. Simply run::

  > cc = rfpipe.pipeline.pipeline_scan(st)

This will log the data reading, preparation, and searching::

  Reading scan 7, segment 0/0, times 16:18:58.220 to 16:19:02.220
  ...
  Read telcalfile /lustre/evla/test/realfast/repeater/16A-459_TEST_1hr_000.57633.66130137732.GN with 1 sources, 3 times, 16 IFIDs, and 27 antennas
  Selecting 432 solutions from calibrator J0555+3948 separated by 3.1963333406019956 min.
  flag by badchtslide: 0/1600 pol-times and 0/512 pol-chans flagged.
  ...
  Correcting by delay/resampling 37/1 ints in single mode
  Imaging 763 ints (0-762) in seg 0 at DM/dt 560.0/1 with image 512x512 (uvres 104) with fftw
  Got one! SNR 24.5 candidate at (0, 400, 0, 0, 0) and (l,m) = (-0.0003943810096153846,0.0005446213942307692)
  ...

Congratulations, you just found FRB 121102 with the VLA!

Note that the flagging algorithm is tunable via the ``Preferences`` and can affect the quality of the search. Also, the typical search will set the preference for ``timesub='mean'``, which subtracts the mean visibility calculated over the segment (after flagging). In the present case, we do not use mean subtraction because the FRB is very bright and biases the mean in this short segment of data.

Looking at the ``CandCollection``::

  > print(cc)
  CandCollection for 16A-459_TEST_1hr_000.57633.66130137732.scan7.cut, scan 7 with 4 candidates

  > cc.array
  array([(0, 400, 0, 0, 0, 22.40985 , 4611.461, -0.00039438, 0.00054462),
         (0, 400, 1, 0, 0, 22.1478  , 4529.001, -0.00039438, 0.00054462),
         (0, 399, 2, 0, 0,  8.587416, 1359.625, -0.00039438, 0.00054462),
         (0, 400, 2, 0, 0, 21.640526, 4386.424, -0.00039438, 0.00054462)],
        dtype=[('segment', '<i4'), ('integration', '<i4'), ('dmind', '<i4'), ('dtind', '<i4'), ('beamnum', '<i4'), ('snr1', '<f4'), ('immax1', '<f4'), ('l1', '<f4'), ('m1', '<f4')])
  
  > cc.candmjd
  array([57633.67986366, 57633.67986366, 57633.6798636 , 57633.67986366])

  > cc.prefs
  Preferences(rfpipe_version='0.9.6', chans=None, spw=None, excludeants=(), selectpol='auto', fileroot=None, read_tdownsample=1, read_fdownsample=1, l0=0.0, m0=0.0, timesub=None, flaglist=[('badchtslide', 4.0, 10), ('blstd', 3.0, 0.05)], flagantsol=True, badspwpol=2.0, applyonlineflags=True, gainfile='16A-459_TEST_1hr_000.57633.66130137732.GN', simulated_transient=None, nthread=1, segmenttimes=None, memory_limit=16, maximmem=16, fftmode='fftw', dmarr=[560, 565, 570], dtarr=None, dm_maxloss=0.05, mindm=0, maxdm=0, dm_pulsewidth=3000, searchtype='image1', sigma_image1=7, sigma_image2=None, nfalse=None, uvres=0, npixx=0, npixy=0, npix_max=512, uvoversample=1.0, savenoise=False, savecands=False, candsfile=None, workdir='/lustre/evla/test/realfast/repeater', timewindow=30, loglevel='INFO')

You can see how the ``CandCollection`` has columns that are integers (segment, integration, dmind, dtind, beamnum). Those are indices that define the location of the transient in the data. For any given set of these five values, one can use the state to uniquely find (and reproduce) the state of the data associated with the candidate.

The other columns in this array are features of the data used to make the detection. The basic set of features are the SNR and location (l1, m1), however others are supported, include a range of statistical measurements of the image and spectra. The features are calculated at detection time, so they must be set with the state. You can see them like this::

  > st.features
  ('snr1', 'immax1', 'l1', 'm1')

