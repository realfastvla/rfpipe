.. _reproduce:

==========================================
Reproducing realfast Candidate Transients
==========================================

The rfpipe library has been developed for the realfast project, a real-time transient search instrument at the VLA. The realfast project will capture VLA visibility data for candidate transients detected in real time. Those results can be reproduced offline by rfpipe.

The ``rfpipe.reproduce`` defines a set of functions to make it easier to reproduce a candidate. The functions are defined as a series of operations. The output of the earlier operations an be input to the later stages; it is also fine to run the last stage, which will automatically run earlier stages.

.. autofunction:: rfpipe.reproduce.pipeline_dataprep

.. autofunction:: rfpipe.reproduce.pipeline_datacorrect

.. autofunction:: rfpipe.reproduce.pipeline_imdata

.. autofunction:: rfpipe.reproduce.pipeline_candidate

First, one needs to read the ``CandCollection`` locally. The ``iter_cands`` function provides a convenient way to step through the pickle files used to hold candidate data.

.. autofunction:: rfpipe.candidates.iter_cands

To get the first collection, you could take the pickle file output by the search produced in one of the other Use Cases (see :ref:`pipeline` or :ref:`simulating`)::

  > for cc in rfpipe.candidates.iter_cands('cands_16A-459_TEST_1hr_000.57633.66130137732.scan7.cut.7.1.pkl'):
  > print(cc)
  CandCollection for 16A-459_TEST_1hr_000.57633.66130137732.scan7.cut, scan 7 with 3 candidates

This will return the candcollection that defines all candidates and allows you to regenerate the state that found them. The candidate is defined with a set of five integers, which, when associated with a state, uniquely identifies the data used to make the detection. To capture this "candidate location"::

  > maxsnr = np.where(cc.array['snr1'] == cc.array['snr1'].max())[0][0]
  > candloc = cc.array[['segment', 'integration', 'dmind', 'dtind', 'beamnum']][maxsnr].copy().view(('i4', 5))
  > st = cc.state

(Some type wackiness here that will probably be simplified in the future.)

Now to reproduce each stage of the analysis, we can run::

  > data_prep = rfpipe.reproduce.pipeline_dataprep(st, candloc)
  > data_dmdt = rfpipe.reproduce.pipeline_datacorrect(st, candloc, data_prep=data_prep)
  > candcollection = rfpipe.reproduce.pipeline_imdata(st, candloc, data_dmdt=data_dmdt)
  > candcollection = rfpipe.reproduce.pipeline_candidate(st, candloc, candcollection=candcollection)

The final ``rfpipe.reproduce.pipeline_candidate`` function will run the final plot generation and saving of candidate pickle file but returns its input candcollection for convenience.
