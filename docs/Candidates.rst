==============
Candidates
==============

The search functions (or the pipeline) will return objects that summarize the candidates detected. The goal of the search process is to reduce the data volume (e.g., the TB per hour of visibilities) into a manageable summary of candidate transients. This out product is known as a candcollection.

.. autoclass:: rfpipe.candidates.CandCollection
   :members:
   :undoc-members:

Each candcollection has an ``array`` object that is a numpy array that summarizes the candidate transients. The ``array`` is defined with the first five columns for the location of the candidate (sometimes referred to as the "candidate location"):

* segment (see :ref:`segments`)
* integration
* dmind (index within the ``dmarr`` array of dispersion measures)
* dtind (index within the ``dtarr`` array of resampling widths)
* beamnum (index to a set of rephasing centers; not currently supported).

The ``Preferences`` support the definition of "features" that can be calculated for each candidate. These features are associated with each row in the ``array``. The default set of features is:

* snr1 (signal to noise ratio in the image)
* immax1 (maximum pixel value)
* l1 (peak pixel location offset in x in radians)
* m1 (peak pixel location offset in y in radians).

.. _candplots:

Candidate Visualization
=========================

Users have the option of generating visualizations of candidates from a search. This is done during the search, since the visualization includes the original data, images, and spectra for the candidate transient. A candidate visualization is saved as a png with a name that includes the input dataset, plus the location of the candidate in the data (e.g., segment, integration, etc.).

Since the generation of candidate visualizations takes longer than the basic analysis and takes up disk space, we (currently) only generate visualizations for a subset of all candidates. The current approach is the generate a candidate plot only at the maximum SNR in a segment.

The data used to generate the plot come from a ``CandData`` class:

.. autoclass:: rfpipe.candidates.CandData
   :members:
   :undoc-members:

The ``loc``, ``image``, and ``data`` objects contain numpy arrays used to generate the plot. They may also be useful for modifying the plot or some simple reanalysis of the candidate.