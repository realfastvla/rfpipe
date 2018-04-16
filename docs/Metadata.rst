=========================
Observational Metadata
=========================

Metadata define the properties of the data. Properties such as observing frequency, antenna positions, and time of observation are all metadata defined as the data are recorded.

Metadata are required to interpret the data that is read and to calculate the parameters of the search. In general, the metadata is provided by someone else (e.g., in an SDM file), although rfpipe has tools for generate metadata.

.. autofunction:: rfpipe.metadata.mock_metadata
  :noindex:

This can be useful for simulating data with ``datasource='sim'``. However, the most common use case is reading data/metadata from an SDM file. In this case, metadata can be read directly from the SDM file.

.. autofunction:: rfpipe.metadata.sdm_metadata
  :noindex: