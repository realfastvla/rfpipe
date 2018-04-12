=========================
Observational Metadata
=========================

Metadata define the properties of the data. Properties such as observing frequency, antenna positions, and time of observation are all defined as the data are recorded.

Metadata are required to interpret the data that is read and to calculate the parameters of the search. In general, the metadata is provided by someone else (e.g., in an SDM file), although rfpipe has tools for generate metadata.

.. _metadataapi:

Metadata API
=========================

.. autoclass:: rfpipe.metadata.Metadata
   :members:
   :undoc-members:
