.. _state:

==============
Search State
==============

The definition of a search in rfpipe is controlled with a ``State`` object. The state object is passed to many of the core functions that run the search, which then defines how the functions operate (e.g., values of DM to search, memory limits of the computer, number of pixels in an image, thresholds to apply, etc.).

The transient search is uniquely defined by the combination of metadata and preferences (see :ref:`metadataapi` :ref:`preferencesapi`). This design allows users to take the preferences from a search conducted in realtime at the VLA and reproduce the analysis on their own computer.

Examples of how preferences can be used to overload default behavior:

* Set memory_limit or maxdm to define segmenttimes,
* Set dmarr or maxdm to define dmarr, and
* Set (uvoversample, npix_max) to define (npixx, npixy).

.. _segments:

Scans and Segments
===================

One of the key concepts of rfpipe is the division of data into time windows. A fast transient search is unique in that the integration time is much smaller than typical. In particular, "slow" interferometry typically samples close to the inverse fringe rate of the longest baseline, since that defines the slowest sampling that preserves image quality and sensitivity. "Fast" interferometry instead can assume that it can search many integrations as if the array was stationary.

In rfpipe this is used to define a ``segment``, a time window that is at most equal to the inverse fringe rate of the longest baseline. The calculation of the segment times is made during the definition of the ``State`` and is kept as a property of that object. The fringe rate is nominally defined by the ``Metadata`` (e.g., by defining the baseline lengths), but the ``Preferences`` also influence the definition of segments by controlling the search algorithm (e.g., by limiting the size of the image).

.. _stateapi:

State API
==============

.. autoclass:: rfpipe.state.State
   :members:
   :undoc-members:

