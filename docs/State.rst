==============
Search State
==============

The definition of a search in rfpipe is controlled with a State object (class docs at :ref:`stateauto`). The state object is passed to many of the core functions that run the search, which then defines how the functions operate (e.g., values of DM to search, memory limits of the computer, number of pixels in an image, thresholds to apply, etc.).

All decisions about running the pipeline are driven from the metadata (e.g., observing frequency, array configuration, number of integrations, etc.) and preferences. These are defined with the Metadata (see :ref:`metadataauto`) and Preferences (see :ref:`preferencesauto`) classes. For a given metadata and preference object, the state is uniquely defined. This design allows users to take the preferences from a search conducted in realtime at the VLA and reproduce the analysis on their own computer.

.. _stateapi:

State API
==============

.. autoclass:: rfpipe.state.State
   :members:
   :undoc-members:

