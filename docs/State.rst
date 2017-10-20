Search State, Preferences, and Metadata
####################################################

The definition of a search in rfpipe is controlled with a the state.State object. A state is uniquely defined by two other classes, preferences and metadata. The preferences are parameters that define what the user wants to search, how to search, etc.. The metadata defines the way the data were recorded (e.g., observing frequency, time resolution, etc.). The state object combines these two other classes to derive run-time properties that control the search.

.. autoclass:: rfpipe.state.State
   :members:
   :undoc-members:

.. autoclass:: rfpipe.preferences.Preferences
   :members:
   :undoc-members:

.. autoclass:: rfpipe.metadata.Metadata
   :members:
   :undoc-members:
