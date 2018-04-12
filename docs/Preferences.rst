=========================
Search Preferences
=========================

Preferences are a way for a user to guide the definition of the transient search. The preferences are a way to succinctly define those preferences, which are later interpreted by the search state. Preferences have sensible defaults, so the state can be defined even if no preferences are provided.

The most commonly used preferences define the search parameters (e.g., the range of dispersion measures), the computational limits (e.g., the maximum memory or maximum image size), and the search algorithm (e.g., CUDA or FFTW imaging).

The simplest way to define the search preferences is by passing arguments during the definition of the State object. This call can take a file as input (a preffile) and overload preference values with the inprefs (a dict).

.. _preferencesapi:

Preferences API
================

.. autoclass:: rfpipe.preferences.Preferences
   :members:
   :undoc-members:
