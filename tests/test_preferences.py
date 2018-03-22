from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe
import pytest
import os.path
import os

_install_dir = os.path.abspath(os.path.dirname(__file__))


def test_parse():
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    prefs = rfpipe.preferences.parsepreffile(preffile)

    assert isinstance(prefs, dict)


def test_parse2():
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    prefdict = rfpipe.preferences.parsepreffile(preffile)
    prefs = rfpipe.preferences.Preferences(**prefdict)

    assert isinstance(prefs, rfpipe.preferences.Preferences)


def test_overload():
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    prefdict = rfpipe.preferences.parsepreffile(preffile)

    prefdict['flaglist'] = []

    prefs = rfpipe.preferences.Preferences(**prefdict)

    assert prefs.flaglist == []


def test_name():
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    prefdict = rfpipe.preferences.parsepreffile(preffile)
    prefs0 = rfpipe.preferences.Preferences(**prefdict)

    prefdict['flaglist'] = []
    prefs = rfpipe.preferences.Preferences(**prefdict)

    assert prefs0.name != prefs.name


def test_state():
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    inmeta = rfpipe.metadata.mock_metadata(0, 1, 27, 16, 32, 4, 1e6)
    st = rfpipe.state.State(inmeta=inmeta, preffile=preffile,
                            inprefs={'chans': list(range(10))})
    assert st.chans == range(10)
