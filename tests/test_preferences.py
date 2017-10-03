import rfpipe
import pytest
import os.path
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

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
