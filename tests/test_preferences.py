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
