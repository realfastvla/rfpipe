from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import pytest
import rfpipe
from astropy import time


def test_create():
    st = rfpipe.state.State(validate=False, showsummary=False)
    assert st.metadata.atdefaults()


@pytest.fixture(scope="module")
def mockstate():
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.3/(24*3600), 27, 4, 32*4, 4,
                                         5e3, datasource='sim', antconfig='D')
    return rfpipe.state.State(inmeta=meta)


def test_mock(mockstate):
    assert mockstate.datashape == (60, 351, 128, 2)


def test_pol(mockstate):
    assert len(mockstate.metadata.pols_orig) == 4 and len(mockstate.pols) == 2


def test_mocknseg(mockstate):
    assert mockstate.nsegment == 1


def test_version(mockstate):
    assert mockstate.version


def test_clearcache(mockstate):
    segmenttimes = mockstate.segmenttimes
    mockstate.clearcache()
    mockstate.summarize()
    assert (segmenttimes == mockstate.segmenttimes).all()


def test_lowmem():
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.3/(24*3600), 27, 4, 32*4, 4,
                                         5e3, datasource='sim')
    st = rfpipe.state.State(inmeta=meta, inprefs={'memory_limit': 0.1})
    assert st.validate()
