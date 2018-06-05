from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe
import pytest
from astropy import time
import numpy as np
import os.path

_install_dir = os.path.abspath(os.path.dirname(__file__))


# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'npix_max': 128, 'uvres': 500, 'timesub': None}]
candlocs = [np.array([0, 5, 0, 0, 0])]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
        t0 = time.Time.now().mjd
        meta = rfpipe.metadata.mock_metadata(t0, t0+0.05/(24*3600), 27, 4,
                                             32*4, 4, 5e3, datasource='sim',
                                             antconfig='D')
        return rfpipe.state.State(inmeta=meta, inprefs=request.param)


@pytest.fixture(scope="module", params=candlocs)
def candloc(request):
    return request.param


def test_candidate(mockstate, candloc):
    candcollection = rfpipe.reproduce.pipeline_candidate(mockstate, candloc)
    assert isinstance(candcollection, rfpipe.candidates.CandCollection)
    assert np.all(candcollection.array[0]['integration'] == candloc[1])


######
# TODO: figure out if full backwards compat is possible with python 2/3 compat
######
#def test_iter():
#    candsfile = os.path.join(_install_dir,
#                             'data/cands_test_58253.93959741861.2.1.pkl')
#    for cc in rfpipe.candidates.iter_cands(candsfile, select='candcollection'):
#        assert len(cc) > 0
#        assert cc.state.validate()
#
#
#def test_parse():
#    candsfile = os.path.join(_install_dir,
#                             'data/cands_17A-396_TEST_30m_001.57849.887411006945_merge.pkl')
#    candcollections = rfpipe.reproduce.oldcands_read(candsfile)
#    assert len(candcollections) == 43
#    st, cc = candcollections[0]
#    assert isinstance(st, rfpipe.state.State)
#    assert len(cc.array) > 0
