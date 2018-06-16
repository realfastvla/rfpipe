from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe
import pytest
import os.path
from astropy import time
import numpy as np

_install_dir = os.path.abspath(os.path.dirname(__file__))


def test_sdmmeta():
    sdmfile = os.path.join(_install_dir, 'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')
    meta = rfpipe.metadata.sdm_metadata(sdmfile, 7)
    assert isinstance(meta, dict)


def test_sdm():
    sdmfile = os.path.join(_install_dir,
                           'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')
    inprefs = {'gainfile': os.path.join(_install_dir,
                                        'data/16A-459_TEST_1hr_000.57633.66130137732.GN')}
    state = rfpipe.state.State(sdmfile=sdmfile, sdmscan=7, inprefs=inprefs)
    res = rfpipe.pipeline.pipeline_seg(state, 0)


def test_state():
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 32*4, 4, 5e3,
                                         datasource='sim', antconfig='D')
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    st = rfpipe.state.State(preffile=preffile, inmeta=meta,
                            inprefs={'chans': range(10, 20)})
    assert st.chans == range(10, 20)
    assert len(rfpipe.util.get_uvw_segment(st, 0))
