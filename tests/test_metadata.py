import rfpipe
import pytest
import os.path
from astropy import time

_install_dir = os.path.abspath(os.path.dirname(__file__))


def test_sdmmeta():
    sdmfile = os.path.join(_install_dir, 'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')
    meta = rfpipe.metadata.sdm_metadata(sdmfile, 7)
    assert isinstance(meta, dict)


def test_sdm():
    sdmfile = os.path.join(_install_dir,
                           'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')
    inprefs = {'gainfile': os.path.join(_install_dir,
                                        '16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1.GN')}
    state = rfpipe.state.State(sdmfile=sdmfile, sdmscan=7, inprefs=inprefs)
    res = rfpipe.pipeline.pipeline_seg(state, 0)


def test_state():
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 2, 5e3,
                                         datasource='sim')
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    st = rfpipe.state.State(preffile=preffile, inmeta=meta,
                            inprefs={'chans': range(10, 20)})
    assert st.chans == range(10, 20)
    assert len(st.get_uvw_segment(0))
