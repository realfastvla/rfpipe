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
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 2, 5e3,
                                         datasource='sim')
    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    st = rfpipe.state.State(preffile=preffile, inmeta=meta,
                            inprefs={'chans': range(10, 20)})
    assert st.chans == range(10, 20)
    assert len(st.get_uvw_segment(0))


def test_cal():
    sdmfile = os.path.join(_install_dir,
                           'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')

    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    inprefs = {'gainfile': os.path.join(_install_dir,
                                        'data/16A-459_TEST_1hr_000.57633.66130137732.GN'),
               'maxdm': 0}

    st = rfpipe.state.State(sdmfile=sdmfile, sdmscan=7, preffile=preffile,
                            inprefs=inprefs)
    segment = 0
    data = rfpipe.source.read_segment(st, segment)
    datacal = rfpipe.calibration.apply_telcal(st, data, sign=1)
    datauncal = rfpipe.calibration.apply_telcal(st, data, sign=-1)
    assert np.all(datauncal == data)


def test_simcal():
    sdmfile = os.path.join(_install_dir,
                           'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')

    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    inprefs = {'gainfile': os.path.join(_install_dir,
                                        'data/16A-459_TEST_1hr_000.57633.66130137732.GN'),
               'maxdm': 0,
               'simulated_transient': [(0, 0, 0, 0, 1., 0., 0.)]}

    st = rfpipe.state.State(sdmfile=sdmfile, sdmscan=7, preffile=preffile,
                            inprefs=inprefs)
    segment = 0
    data = rfpipe.source.read_segment(st, segment)
    datacal = rfpipe.calibration.apply_telcal(st, data, sign=1)
    datauncal = rfpipe.calibration.apply_telcal(st, data, sign=-1)
    assert np.all(datauncal == data)
