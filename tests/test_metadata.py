import rfpipe
import pytest
import evla_mcast
import os.path
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

_install_dir = os.path.abspath(os.path.dirname(__file__))

@pytest.fixture(scope="module")
def config():
    config = evla_mcast.scan_config.ScanConfig(vci=os.path.join(_install_dir, 'data/vci.xml'),
                                               obs=os.path.join(_install_dir, 'data/obs.xml'),
                                               ant=os.path.join(_install_dir, 'data/antprop.xml'),
                                               requires=['ant', 'vci', 'obs'])
    config.stopTime = config.startTime+100/(24*3600.)

    return config


@pytest.fixture(scope="module", params=[{'npix_max': 128},
                                        {'memory_limit': 1., 'maxdm': 100},
                                        {'maxdm': 100}])
def inprefs(request):
    return request.param


def test_configstate(config, inprefs):
    st = rfpipe.state.State(config=config, inprefs=inprefs, preffile=None)

    assert st.nints
    assert st.metadata.nints
    assert st.metadata.endtime_mjd
    assert len(st.segmenttimes)


def test_metastate(config, inprefs):
    meta = rfpipe.metadata.config_metadata(config, datasource='sim')

    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs, preffile=None)

    assert st.nints
    assert st.metadata.nints
    assert st.metadata.endtime_mjd
    assert len(st.segmenttimes)


def test_sim(config, inprefs):
    meta = rfpipe.metadata.config_metadata(config, datasource='sim')

    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs,
                            preffile=os.path.join(_install_dir,
                                                  'data/realfast.yml'))

    segment = 0
    data = rfpipe.source.read_segment(st, segment)
    assert data.shape == st.datashape


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
