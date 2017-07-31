import rfpipe
import pytest
from astropy import time
from numpy import array
import os.path

_install_dir = os.path.abspath(os.path.dirname(__file__))

# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'npix_max': 512}]
candlocs = [array([0, 10, 0, 0, 0])]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
        t0 = time.Time.now().mjd
        meta = rfpipe.metadata.mock_metadata(t0, t0+0.3/(24*3600), 27, 4, 2,
                                             5e3, datasource='sim')
        return rfpipe.state.State(inmeta=meta, inprefs=request.param)


@pytest.fixture(scope="module", params=candlocs)
def candloc(request):
    return request.param


def test_candidate(mockstate, candloc):
    candidate = rfpipe.reproduce.pipeline_candidate(mockstate, candloc)
    assert isinstance(candidate, dict)
    assert candidate.keys()[0] == candloc


def test_parse():
    candsfile = os.path.join(_install_dir,
                             'data/cands_17A-396_TEST_30m_001.57849.887411006945_merge.pkl')
    canddflist = rfpipe.reproduce.oldcands_read(candsfile)
    assert len(canddflist) == 43
    st, df = canddflist[0]
    assert isinstance(st, rfpipe.state.State)
    assert len(df) == 18
