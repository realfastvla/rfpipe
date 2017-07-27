import rfpipe
import pytest
from astropy import time
from numpy import array

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
