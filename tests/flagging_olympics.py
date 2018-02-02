import pytest
import rfpipe
import os.path
from astropy import time
import numpy as np

inprefs = [{'simulated_transient': [(0, 0, 0, 5e-3, 0.1, 0., 0.)], 'timesub': None,
            'flaglist': [], 'maxdm': 0, 'dtarr': [1], 'npix_max': 512},
            {'simulated_transient': [(0, 30, 25, 10e-3, 1., 0.001, 0.001)],
            'dmarr': [0, 100], 'dtarr': [1, 2], 'npix_max': 512, 'savecands': True,
            'savenoise': True, 'timesub': 'mean'}]


@pytest.fixture(scope="module", params=inprefs)
def st(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 32*4, 4, 5e3,
                                         datasource='sim')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


# simulate two DMs
@pytest.fixture(scope="module")
def data(st):
    segment = 0
    data_read = rfpipe.source.read_segment(st, segment)
    return rfpipe.source.data_prep(st, segment, data_read)


def test_cuda(st, data, dmind):
    segment = 0
    canddatalist = rfpipe.search.dedisperse_image_cuda(st, segment, data, dmind)
    assert len(canddatalist)


def test_fftw(st, data, dmind, dtind):
    segment = 0
    canddatalist = rfpipe.search.dedisperse_image_fftw(st, segment, data, dmind, dtind)
    assert len(canddatalist)
