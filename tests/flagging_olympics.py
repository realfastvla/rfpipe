import pytest
import rfpipe
import os.path
from astropy import time
import numpy as np

#
# flagging test script #
#
# success means that a standard set of simulated transients is found
# in four set ups: no timesub/flagging, no timesub, no flagging, with both.

transients = [(0, 10, 0, 5e-3, 0.1, -0.001, -0.001), (0, 30, 50, 5e-3, 0.1, 0.001, 0.001)]

inprefs = [{'maxdm': 100, 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients,
            'timesub': None, 'flaglist': []},
           {'maxdm': 100, 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients,
            'timesub': 'mean', 'flaglist': []},
           {'maxdm': 100, 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients,
            'timesub': None},
           {'maxdm': 100, 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients,
            'timesub': 'mean'}]


@pytest.fixture(scope="module", params=inprefs)
def st(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 32*4, 4, 5e3,
                                         datasource='sim')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


@pytest.fixture(scope="module")
def data(st):
    return rfpipe.source.read_segment(st, 0)


@pytest.fixture(scope="module")
def data_prep(st, data):
    return rfpipe.source.data_prep(st, 0, data)


@pytest.fixture(scope="module")
def data_prep_rfi(st, data):
    data[5] += 0.01
    data[:, :, 30] += 0.01
    for i in range(110, 120):
        data[22, :, i, 0] += np.random.normal(0, 0.1, (st.nbl,))
    return rfpipe.source.data_prep(st, 0, data)


def test_cuda(st, data_prep):
    rfgpu = pytest.importorskip('rfgpu')
    for dmind in range(len(st.dmarr)):
        canddatalist = rfpipe.search.dedisperse_image_cuda(st, 0, data_prep, dmind)
        assert len(canddatalist)


def test_fftw(st, data_prep):
    for dmind in range(len(st.dmarr)):
        for dtind in range(len(st.dtarr)):
            canddatalist = rfpipe.search.dedisperse_image_fftw(st, 0, data_prep, dmind, dtind)
            assert len(canddatalist)


def test_cuda_rfi(st, data_prep_rfi):
    rfgpu = pytest.importorskip('rfgpu')
    for dmind in range(len(st.dmarr)):
        canddatalist = rfpipe.search.dedisperse_image_cuda(st, 0, data_prep_rfi, dmind)
        assert len(canddatalist)


def test_fftw_rfi(st, data_prep_rfi):
    for dmind in range(len(st.dmarr)):
        for dtind in range(len(st.dtarr)):
            canddatalist = rfpipe.search.dedisperse_image_fftw(st, 0, data_prep_rfi, dmind, dtind)
            assert len(canddatalist)
