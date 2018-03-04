import pytest
import rfpipe
from astropy import time
import numpy as np
import os.path

#
# flagging test script #
#
# success means that a standard set of simulated transients is found
# in four set ups: no timesub/flagging, no timesub, no flagging, with both.

# insert transients at first, middle, and last integration of simulated data
transients = [(0, i, 50, 5e-3, 0.1, -0.001, -0.001) for i in [0, 20, 42]]

# With/without flagging/timesub
inprefs = [{'dmarr': [50], 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients, 'memory_limit': 0.4,
            'timesub': None, 'flaglist': []},
           {'dmarr': [50], 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients, 'memory_limit': 0.4,
            'timesub': 'mean', 'flaglist': []},
           {'dmarr': [50], 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients, 'memory_limit': 0.4,
            'timesub': None},
           {'dmarr': [50], 'dtarr': [1], 'npix_max': 1024,
            'simulated_transient': transients, 'memory_limit': 0.4,
            'timesub': 'mean'}]

# FRB 121102 data
indata = [('16A-459_TEST_1hr.57623.72670021991.cut', 6, '16A-459_TEST_1hr.57623.72670021991.GN'),
          ('16A-459_TEST_1hr_000.57633.66130137732.scan7.cut', 7, '16A-459_TEST_1hr_000.57633.66130137732.GN'),
          ('16A-459_TEST_1hr_000.57633.66130137732.scan13.cut', 13, '16A-459_TEST_1hr_000.57633.66130137732.GN'),
          ('16A-496_sb32698778_1_02h00m.57638.42695471065.cut', 29, '16A-496_sb32698778_1_02h00m.57638.42695471065.GN'),
          ('16A-496_sb32698778_1_02h00m_001.57643.38562630787.cut', 29, '16A-496_sb32698778_1_02h00m_001.57643.38562630787.GN'),
          ('16A-496_sb32698778_1_02h00m.57645.38915079861.cut', 16, '16A-496_sb32698778_1_02h00m.57645.38915079861.GN'),
          ('16A-496_sb32698778_1_02h00m_000.57646.38643644676.cut', 32, '16A-496_sb32698778_1_02h00m_000.57646.38643644676.GN'),
          ('16A-496_sb32698778_1_02h00m_000.57648.37452900463.cut', 25, '16A-496_sb32698778_1_02h00m_000.57648.37452900463.GN'),
          ('16A-496_sb32698778_1_02h00m_001.57649.37461215278.cut', 31, '16A-496_sb32698778_1_02h00m_001.57649.37461215278.GN')]

# ideal SNR of FRB 121102 detections at DM=560.0
snrs = {'16A-459_TEST_1hr.57623.72670021991.cut': 37.,
        '16A-459_TEST_1hr_000.57633.66130137732.scan7.cut': 174.,
        '16A-459_TEST_1hr_000.57633.66130137732.scan13.cut': 14.,
        '16A-496_sb32698778_1_02h00m.57638.42695471065.cut': 11.,
        '16A-496_sb32698778_1_02h00m_001.57643.38562630787.cut': 100.,
        '16A-496_sb32698778_1_02h00m.57645.38915079861.cut': 8,
        '16A-496_sb32698778_1_02h00m_000.57646.38643644676.cut': 15.,
        '16A-496_sb32698778_1_02h00m_000.57648.37452900463.cut': 25.,
        '16A-496_sb32698778_1_02h00m_001.57649.37461215278.cut': 29.}

needsdata = pytest.mark.skipif('repeater' not in os.getcwd(),
                               reason='Must be in repeater data directory')

# (1) simulated data

@pytest.fixture(scope="module", params=inprefs)
def stsim(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 32*4, 4, 5e3,
                                         datasource='sim')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


@pytest.fixture(scope="module")
def data_prep(stsim):
    data = rfpipe.source.read_segment(stsim, 0)
    return rfpipe.source.data_prep(stsim, 0, data)


@pytest.fixture(scope="module")
def data_prep_rfi(stsim):
    data = rfpipe.source.read_segment(stsim, 0)
    data[5] += 0.01
    data[:, :, 30] += 0.01
    for i in range(110, 120):
        data[22, :, i, 0] += np.random.normal(0, 0.1, (stsim.nbl,))
    return rfpipe.source.data_prep(stsim, 0, data)


def test_fftw_sim_rfi(stsim, data_prep_rfi):
    for dmind in range(len(stsim.dmarr)):
        for dtind in range(len(stsim.dtarr)):
            canddatalist = rfpipe.search.dedisperse_image_fftw(stsim, 0,
                                                               data_prep_rfi,
                                                               dmind, dtind)
            assert len(canddatalist)


def test_fftw_sim(stsim, data_prep):
    for dmind in range(len(stsim.dmarr)):
        for dtind in range(len(stsim.dtarr)):
            canddatalist = rfpipe.search.dedisperse_image_fftw(stsim, 0,
                                                               data_prep,
                                                               dmind, dtind)
            assert len(canddatalist)


def test_cuda_sim_rfi(stsim, data_prep_rfi):
    rfgpu = pytest.importorskip('rfgpu')
    canddatalist = rfpipe.search.dedisperse_image_cuda(stsim, 0, data_prep_rfi)
    assert len(canddatalist)


def test_cuda_sim(stsim, data_prep):
    rfgpu = pytest.importorskip('rfgpu')
    canddatalist = rfpipe.search.dedisperse_image_cuda(stsim, 0, data_prep)
    assert len(canddatalist)


# (2) known FRB data

@pytest.fixture(scope="module", params=indata)
def stdata(request):
    sdmname, sdmscan, gainfile = request.param
    inmeta = rfpipe.metadata.sdm_metadata(sdmname, sdmscan)
    inprefs = {'dmarr': [555, 565], 'dtarr': [1], 'npix_max': 1024,
               'timesub': 'mean', 'gainfile': gainfile, 'sigma_image1': 6.}
    return rfpipe.state.State(inmeta=inmeta, inprefs=inprefs)


@pytest.fixture(scope="module")
def data_prep_data(stdata):
    data = rfpipe.source.read_segment(stdata, 0)
    return rfpipe.source.data_prep(stdata, 0, data)


@needsdata
def test_fftw_data(stdata, data_prep_data):
    canddatalist = []
    for dmind in range(len(stdata.dmarr)):
        for dtind in range(len(stdata.dtarr)):
            canddatalist += rfpipe.search.dedisperse_image_fftw(stdata, 0,
                                                                data_prep_data,
                                                                dmind, dtind)
    cc = rfpipe.candidates.calc_features(canddatalist)
    snrmax = cc.array['snr1'].max()
    assert snrmax >= 0.7*snrs[stdata.metadata.datasetId]


@needsdata
def test_cuda_data(stdata, data_prep_data):
    rfgpu = pytest.importorskip('rfgpu')

    canddatalist = rfpipe.search.dedisperse_image_cuda(stdata, 0,
                                                        data_prep_data)
    cc = rfpipe.candidates.calc_features(canddatalist)
    snrmax = cc.array['snr1'].max()
    assert snrmax >= 0.7*snrs[stdata.metadata.datasetId]


@needsdata
def test_prepnsearch(stdata, data_prep_data):
    rfgpu = pytest.importorskip('rfgpu')

    st.prefs.fftmode = 'cuda'
    cdl0 = prep_and_search(st, segment, data)
    cc0 = rfpipe.candidates.calc_features(cdl0)
    st.prefs.fftmode = 'fftw'
    cdl1 = prep_and_search(st, segment, data)
    cc1 = rfpipe.candidates.calc_features(cdl1)
    assert cdl0.array == cdl1.array
