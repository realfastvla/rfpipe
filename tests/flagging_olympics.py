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
transients = [(0, i, 50, 5e-3, 0.1, -0.001, -0.001) for i in [0, 20]]

searchtype = 'imagek'

# With/without flagging/timesub
inprefs = [{'dmarr': [50], 'dtarr': [1], 'npix_max': 2048,
            'simulated_transient': transients, 'memory_limit': 1,
            'timesub': None, 'flaglist': [], 'searchtype': searchtype,
            'sigma_arm': 3, 'sigma_arms': 5, 'sigma_kalman': 1},
           {'dmarr': [50], 'dtarr': [1], 'npix_max': 2048,
            'simulated_transient': transients, 'memory_limit': 1,
            'timesub': 'mean', 'flaglist': [], 'searchtype': searchtype,
            'sigma_arm': 3, 'sigma_arms': 5, 'sigma_kalman': 1},
           {'dmarr': [50], 'dtarr': [1], 'npix_max': 2048,
            'simulated_transient': transients, 'memory_limit': 1,
            'timesub': None, 'searchtype': searchtype,
            'sigma_arm': 3, 'sigma_arms': 5, 'sigma_kalman': 1},
           {'dmarr': [50], 'dtarr': [1], 'npix_max': 2048,
            'simulated_transient': transients, 'memory_limit': 1,
            'timesub': 'mean', 'searchtype': searchtype,
            'sigma_arm': 3, 'sigma_arms': 5, 'sigma_kalman': 1}]

# FRB 121102 data
indata = [('16A-459_TEST_1hr.57623.72670021991.cut', 6, '16A-459_TEST_1hr.57623.72670021991.GN'),
          ('16A-459_TEST_1hr_000.57633.66130137732.scan7.cut', 7, '16A-459_TEST_1hr_000.57633.66130137732.GN'),
          ('16A-459_TEST_1hr_000.57633.66130137732.scan13.cut', 13, '16A-459_TEST_1hr_000.57633.66130137732.GN'),
          ('16A-496_sb32698778_1_02h00m.57638.42695471065.cut', 29, '16A-496_sb32698778_1_02h00m.57638.42695471065.GN'),
          ('16A-496_sb32698778_1_02h00m_001.57643.38562630787.cut', 29, '16A-496_sb32698778_1_02h00m_001.57643.38562630787.GN'),
          ('16A-496_sb32698778_1_02h00m.57645.38915079861.cut', 16, '16A-496_sb32698778_1_02h00m.57645.38915079861.GN'),
          ('16A-496_sb32698778_1_02h00m_000.57646.38643644676.cut', 32, '16A-496_sb32698778_1_02h00m_000.57646.38643644676.GN'),
          ('16A-496_sb32698778_1_02h00m_000.57648.37452900463.cut', 25, '16A-496_sb32698778_1_02h00m_000.57648.37452900463.GN'),
          ('16A-496_sb32698778_1_02h00m_001.57649.37461215278.cut', 31, '16A-496_sb32698778_1_02h00m_001.57649.37461215278.GN'),
          ('realfast_19A-331_sb35982945_1_1.58541.41868983796_1551262870989', 1, '19A-331_sb35982945_1_1.58541.41868983796.GN'),
          ('realfast_19A-331_sb35982945_1_1.58541.41868983796_1551267058385', 1, '19A-331_sb35982945_1_1.58541.41868983796.GN')]

# ideal SNR of FRB 121102 detections at DM=560.0
snrs = {'16A-459_TEST_1hr.57623.72670021991.cut': 37.,
        '16A-459_TEST_1hr_000.57633.66130137732.scan7.cut': 174.,
        '16A-459_TEST_1hr_000.57633.66130137732.scan13.cut': 14.,
        '16A-496_sb32698778_1_02h00m.57638.42695471065.cut': 11.,
        '16A-496_sb32698778_1_02h00m_001.57643.38562630787.cut': 100.,
        '16A-496_sb32698778_1_02h00m.57645.38915079861.cut': 8,
        '16A-496_sb32698778_1_02h00m_000.57646.38643644676.cut': 15.,
        '16A-496_sb32698778_1_02h00m_000.57648.37452900463.cut': 25.,
        '16A-496_sb32698778_1_02h00m_001.57649.37461215278.cut': 29.,
        'realfast_19A-331_sb35982945_1_1.58541.41868983796_1551262870989': 0,
        'realfast_19A-331_sb35982945_1_1.58541.41868983796_1551267058385': 0}

needsdata = pytest.mark.skipif('olympics' not in os.getcwd(),
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

@pytest.mark.simfftw
def test_fftw_sim_rfi(stsim, data_prep_rfi):
    cc = rfpipe.search.dedisperse_search_fftw(stsim, 0, data_prep_rfi)
    assert len(cc)


@pytest.mark.simfftw
def test_fftw_sim(stsim, data_prep):
    cc = rfpipe.search.dedisperse_search_fftw(stsim, 0, data_prep)
    assert len(cc)


@pytest.mark.simcuda
def test_cuda_sim_rfi(stsim, data_prep_rfi):
    rfgpu = pytest.importorskip('rfgpu')
    cc = rfpipe.search.dedisperse_search_cuda(stsim, 0, data_prep_rfi)
    assert len(cc)


@pytest.mark.simcuda
def test_cuda_sim(stsim, data_prep):
    rfgpu = pytest.importorskip('rfgpu')
    cc = rfpipe.search.dedisperse_search_cuda(stsim, 0, data_prep)
    assert len(cc)


# (2) known FRB data

@pytest.fixture(scope="module", params=indata)
def stdata(request):
    sdmname, sdmscan, gainfile = request.param
    inmeta = rfpipe.metadata.sdm_metadata(sdmname, sdmscan)
    inprefs = {'maxdm': 600, 'dtarr': [1], 'npix_max': 4096,
               'timesub': 'mean', 'gainfile': gainfile, 'sigma_image1': 7.,
               'sigma_arm': 3, 'sigma_arms': 5, 'sigma_kalman': 1,
               'searchfeatures': ('snr1', 'l1', 'm1', 'immax1'),
               'searchtype': 'image', 'memory_limit': 64,}
#               'flaglist': [('badchtslide', 5., 10), ('badchtslide', 4., 10), ('badchtslide', 3., 10), ('blstd', 4.0, 0.008), ('blstd', 3.0, 0.008)]}
    return rfpipe.state.State(inmeta=inmeta, inprefs=inprefs)

@needsdata
@pytest.mark.datafftw
def test_fftw_data(stdata):
    rfgpu = pytest.importorskip('rfgpu')
    stdata.prefs.fftmode = 'fftw'
    stdata.prefs.npix_max = 1024
    stdata.prefs.memory_limit = 5
    stdata.clearcache()
    stdata.summarize()
    cc = rfpipe.pipeline.pipeline_scan(stdata)

    if len(cc):
        snrmax = cc.array['snr1'].max()
    else:
        snrmax = 0
    snrnom = snrs[stdata.metadata.datasetId]
    if snrnom > 0:
        assert snrmax >= 0.5*snrnom, "Expected snr>{0}, but detected {1}".format(0.5*snrnom, snrmax)
    else:
        assert snrmax == snrnom, "Expected no detection, but snrmax is {0}".format(snrmax)


@needsdata
@pytest.mark.datacuda
def test_cuda_data(stdata):
    rfgpu = pytest.importorskip('rfgpu')
    stdata.prefs.fftmode = 'cuda'
    stdata.prefs.memory_limit = 5
    stdata.clearcache()
    stdata.summarize()
    cc = rfpipe.pipeline.pipeline_scan(stdata)

    if len(cc):
        snrmax = cc.array['snr1'].max()
    else:
        snrmax = 0
    snrnom = snrs[stdata.metadata.datasetId]
    if snrnom > 0:
        assert snrmax >= 0.7*snrnom, "Expected snr>{0}, but detected {1}".format(0.7*snrnom, snrmax)
    else:
        assert snrmax == snrnom, "Expected no detection, but snrmax is {0}".format(snrmax)


@needsdata
@pytest.mark.datafftwcuda
def test_prepnsearch(stdata, data_prep_data):
    rfgpu = pytest.importorskip('rfgpu')

    stdata.prefs.fftmode = 'cuda'
    stdata.prefs.memory_limit = 5
    stdata.clearcache()
    stdata.summarize()
    cc0 = rfpipe.pipeline.pipeline_scan(stdata)
    stdata.prefs.fftmode = 'fftw'
    stdata.prefs.npix_max = 1024
    stdata.prefs.memory_limit = 5
    stdata.clearcache()
    stdata.summarize()
    cc1 = rfpipe.pipeline.pipeline_scan(stdata)
    assert len(cc0.array) == len(cc1.array), "FFTW and CUDA search not returning the same results"
