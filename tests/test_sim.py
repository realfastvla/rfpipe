from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe, rfpipe.candidates
import pytest
from astropy import time
from numpy import degrees, nan, argmax, abs

tparams = [(0, 0, 0, 5e-3, 0.3, 0.0001, 0.0),]
# simulate no flag, transient/no flag, transient/flag
inprefs = [({'flaglist': [], 'chans': list(range(32)), 'sigma_image1': None,
             'spw': [0], 'savecandcollection': True, 'savenoise': True,
             'savecanddata': True, 'returncanddata': True, 'saveplots': True,
             'fftmode': 'fftw', 'searchtype': 'imagek'}, 1),
           ({'simulated_transient': tparams, 'dmarr': [0, 1, 2], 'dtarr': [1, 2],
             'savecanddata': True, 'savenoise': True, 'saveplots': True,
             'returncanddata': True, 'savecandcollection': True,
             'timesub': 'mean', 'fftmode': 'fftw', 'searchtype': 'imagek',
             'sigma_image1': 10, 'sigma_kalman': 1,
             'clustercands': True, 'flaglist': []}, 2),
           ({'simulated_transient': tparams, 'dmarr': [0, 1, 2], 'dtarr': [1, 2],
             'savecanddata': True, 'savenoise': True, 'saveplots': True,
             'returncanddata': True, 'savecandcollection': True,
             'timesub': 'cs', 'fftmode': 'fftw', 'searchtype': 'imagek',
             'sigma_image1': 10, 'sigma_kalman': 1,
             'clustercands': True, 'flaglist': []}, 2),]
#           ({'simulated_transient': tparams, 'dmarr': [0], 'dtarr': [1],
#             'savecands': True, 'savenoise': True,
#             'sigma_image1': 10, 'sigma_kalman': 1, 'sigma_arm': 2,
#             'sigma_arms': 4, 'timesub': None, 'fftmode': 'fftw',
#             'searchtype': 'armkimage', 'flaglist': []}, 2)  # sigma_arms forced very low
#TODO:      support arbitrary channel selection and
#           {'read_tdownsample': 2, 'read_fdownsample': 2, 'npix_max': 512},


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    inprefs, scan = request.param
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.1/(24*3600), 20, 4, 32*4, 2,
                                         5e3, scan=scan, datasource='sim',
                                         antconfig='D')
    return rfpipe.state.State(inmeta=meta, inprefs=inprefs)


# simulate two DMs
@pytest.fixture(scope="module")
def mockdata(mockstate):
    segment = 0
    data = rfpipe.source.read_segment(mockstate, segment)
    data[0, 0, 0, 0] = nan
    return rfpipe.source.data_prep(mockstate, segment, data)


@pytest.fixture(scope="module")
def mockcc(mockstate):
    cc = rfpipe.pipeline.pipeline_scan(mockstate)
    return cc


def test_dataprep(mockstate, mockdata):
    assert mockdata.shape == mockstate.datashape


def test_noise(mockstate, mockdata):
    for noises in rfpipe.candidates.iter_noise(mockstate.noisefile):
        assert len(noises)


def test_pipelinescan(mockcc):
    if mockcc.prefs.simulated_transient is not None:
        rfpipe.candidates.makesummaryplot(mockcc)
    assert mockcc is not None


def test_voevent(mockcc):
    if mockcc.prefs.simulated_transient is not None:
        name = rfpipe.candidates.make_voevent(mockcc)
        assert name is not None


def test_candids(mockcc):
    if mockcc.prefs.simulated_transient is not None:
        assert len(mockcc.candids)


def test_cc(mockcc):
    if mockcc.prefs.returncanddata:
        assert isinstance(mockcc.canddata, list)
        assert len(mockcc.canddata) == len(mockcc)

    if mockcc.prefs.savecandcollection:
        ccs = rfpipe.candidates.iter_cands(mockcc.state.candsfile)
        cc = sum(ccs)
        assert len(cc) == len(mockcc)
        if cc.prefs.returncanddata:
            assert isinstance(cc.canddata, list)
            assert len(cc.canddata) == len(cc)
            assert len(cc.canddata) == len(mockcc.canddata)


def test_phasecenter_detection():
    inprefs = {'simulated_transient': [(0, 1, 0, 5e-3, 0.3, -0.001, 0.),
                                       (0, 9, 0, 5e-3, 0.3, 0., 0.),
                                       (0, 19, 0, 5e-3, 0.3, 0.001, 0.)],
               'dmarr': [0], 'dtarr': [1], 'timesub': None, 'fftmode': 'fftw', 'searchtype': 'image',
               'sigma_image1': 10, 'flaglist': [], 'uvres': 60, 'npix_max': 128, 'max_candfrac': 0}

    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.1/(24*3600), 20, 4, 32*4, 2,
                                         5e3, scan=1, datasource='sim',
                                         antconfig='D')

    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs)
    cc = rfpipe.pipeline.pipeline_scan(st)
    assert cc.array['l1'][0] <= 0.
    assert cc.array['l1'][1] == 0.
    assert cc.array['l1'][2] >= 0.
    assert all(abs(cc.array['m1']) <= 0.0003)


def test_phasecenter_detection_shift():
    inprefs = {'simulated_transient': [(0, 1, 0, 5e-3, 0.3, -0.001, 0.),
                                       (0, 9, 0, 5e-3, 0.3, 0., 0.),
                                       (0, 19, 0, 5e-3, 0.3, 0.001, 0.)],
               'dmarr': [0], 'dtarr': [1], 'timesub': None, 'fftmode': 'fftw', 'searchtype': 'image',
               'sigma_image1': 10, 'flaglist': [], 'uvres': 60, 'npix_max': 128, 'max_candfrac': 0}

    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.1/(24*3600), 20, 4, 32*4, 2,
                                         5e3, scan=1, datasource='sim',
                                         antconfig='D')

    meta['phasecenters'] = [(t0, t0+0.01/(24*3600), degrees(0.001), 0.),
                            (t0+0.01/(24*3600), t0+0.05/(24*3600), 0., 0.),
                            (t0+0.05/(24*3600), t0+0.1/(24*3600), degrees(-0.001), 0.)]
    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs)
    cc = rfpipe.pipeline.pipeline_scan(st)
    assert all(cc.array['l1'] == 0.)
    assert all(cc.array['m1'] == 0.)


def test_wide_transient():
    print("Try injecting a transient of width 40ms at integration 8")
    inprefs = {'simulated_transient': [(0, 8, 0, 40e-3, 0.3, 0., 0.)],
           'dmarr': [0], 'dtarr': [1,2,4,8], 'timesub': None, 'fftmode': 'fftw', 'searchtype': 'image',
           'sigma_image1': 10, 'flaglist': [], 'uvres': 60, 'npix_max': 128, 'max_candfrac': 0}

    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.1/(24*3600), 20, 4, 32*4, 2,
                                         5e3, scan=1, datasource='sim',
                                         antconfig='D')
    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs)
    cc = rfpipe.pipeline.pipeline_scan(st)
    ind = argmax(cc.array['snr1'])
    assert cc.array['dtind'][ind] == 3
    assert cc.array['integration'][ind]*2**cc.array['dtind'][ind] == 8    
    
    print("Try injecting a transient of width 20ms at integration 8")
    inprefs['simulated_transient'] = [(0, 8, 0, 20e-3, 0.3, 0., 0.)]
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.1/(24*3600), 20, 4, 32*4, 2,
                                         5e3, scan=1, datasource='sim',
                                         antconfig='D')

    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs)

    cc = rfpipe.pipeline.pipeline_scan(st)
    ind = argmax(cc.array['snr1'])
    assert cc.array['dtind'][ind] == 2
    assert cc.array['integration'][ind]*2**cc.array['dtind'][ind] == 8    
