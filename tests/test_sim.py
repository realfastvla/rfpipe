from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe, rfpipe.candidates
import pytest
from astropy import time
from numpy import degrees, nan

tparams = [(0, 0, 0, 5e-3, 0.3, 0.0001, 0.0),]
# simulate no flag, transient/no flag, transient/flag
inprefs = [({'flaglist': [], 'chans': list(range(32)),
             'spw': [0], 'savecandcollection': True, 'savenoise': True,
             'fftmode': 'fftw', 'searchtype': 'imagek'}, 1),
           ({'simulated_transient': tparams, 'dmarr': [0, 1, 2], 'dtarr': [1, 2],
             'savecanddata': True, 'savenoise': True, 'saveplots': True,
             'returncanddata': True,
             'timesub': None, 'fftmode': 'fftw', 'searchtype': 'imagek',
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


def test_dataprep(mockstate, mockdata):
    assert mockdata.shape == mockstate.datashape


def test_noise(mockstate, mockdata):
    for noises in rfpipe.candidates.iter_noise(mockstate.noisefile):
        assert len(noises)


def test_pipelinescan(mockstate):
    cc = rfpipe.pipeline.pipeline_scan(mockstate)
    if mockstate.prefs.simulated_transient is not None:
        rfpipe.candidates.makesummaryplot(mockstate.candsfile)
    assert cc is not None
    if mockstate.prefs.returncanddata:
        assert isinstance(cc.canddata, list)
        assert len(cc.canddata) == len(cc)


def test_phasecenter_detection():
    inprefs = {'simulated_transient': [(0, 0, 0, 5e-3, 0.3, 0., 0.),
                                       (0, 9, 0, 5e-3, 0.3, 0., 0.),
                                       (0, 10, 0, 5e-3, 0.3, 0.001, 0.),
                                       (0, 19, 0, 5e-3, 0.3, 0.001, 0.)],
               'dmarr': [0], 'dtarr': [1], 'timesub': None, 'fftmode': 'fftw', 'searchtype': 'image',
               'sigma_image1': 10, 'flaglist': [], 'uvres': 60, 'npix_max': 128, 'max_candfrac': 0}

    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.1/(24*3600), 20, 4, 32*4, 2,
                                         5e3, scan=1, datasource='sim',
                                         antconfig='D')

    print("Try no phasecenter shift")
    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs)
    cc = rfpipe.pipeline.pipeline_scan(st)
    assert all(cc.array['l1'][0:2] == 0.)
    assert not any(cc.array['l1'][2:] == 0.)
    assert all(cc.array['m1'] == 0.)

    print("Try phasecenter shift at integration 10")
    meta['phasecenters'] = [(t0, t0+0.05/(24*3600), 0., 0.),
                            (t0+0.05/(24*3600), t0+0.1/(24*3600), degrees(0.001), 0.)]
    st = rfpipe.state.State(inmeta=meta, inprefs=inprefs)
    cc = rfpipe.pipeline.pipeline_scan(st)
    assert all(cc.array['l1'] == 0.)
    assert all(cc.array['m1'] == 0.)
