import rfpipe
import pytest
from astropy import time
from numpy import ndarray

# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'npix_max': 512, 'chans': range(32), 'spw': [0]},
           {'simulated_transient': [(0, 30, 25, 10e-3, 1., 0.001, 0.001)],
            'maxdm': 50, 'dtarr': [1, 2], 'npix_max': 512, 'savecands': True,
            'savenoise': True, 'timesub': 'mean'}]
#TODO:      support arbitrary channel selection and
#           {'read_tdownsample': 2, 'read_fdownsample': 2, 'npix_max': 512},


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 2, 5e3,
                                         datasource='sim')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


# simulate two DMs
@pytest.fixture(scope="module")
def mockdata(mockstate):
    segment = 0
    data = rfpipe.source.read_segment(mockstate, segment)
    return rfpipe.source.data_prep(mockstate, segment, data)


@pytest.fixture(scope="module", params=[10, 100])
def mockdm(request, mockstate, mockdata):
    delay = rfpipe.util.calc_delay(mockstate.freq, mockstate.freq.max(),
                                   request.param, mockstate.inttime)
    return rfpipe.search.dedisperse(mockdata, delay)


@pytest.fixture(scope="module")
def wisdom(mockstate):
    return rfpipe.search.set_wisdom(mockstate.npixx, mockstate.npixy)


def test_dataprep(mockstate, mockdata):
    assert mockdata.shape == mockstate.datashape


def test_search(mockstate, mockdm, wisdom):
    segment = 0
    dmind = 0
    dtind = 0
    uvw = rfpipe.util.get_uvw_segment(mockstate, segment)

    canddatalist = rfpipe.search.search_thresh(mockstate, segment, mockdm,
                                               dmind, dtind, wisdom=wisdom)

    candcollection = rfpipe.candidates.calc_features(canddatalist)
    assert type(candcollection.array) == ndarray

    if mockstate.prefs.simulated_transient:
        assert len(candcollection.array)


def test_pipeline(mockstate):
    res = rfpipe.pipeline.pipeline_seg(mockstate, 0)

#    assert len(res) == len(mockstate.dmarr)*len(mockstate.dtarr)
