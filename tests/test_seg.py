import rfpipe
import pytest
from astropy import time

# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'npix_max': 32, 'sigma_image1': 0},
           {'flaglist': [], 'npix_max': 32, 'sigma_image1': 0, 'maxdm': 100}
           ]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.5/(24*3600), 27, 4, 2, 5e3,
                                         datasource='sim')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


@pytest.fixture(scope="module")
def mockdata(mockstate):
    segment = 0
    data = rfpipe.source.read_segment(mockstate, segment)
    return rfpipe.source.data_prep(mockstate, data)


def test_dataprep(mockstate, mockdata):
    assert mockdata.shape == mockstate.datashape


def test_search(mockstate):
    segment = 0
    wisdom = rfpipe.search.set_wisdom(mockstate.npixx, mockstate.npixy)
    for dmind in range(len(st.dmarr)):
        delay = rfpipe.util.calc_delay(mockstate.freq, mockstate.freq.max(),
                                       st.dmarr[dmind], mockstate.inttime)
        data_dm = rfpipe.search.dedisperse(mockdata, delay)

        for dtind in range(len(st.dtarr)):
            data_dmdt = rfpipe.search.resample(data_dm, st.dtarr[dtind])

            canddatalist = rfpipe.search.search_thresh(mockstate, data_dmdt,
                                                       segment, dmind, dtind,
                                                       wisdom=wisdom)

            assert len(canddatalist) == st.readints

            features = rfpipe.search.calc_features(canddatalist)


def test_pipeline(mockstate):
    res = rfpipe.pipeline.pipeline_seg(mockstate, 0)
