import rfpipe
import pytest
from astropy import time
import numpy as np

# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'npix_max': 32, 'sigma_image1': 0, 'spw': [0, 1]},
           {'flaglist': [], 'npix_max': 32, 'sigma_image1': 0, 'spw': [2, 3],
           'maxdm': 100, 'dtarr': [1, 2]}
           ]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.3/(24*3600), 27, 4, 2, 10e3,
                                         datasource='sim')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


def mockdata(mockstate, segment):
    data = rfpipe.source.read_segment(mockstate, segment)
    return rfpipe.source.data_prep(mockstate, data)


def test_dataprep(mockstate):
    data_prep = mockdata(mockstate, 0)
    assert data_prep.shape == mockstate.datashape


def test_search(mockstate):

    wisdom = rfpipe.search.set_wisdom(mockstate.npixx, mockstate.npixy)

    features = []
    for segment in range(mockstate.nsegment):
        data_prep = mockdata(mockstate, segment)

        for dmind in range(len(mockstate.dmarr)):
            delay = rfpipe.util.calc_delay(mockstate.freq,
                                           mockstate.freq.max(),
                                           mockstate.dmarr[dmind],
                                           mockstate.inttime)
            data_dm = rfpipe.search.dedisperse(data_prep, delay)

            for dtind in range(len(mockstate.dtarr)):
                data_dmdt = rfpipe.search.resample(data_dm,
                                                   mockstate.dtarr[dtind])

                canddatalist = rfpipe.search.search_thresh(mockstate,
                                                           data_dmdt,
                                                           segment, dmind,
                                                           dtind,
                                                           wisdom=wisdom)

                assert len(canddatalist) == mockstate.readints//mockstate.dtarr[dtind]

                features.append(rfpipe.search.calc_features(canddatalist))

    alltimes = np.linspace(mockstate.meta.starttime_mjd,
                           mockstate.meta.stoptime_mjd, mockstate.nints)
    integs0 = []
    integs1 = []
    for feature in features:
        for (seg, integ, dmind, dtind, beamnum) in feature:
            if dtind == 0:
                integs0.append(integ)
            elif dtind == 1:
                integs1.append(integ)

    assert len(alltimes) == len(integs0)
    assert len(alltimes)//2 == len(integs1)
