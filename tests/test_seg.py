import rfpipe
import pytest
from astropy import time
import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'npix_max': 32, 'sigma_image1': -999,
            'spw': [0, 1], 'uvres': 50000},
           {'flaglist': [], 'npix_max': 32, 'sigma_image1': -999,
            'spw': [2, 3], 'dmarr': [0, 100], 'dtarr': [1, 2], 'uvres': 50000}]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+1./(24*3600), 27, 4, 2, 10e3,
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

    candcollections = []
    times = []
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

                candcollection = rfpipe.candidates.calc_features(canddatalist)
                candcollections.append(candcollection)
                times += [canddata.time_top for canddata in canddatalist]
                assert len(canddatalist) == (mockstate.readints-mockstate.dmshifts[dmind])//mockstate.dtarr[dtind]

    times = np.sort(times)
    deltat = np.array([times[i+1] - times[i] for i in range(len(times)-1)])
    assert all(deltat < mockstate.inttime)

    integs0_0 = []
    integs1_0 = []
    integs0_1 = []
    for candcollection in candcollections:
        for i in range(len(candcollection.array)):
            (seg, integ, dmind, dtind, beamnum) = cand[i,:5]

            if dtind == 0 and dmind == 0:
                integs0_0.append(integ)
            elif dtind == 1 and dmind == 0:
                integs1_0.append(integ)
            elif dtind == 0 and dmind == 1:
                integs0_1.append(integ)

    assert mockstate.searchints == len(integs0_0)
    if 2 in mockstate.dtarr:
        assert mockstate.nsegment*(mockstate.readints//2) == len(integs1_0)
        assert mockstate.nsegment*(mockstate.readints-mockstate.dmshifts[1]) == len(integs0_1)
