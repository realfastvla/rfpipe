from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe
import pytest
from astropy import time
import numpy as np


# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'npix_max': 32, 'sigma_image1': -999,
            'spw': [0, 1], 'uvres': 50000},
           {'flaglist': [], 'npix_max': 32, 'sigma_image1': -999,
            'spw': [2, 3], 'dmarr': [0, 100], 'dtarr': [1, 2], 'uvres': 50000}]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+1./(24*3600), 27, 4, 32*4, 4, 10e3,
                                         datasource='sim')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


def mockdata(mockstate, segment):
    data = rfpipe.source.read_segment(mockstate, segment)
    return rfpipe.source.data_prep(mockstate, segment, data)


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

                candcollection = rfpipe.search.search_thresh_fftw(mockstate,
                                                                  segment,
                                                                  data_dmdt,
                                                                  dmind,
                                                                  dtind,
                                                                  wisdom=wisdom)

                assert len(candcollection) == len(mockstate.get_search_ints(segment, dmind, dtind))
                candcollections.append(candcollection)

    times = np.concatenate([cc.candmjd/(24*3600) for cc in candcollections])
    times = np.sort(times - times.min())
    deltat = np.array([times[i+1] - times[i] for i in range(len(times)-1)])
    assert (deltat < mockstate.inttime).all()

    integs0_0 = []
    integs1_0 = []
    integs0_1 = []
    for candcollection in candcollections:
        for i in range(len(candcollection.array)):
            (seg, integ, dmind, dtind, beamnum) = candcollection.array[[str(ff) for ff in (mockstate.search_dimensions)]][i]

            if dtind == 0 and dmind == 0:
                integs0_0.append(integ)
            elif dtind == 1 and dmind == 0:
                integs1_0.append(integ)
            elif dtind == 0 and dmind == 1:
                integs0_1.append(integ)

    assert mockstate.searchints == len(integs0_0)
    if 2 in mockstate.dtarr:
        rr = 0
        for segment in range(mockstate.nsegment):
            integs = mockstate.get_search_ints(segment, 0, 1)
            rr += max(integs) - min(integs) + 1
        assert rr == len(integs1_0)
        rr = 0
        for segment in range(mockstate.nsegment):
            integs = mockstate.get_search_ints(segment, 1, 0)
            rr += max(integs) - min(integs) + 1
        assert rr == len(integs0_1)
