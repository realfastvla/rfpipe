from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe
import pytest
from astropy import time
import numpy as np


# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'sigma_image1': -999,
            'spw': [0, 1], 'fftmode': 'fftw',
            'searchtype': 'image'},
           {'flaglist': [], 'sigma_image1': -999,
            'spw': [2, 3], 'dmarr': [0, 100], 'dtarr': [1, 2],
            'fftmode': 'fftw', 'searchtype': 'image'}]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    t0 = time.Time.now().mjd
    meta = rfpipe.metadata.mock_metadata(t0, t0+0.1/(24*3600), 27, 4, 32*4, 4,
                                         10e3, datasource='sim', antconfig='D')
    return rfpipe.state.State(inmeta=meta, inprefs=request.param)


def mockdata(mockstate, segment):
    data = rfpipe.source.read_segment(mockstate, segment)
    return rfpipe.source.data_prep(mockstate, segment, data)


def test_dataprep(mockstate):
    data_prep = mockdata(mockstate, 0)
    assert data_prep.shape == mockstate.datashape


def test_search(mockstate):
    wisdom = rfpipe.search.set_wisdom(mockstate.npixx, mockstate.npixy)

    times = []
    for segment in range(mockstate.nsegment):
        data_prep = mockdata(mockstate, segment)
        candcollection = rfpipe.search.dedisperse_search_fftw(mockstate,
                                                              segment,
                                                              data_prep,
                                                              wisdom=wisdom)

        assert len(candcollection) == sum([len(mockstate.get_search_ints(segment, dmind, dtind))
                                           for dmind in range(len(mockstate.dmarr))
                                           for dtind in range(len(mockstate.dtarr))])

    times = candcollection.candmjd/(24*3600)
    times = np.sort(times - times.min())
    deltat = np.array([times[i+1] - times[i] for i in range(len(times)-1)])
    assert (deltat < mockstate.inttime).all()

    integs0_0 = []
    integs1_0 = []
    integs0_1 = []
    for i in range(len(candcollection)):
        (seg, integ, dmind, dtind, beamnum) = candcollection.locs[i]
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
