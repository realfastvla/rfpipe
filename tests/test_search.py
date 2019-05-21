from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe
import pytest
from astropy import time
import numpy as np


@pytest.fixture(scope="module")
def st():
        inprefs = {'flaglist': [], 'npix_max': 128, 'uvres': 500, 'nthread': 1,
                   'fftmode': 'fftw', 'searchtype': 'imagek'}
        t0 = time.Time.now().mjd
        meta = rfpipe.metadata.mock_metadata(t0, t0+0.05/(24*3600), 10, 4, 32*4,
                                             2, 5e3, datasource='sim', antconfig='D')
        return rfpipe.state.State(inmeta=meta, inprefs=inprefs)


@pytest.fixture(scope="module")
def data(st):
        segment = 0
        return rfpipe.source.read_segment(st, segment)


def test_prepsearch(st, data):
    segment = 0
    data[:, :, 10:12] = 0j  # test zeroed channels
    cc = rfpipe.pipeline.prep_and_search(st, segment, data)
    assert len(cc) == 0


def test_excess(st, data):
    segment = 0
    st.prefs.max_candfrac = 0.01
    data += 1.
    cc = rfpipe.pipeline.prep_and_search(st, segment, data)
    st.prefs.max_candfrac = 0.2
    assert len(cc) == 0


def test_nosearch(st, data):
    segment = 0
    st.prefs.searchtype = None
    cc = rfpipe.pipeline.prep_and_search(st, segment, data)
    st.prefs.searchtype = 'imagek'
    assert len(cc) == 0


def test_dm_singlemulti(st, data):
    dm = 100
    datap = rfpipe.source.data_prep(st, 0, data)
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)
    data1 = rfpipe.search.dedisperse(datap, delay, parallel=False)
    data2 = rfpipe.search.dedisperse(datap, delay, parallel=True)
    data3 = rfpipe.search.dedisperse(datap, delay, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)


def test_resample_singlemulti(st, data):
    dt = 2
    datap = rfpipe.source.data_prep(st, 0, data)
    data1 = rfpipe.search.resample(datap, dt, parallel=False)
    data2 = rfpipe.search.resample(datap, dt, parallel=True)
    data3 = rfpipe.search.resample(datap, dt, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)


def test_dmresample_single(st, data):
    dm = 100
    dt = 2
    datap = rfpipe.source.data_prep(st, 0, data)
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperse(datap, delay, parallel=False)
    data2 = rfpipe.search.resample(data1, dt, parallel=False)
    data3 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=False,
                                             resamplefirst=False)
    assert np.allclose(data3, data2)


def test_dmresample_multi1(st, data):
    dm = 100
    dt = 1
    datap = rfpipe.source.data_prep(st, 0, data)
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperse(datap, delay, parallel=True)
    data2 = rfpipe.search.resample(data1, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=True,
                                             resamplefirst=False)
    assert np.allclose(data3, data2)


def test_dmresample_multi2(st, data):
    dm = 100
    dt = 2
    datap = rfpipe.source.data_prep(st, 0, data)
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperse(datap, delay, parallel=True)
    data2 = rfpipe.search.resample(data1, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=True,
                                             resamplefirst=False)
    assert np.allclose(data3, data2)


def test_dmresample_singlemulti1(st, data):
    dm = 100
    dt = 1
    datap = rfpipe.source.data_prep(st, 0, data)
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=False)
    data2 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)


def test_dmresample_singlemulti2(st, data):
    dm = 100
    dt = 2
    datap = rfpipe.source.data_prep(st, 0, data)
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=False)
    data2 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(datap, delay, dt, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)
