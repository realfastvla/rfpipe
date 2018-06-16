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
                   'fftmode': 'fftw'}
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
    cc = rfpipe.search.prep_and_search(st, segment, data)


def test_dm_singlemulti(st, data):
    dm = 100
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)
    data1 = rfpipe.search.dedisperse(data, delay, parallel=False)
    data2 = rfpipe.search.dedisperse(data, delay, parallel=True)
    data3 = rfpipe.search.dedisperse(data, delay, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)


def test_resample_singlemulti(st, data):
    dt = 2
    data1 = rfpipe.search.resample(data, dt, parallel=False)
    data2 = rfpipe.search.resample(data, dt, parallel=True)
    data3 = rfpipe.search.resample(data, dt, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)


def test_dmresample_single(st, data):
    dm = 100
    dt = 2
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperse(data, delay, parallel=False)
    data2 = rfpipe.search.resample(data1, dt, parallel=False)
    data3 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=False)
    assert np.allclose(data3, data2)


def test_dmresample_multi1(st, data):
    dm = 100
    dt = 1
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperse(data, delay, parallel=True)
    data2 = rfpipe.search.resample(data1, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=True)
    assert np.allclose(data3, data2)


def test_dmresample_multi2(st, data):
    dm = 100
    dt = 2
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperse(data, delay, parallel=True)
    data2 = rfpipe.search.resample(data1, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=True)
    assert np.allclose(data3, data2)


def test_dmresample_singlemulti1(st, data):
    dm = 100
    dt = 1
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=False)
    data2 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)


def test_dmresample_singlemulti2(st, data):
    dm = 100
    dt = 2
    delay = rfpipe.util.calc_delay(st.freq, st.freq.max(), dm,
                                   st.inttime)

    data1 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=False)
    data2 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=True)
    data3 = rfpipe.search.dedisperseresample(data, delay, dt, parallel=False)

    assert np.allclose(data1, data2)
    assert np.allclose(data3, data2)
