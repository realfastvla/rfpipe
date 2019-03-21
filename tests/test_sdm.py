from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import rfpipe
import pytest
import os.path
import numpy as np

_install_dir = os.path.abspath(os.path.dirname(__file__))

# simulate no flag, transient/no flag, transient/flag
inprefs = [{'simulated_transient': [(0, 0, 0, 5e-3, 1., 0., 0.)], 'timesub': None,
            'flaglist': [], 'maxdm': 0, 'dtarr': [1], 'npix_max': 512},
           {'simulated_transient': [(0, 0, 0, 5e-3, 1., 0., 0.)], 'timesub': None,
            'flaglist': [], 'maxdm': 0, 'dtarr': [1], 'npix_max': 512, 'savesols': True,
            'gainfile': os.path.join(_install_dir,
                                     'data/16A-459_TEST_1hr_000.57633.66130137732.GN')}]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    sdmfile = os.path.join(_install_dir,
                           'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')

    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    return rfpipe.state.State(sdmfile=sdmfile, sdmscan=7, preffile=preffile,
                              inprefs=request.param)


def test_cal_and_uncal(mockstate):
    segment = 0
    data = rfpipe.source.data_prep(mockstate, segment, rfpipe.source.read_segment(mockstate, segment))
    assert np.any(data)

    datacal = rfpipe.calibration.apply_telcal(mockstate, data, sign=1)
    datauncal = rfpipe.calibration.apply_telcal(mockstate, datacal, sign=-1)
    assert np.allclose(datauncal, data)


def test_simulated_source(mockstate):
    segment = 0
    data = rfpipe.source.read_segment(mockstate, segment)
    dataprep = rfpipe.source.data_prep(mockstate, segment, data)
    assert dataprep.mean() > 0.9

    im = rfpipe.search.grid_image(dataprep,
                                  rfpipe.util.get_uvw_segment(mockstate,
                                                              segment),
                                  mockstate.npixx, mockstate.npixy,
                                  mockstate.uvres, mockstate.fftmode,
                                  1, integrations=0)
    assert im[0].max()/im[0].std() > 10


def test_simulated_source_zeroshift(mockstate):
    segment = 0
    mockstate.metadata.phasecenters = [(mockstate.metadata.starttime_mjd, mockstate.metadata.starttime_mjd+1, 0., 0.),]
    data = rfpipe.source.read_segment(mockstate, segment)
    dataprep = rfpipe.source.data_prep(mockstate, segment, data)
    assert dataprep.mean() > 0.9

    im = rfpipe.search.grid_image(dataprep,
                                  rfpipe.util.get_uvw_segment(mockstate,
                                                              segment),
                                  mockstate.npixx, mockstate.npixy,
                                  mockstate.uvres, mockstate.fftmode,
                                  1, integrations=0)
    assert im[0].max()/im[0].std() > 10
