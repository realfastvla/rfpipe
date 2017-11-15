import rfpipe
import pytest
import os.path
from astropy import time
import numpy as np

_install_dir = os.path.abspath(os.path.dirname(__file__))

# simulate no flag, transient/no flag, transient/flag
inprefs = [{'flaglist': [], 'maxdm': 0, 'dtarr': [1], 'npix_max': 512},
           {'simulated_transient': [(0, 0, 0, 5e-3, 1., 0.001, 0.001)],
            'maxdm': 0, 'dtarr': [1], 'npix_max': 512}]


@pytest.fixture(scope="module", params=inprefs)
def mockstate(request):
    sdmfile = os.path.join(_install_dir,
                           'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')

    preffile = os.path.join(_install_dir, 'data/realfast.yml')
    return rfpipe.state.State(sdmfile=sdmfile, sdmscan=7, preffile=preffile,
                              inprefs=request.param)


def test_cal(mockstate):
    segment = 0
    data = rfpipe.source.read_segment(mockstate, segment)
    datacal = rfpipe.calibration.apply_telcal(mockstate, data, sign=1)
    datauncal = rfpipe.calibration.apply_telcal(mockstate, data, sign=-1)
    assert np.all(datauncal == data)
