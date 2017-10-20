import rfpipe
import pytest
import os.path
import os

_install_dir = os.path.abspath(os.path.dirname(__file__))


def test_sdmmeta():
    sdmfile = os.path.join(_install_dir, 'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')
    meta = rfpipe.metadata.sdm_metadata(sdmfile, 7)
    assert isinstance(meta, dict)


def test_sdm():
    sdmfile = os.path.join(_install_dir,
                           'data/16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1')
    inprefs = {'gainfile': os.path.join(_install_dir,
                                        '16A-459_TEST_1hr_000.57633.66130137732.scan7.cut1.GN')}
    state = rfpipe.state.State(sdmfile=sdmfile, sdmscan=7, inprefs=inprefs)
    res = rfpipe.pipeline.pipeline_seg(state, 0)
