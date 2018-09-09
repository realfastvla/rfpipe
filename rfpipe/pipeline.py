from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

from rfpipe import source, search, candidates, state, metadata

import logging
logger = logging.getLogger(__name__)
vys_timeout_default = 10


def pipeline_scan(st, segments=None, cfile=None,
                  vys_timeout=vys_timeout_default):
    """ Given rfpipe state run search pipline on all segments in a scan.
        state/preference has fftmode that will determine functions used here.
    """

    # initialize with empty cc
    candcollection = candidates.CandCollection(prefs=st.prefs,
                                               metadata=st.metadata)

    if not isinstance(segments, list):
        segments = list(range(st.nsegment))

    for segment in segments:
        candcollection += pipeline_seg(st, segment, cfile=cfile,
                                       vys_timeout=vys_timeout)

    return candcollection


def pipeline_seg(st, segment, cfile=None, vys_timeout=vys_timeout_default):
    """ Submit pipeline processing of a single segment on a single node.
    state/preference has fftmode that will determine functions used here.
    """

    data = source.read_segment(st, segment, timeout=vys_timeout, cfile=cfile)
    data_prep = source.data_prep(st, segment, data)

    if st.fftmode == "fftw":
        wisdom = search.set_wisdom(st.npixx, st.npixy)
        candcollection = search.dedisperse_search_fftw(st, segment, data_prep,
                                                       wisdom=wisdom)
    elif st.fftmode == "cuda":
        candcollection = search.dedisperse_search_cuda(st, segment, data_prep)

    candidates.save_cands(st, candcollection=candcollection)

    return candcollection


def pipeline_sdm(sdm, intent='TARGET', inprefs=None, preffile=None):
    """ Get scans from SDM and run search.
    intent can be partial match to any of scan intents.
    """

    scans = list(metadata.getsdm(sdm).scans())
    intents = [scan.intents for scan in scans]
    logger.info("Found {0} scans of intents {1} in {2}"
                .format(len(scans), intents, sdm))

    scannums = [int(scan.idx) for scan in scans
                if scan.bdf.exists and any([intent in scint for scint in scan.intents])]
    logger.info("Searching {0} of {1} scans".format(len(scannums), len(scans)))

    for scannum in scannums:
        st = state.State(sdmfile=sdm, sdmscan=scannum, inprefs=inprefs,
                         preffile=preffile)
        cc = pipeline_scan(st)
