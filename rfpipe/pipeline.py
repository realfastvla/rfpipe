from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)
vys_timeout_default = 10


def pipeline_scan(st, segments=None, cfile=None,
                  vys_timeout=vys_timeout_default, devicenum=None):
    """ Given rfpipe state run search pipline on all segments in a scan.
        state/preference has fftmode that will determine functions used here.
    """

    from rfpipe import candidates

    # initialize with empty cc
    candcollection = candidates.CandCollection(prefs=st.prefs,
                                               metadata=st.metadata)

    if not isinstance(segments, list):
        segments = list(range(st.nsegment))

    for segment in segments:
        candcollection += pipeline_seg(st, segment, devicenum=devicenum, cfile=cfile,
                                       vys_timeout=vys_timeout)

    return candcollection


def pipeline_seg(st, segment, cfile=None, vys_timeout=vys_timeout_default, devicenum=None):
    """ Submit pipeline processing of a single segment on a single node.
    state/preference has fftmode that will determine functions used here.
    """

    from rfpipe import source

    data = source.read_segment(st, segment, timeout=vys_timeout, cfile=cfile)
    candcollection = prep_and_search(st, segment, data, devicenum=devicenum)

    return candcollection


def prep_and_search(st, segment, data, devicenum=None, returnsoltime=False):
    """ Bundles prep and search functions to improve performance in distributed.
    devicenum refers to GPU device for search.
    returnsoltime is option for data_prep to return solution time too.
    """

    from rfpipe import source, search, util, reproduce, candidates

    ret = source.data_prep(st, segment, data, returnsoltime=returnsoltime)
    if returnsoltime:
        data, soltime = ret
    else:
        data = ret
        soltime = None

    if st.prefs.fftmode == "cuda":
        candcollection = search.dedisperse_search_cuda(st, segment, data,
                                                       devicenum=devicenum)
    elif st.prefs.fftmode == "fftw":
        candcollection = search.dedisperse_search_fftw(st, segment, data)
    else:
        logger.warning("fftmode {0} not recognized (cuda, fftw allowed)"
                       .format(st.prefs.fftmode))

    # calc other features for cc, plot, save
    if st.prefs.savecanddata or st.prefs.saveplots:
        if st.prefs.searchtype in ['imagek', 'armkimage', 'armk']:
            spec_std, sig_ts, kalman_coeffs = util.kalman_prep(data)  # TODO: should this be redundant with search too?
        else:
            spec_std, sig_ts, kalman_coeffs = None, None, None

        candcollection = reproduce.reproduce_candcollection(candcollection, data,
                                                            spec_std=spec_std,
                                                            sig_ts=sig_ts,
                                                            kalman_coeffs=kalman_coeffs)

    candidates.save_cands(st, candcollection=candcollection)
    candcollection.soltime = soltime
    return candcollection


def pipeline_sdm(sdm, inprefs=None, intent='TARGET', preffile=None):
    """ Get scans from SDM and run search.
    intent can be partial match to any of scan intents.
    """

    from rfpipe import state, metadata

    scans = list(metadata.getsdm(sdm).scans())
    intents = [scan.intents for scan in scans]
    logger.info("Found {0} scans of intents {1} in {2}"
                .format(len(scans), intents, sdm))

    scannums = [int(scan.idx) for scan in scans
                if scan.bdf.exists and any([intent in scint for scint in scan.intents])]
    logger.info("Searching {0} of {1} scans".format(len(scannums), len(scans)))

    ccs = []
    for scannum in scannums:
        st = state.State(sdmfile=sdm, sdmscan=scannum, inprefs=inprefs,
                         preffile=preffile)
        ccs.append(pipeline_scan(st))
