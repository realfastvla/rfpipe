from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

from rfpipe import source, search, util, candidates

import logging
logger = logging.getLogger(__name__)
vys_timeout_default = 10


def pipeline_scan(st, segments=None, cfile=None, vys_timeout=vys_timeout_default):
    """ Given rfpipe state run search pipline on all segments in a scan. """

    featurelists = []
    if not isinstance(segments, list):
        segments = range(st.nsegment)

    for segment in segments:
        featurelists.append(pipeline_seg(st, segment, cfile=cfile,
                                         vys_timeout=vys_timeout))

    return featurelists  # list of tuples (collection, data)


def pipeline_seg(st, segment, cfile=None, vys_timeout=vys_timeout_default):
    """ Submit pipeline processing of a single segment on a single node.
    """

    # plan fft
    wisdom = search.set_wisdom(st.npixx, st.npixy)

    data = source.read_segment(st, segment, timeout=vys_timeout, cfile=cfile)
    data_prep = source.data_prep(st, data)

    collections = []
    for dmind in range(len(st.dmarr)):
        delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind],
                                st.inttime)
#        data_dm = search.dedisperse(data_prep, delay)

        for dtind in range(len(st.dtarr)):
            data_dmdt = search.dedisperseresample(data_prep, delay,
                                                  st.dtarr[dtind])
            canddatalist = search.search_thresh(st, data_dmdt, segment, dmind,
                                                dtind, wisdom=wisdom)

            collection = candidates.calc_features(canddatalist)
            collections.append(collection)

    return collections
