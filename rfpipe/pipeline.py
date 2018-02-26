from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

from rfpipe import source, search, util, candidates

import logging
logger = logging.getLogger(__name__)
vys_timeout_default = 10


def pipeline_scan(st, segments=None, cfile=None,
                  vys_timeout=vys_timeout_default):
    """ Given rfpipe state run search pipline on all segments in a scan.
        state/preference has fftmode that will determine functions used here.
    """

    featurelists = []
    if not isinstance(segments, list):
        segments = list(range(st.nsegment))

    for segment in segments:
        featurelists.append(pipeline_seg(st, segment, cfile=cfile,
                                         vys_timeout=vys_timeout))

    return featurelists  # list of tuples (collection, data)


def pipeline_seg(st, segment, cfile=None, vys_timeout=vys_timeout_default):
    """ Submit pipeline processing of a single segment on a single node.
    state/preference has fftmode that will determine functions used here.
    """

    uvw = util.get_uvw_segment(st, segment)

    data = source.read_segment(st, segment, timeout=vys_timeout, cfile=cfile)
    data_prep = source.data_prep(st, segment, data)

    collections = []
    if st.fftmode == "fftw":
        imgranges = [[(min(st.get_search_ints(segment, dmind, dtind)),
                      max(st.get_search_ints(segment, dmind, dtind)))
                      for dtind in range(len(st.dtarr))]
                     for dmind in range(len(st.dmarr))]

        wisdom = search.set_wisdom(st.npixx, st.npixy)
        collections += search.dedisperse_image_fftw(st, segment, data,
                                                    wisdom=wisdom)

    elif st.fftmode == "cuda":
        canddatalist = search.dedisperse_image_cuda(st, segment, data_prep)
        collections += candidates.calc_features(canddatalist)

    return collections
