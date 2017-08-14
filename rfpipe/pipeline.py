from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import distributed
from dask import delayed, compute

from rfpipe import state, source, search, util

import logging
logger = logging.getLogger(__name__)
vys_timeout_default = 10


def pipeline_scan_distributed(st, segments=None, host='cbe-node-01',
                              cfile=None, vys_timeout=vys_timeout_default):
    """ Given rfpipe state and dask distributed client, run search pipline """

    saved = []
    if not isinstance(segments, list):
        segments = range(st.nsegment)

    for segment in segments:
        saved.append(pipeline_seg(st, segment, host=host, cfile=cfile,
                     vys_timeout=vys_timeout))

    return saved


def pipeline_vystest(wait, host='cbe-node-01', preffile=None,
                     cfile=None, **kwargs):
    """ Start one segment vysmaw jobs reading a segment each after time wait
    Uses example realfast scan configuration from files.
    """

    st = state.state_vystest(wait, preffile=preffile,
                             **kwargs)

    saved = pipeline_scan_distributed(st, host=host, cfile=cfile)

    return saved


def pipeline_seg(st, segment, host='cbe-node-01', cfile=None,
                 vys_timeout=vys_timeout_default):
    """ Build DAG from delayed objects and execute at end with preferred
        scheduler
    """

    logger.info('Building dask for observation {0}.'.format(st.fileroot))
    cl = distributed.Client('{0}:{1}'.format(host, '8786'))

    # plan fft
    wisdom = cl.submit(search.set_wisdom, st.npixx, st.npixy, pure=True)

    data = cl.submit(source.read_segment, st, segment, timeout=vys_timeout,
                     cfile=cfile, pure=True)
    data_prep = cl.submit(source.data_prep, st, data, pure=True)

    # **TODO: need to add condition on data_prep being nonzero
    saved = []
    for dmind in range(len(st.dmarr)):
        delay = cl.submit(util.calc_delay, st.freq, st.freq.max(),
                          st.dmarr[dmind], st.inttime, pure=True)
        data_dm = cl.submit(search.dedisperse, data_prep, delay, pure=True)

        for dtind in range(len(st.dtarr)):
            data_dmdt = cl.submit(search.resample, data_dm, st.dtarr[dtind],
                                  pure=True)
            canddatalist = cl.submit(search.search_thresh, st, data_dmdt,
                                     segment, dmind, dtind, wisdom=wisdom)

            candidates = cl.submit(search.calc_features, canddatalist,
                                   pure=True)
            saved.append(cl.submit(search.save_cands, st, candidates,
                                   canddatalist))

    return saved


def pipeline_seg_delayed(st, segment, cl=None, cfile=None,
                         vys_timeout=vys_timeout_default):
    """ Build DAG from delayed objects and execute at end with preferred
        scheduler
    """

    # ** how to use distributed arguments?
    # (workers=workers, allow_other_workers=allow_other_workers)

    logger.info('Building dask for observation {0}.'.format(st.fileroot))

    # plan fft
    wisdom = delayed(search.set_wisdom, pure=True)(st.npixx, st.npixy)

    data = delayed(source.read_segment, pure=True)(st, segment,
                                                   timeout=vys_timeout,
                                                   cfile=cfile)
    data_prep = delayed(source.data_prep, pure=True)(st, data)

    # **TODO: need to add condition on data_prep being nonzero
    saved = []
    for dmind in range(len(st.dmarr)):
        delay = delayed(util.calc_delay, pure=True)(st.freq, st.freq.max(),
                                                    st.dmarr[dmind],
                                                    st.inttime)
        data_dm = delayed(search.dedisperse, pure=True)(data_prep, delay)

        for dtind in range(len(st.dtarr)):
            data_dmdt = delayed(search.resample, pure=True)(data_dm,
                                                            st.dtarr[dtind])
            canddatalist = delayed(search.search_thresh,
                                   pure=True)(st, data_dmdt, segment, dmind,
                                              dtind, wisdom=wisdom)

            candidates = delayed(search.calc_features, pure=True)(canddatalist)
            saved.append(delayed(search.save_cands, pure=True)(st, candidates,
                                                               canddatalist))

    if cl:
        # if using distributed client, return collection with futures
        return cl.persist(saved)
    else:
        # otherwise return the delayed objects
        return compute(*saved, num_workers=st.prefs.nthread)
