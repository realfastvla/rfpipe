from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

import numpy as np
import distributed
from collections import OrderedDict
from dask import delayed, compute

from rfpipe import state, source, search, util


vys_timeout_default = 10


def pipeline_scan_distributed(st, host='cbe-node-01', cfile=None, vys_timeout=vys_timeout_default):
    """ Given rfpipe state and dask distributed client, run search pipline """

    saved = []

    cl = distributed.Client('{0}:{1}'.format(host, '8786'))

    for segment in range(st.nsegment):
        saved.append(pipeline_seg(st, segment, cl=cl, cfile=cfile, vys_timeout=vys_timeout))

    return saved


def pipeline_vystest(wait, nsegment=1, host='cbe-node-01', preffile=None, cfile=None, **kwargs):
    """ Start one segment vysmaw jobs reading a segment each after time wait
    Uses example realfast scan configuration from files.
    """

    st = state.state_vystest(wait, nsegment=nsegment, preffile=preffile, **kwargs)

    saved = pipeline_scan(st, host=host, cfile=cfile)

    return saved


def pipeline_seg(st, segment, cl=None, cfile=None, vys_timeout=vys_timeout_default):
    """ Build DAG from delayed objects and execute at end with preferred scheduler """

    # ** how to use distributed arguments? (workers=workers, allow_other_workers=allow_other_workers)

    logger.info('Building dask for observation {0}.'.format(st.fileroot))

    # plan fft
    wisdom = delayed(search.set_wisdom, pure=True)(st.npixx, st.npixy)

    data = delayed(source.read_segment, pure=True)(st, segment, timeout=vys_timeout, cfile=cfile)
    data_prep = delayed(source.data_prep, pure=True)(st, data)

    # **TODO: need to add condition on data_prep being nonzero
    saved = []
    for dmind in range(len(st.dmarr)):
        delay = delayed(util.calc_delay, pure=True)(st.freq, st.freq.max(), st.dmarr[dmind], st.metadata.inttime)
        data_dm = delayed(search.dedisperse, pure=True)(data_prep, delay)

        for dtind in range(len(st.dtarr)):
            # ** could get_uvw_segment be distributed if it was a staticmethod?
            uvw = st.get_uvw_segment(segment)
            ims_thresh = delayed(search.resample_image, pure=True)(data_dm, st.dtarr[dtind], uvw, st.npixx, st.npixy, st.uvres, st.prefs.sigma_image1, wisdom)
#            candplot = delayed(search.candplot)(st, ims_thresh, data_dm)

            search_coords = OrderedDict(zip(['segment', 'dmind', 'dtind', 'beamnum'], [segment, dmind, dtind, 0]))
            candidates = delayed(search.calc_features, pure=True)(st, ims_thresh, search_coords)
            saved.append(delayed(search.save_cands, pure=True)(st, candidates, search_coords))

    if cl:
        # if using distributed client, return futures
        return cl.persist(saved)
    else:
        # otherwise return the delayed objects
        return compute(*saved)


def pipeline_seg_delayed(st, segment, cl, workers=None, cfile=None, vys_timeout=vys_timeout_default):
    """ Build DAG from delayed objects and execute at end with preferred scheduler """

#    saved = delayed(search.save_cands)(st, cands, segment)
#    return cl.persist(saved)

    allow_other_workers = workers != None

    # plan fft
    logger.info('Planning FFT...')
    wisdom = cl.submit(search.set_wisdom, st.npixx, st.npixy, pure=True, workers=workers, allow_other_workers=allow_other_workers)

    logger.info('Reading data...')
    data_prep = cl.submit(source.data_prep, st, segment, timeout=vys_timeout, cfile=cfile, pure=True, workers=workers, allow_other_workers=allow_other_workers)
#    cl.replicate([data_prep, uvw, wisdom])  # spread data around to search faster

    # **TODO: need to add condition on data_prep being nonzero

    logger.info('Iterating search over DM/dt...')
    for dmind in range(len(st.dmarr)):
        delay = cl.submit(util.calc_delay, st.freq, st.freq.max(), st.dmarr[dmind], st.metadata.inttime, pure=True, workers=workers, allow_other_workers=allow_other_workers)
        data_dm = cl.submit(search.dedisperse, data_prep, delay, pure=True, workers=workers, allow_other_workers=allow_other_workers)

        for dtind in range(len(st.dtarr)):
            # schedule stages separately
#            data_resampled = cl.submit(search.resample, data_dm, st['dtarr'][dtind])
#            grids = cl.submit(search.grid_visibilities, data_resampled, uvw, st['freq'], st['npixx'], st['npixy'], st['uvres'])
#            images = cl.submit(search.image_fftw, grids, wisdom=wisdom)
#            ims_thresh = cl.submit(search.threshold_images, images, st['sigma_image1'])
            # schedule them as single call
            uvw = st.get_uvw_segment(segment)
            ims_thresh = cl.submit(search.resample_image, data_dm, st.dtarr[dtind], uvw, st.npixx, st.npixy, st.uvres, st.prefs.sigma_image1, wisdom, pure=True, workers=workers, allow_other_workers=allow_other_workers)

#            candplot = cl.submit(search.candplot, ims_thresh, data_dm)
            search_coords = OrderedDict(segment = segment, dmind = dmind, dtind = dtind, beamnum = 0)

            candidates = cl.submit(search.calc_features, st, ims_thresh, search_coords, pure=True, workers=workers, allow_other_workers=allow_other_workers)
#            features.append(feature)

#    logger.info('Saving candidates...')
#    cands = cl.submit(search.collect_cands, features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
            saved = cl.submit(search.save_cands, st, candidates, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers)

    return saved
