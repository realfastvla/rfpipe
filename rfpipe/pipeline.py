from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

from . import state, source, search, util
import distributed
import numpy as np

vys_timeout_default = 10

def pipeline_scan(st, host='cbe-node-01', cfile=None, vys_timeout=vys_timeout_default):
    """ Given rfpipe state and dask distributed client, run search pipline """

    saved = []

    cl = distributed.Client('{0}:{1}'.format(host, '8786'))

    for segment in range(st.nsegment):
        saved.append(pipeline_seg(st, segment, cl, cfile=cfile, vys_timeout=vys_timeout))

    return saved


def pipeline_vystest(wait, nsegment=1, host='cbe-node-01', preffile=None, cfile=None, **kwargs):
    """ Start one segment vysmaw jobs reading a segment each after time wait
    Uses example realfast scan configuration from files.
    """

    st = state.state_vystest(wait, nsegment=nsegment, preffile=preffile, **kwargs)

    saved = pipeline_scan(st, host=host, cfile=cfile)

    return saved


def pipeline_seg(st, segment, cl, workers=None, cfile=None, vys_timeout=vys_timeout_default):
    """ Run segment pipelne with cl.submit calls """

# alternative formulation with dask.delayed:
#    from dask import delayed
#    saved = delayed(search.save_cands)(st, cands, segment)
#    return cl.persist(saved)


#    features = []
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
            ims_thresh = cl.submit(search.resample_image, data_dm, st.dtarr[dtind], uvw, st.freq, st.npixx, st.npixy, st.uvres, st.prefs.sigma_image1, wisdom, pure=True, workers=workers, allow_other_workers=allow_other_workers)

#            candplot = cl.submit(search.candplot, ims_thresh, data_dm)
            search_coords = OrderedDict(segment = segment, dmind = dmind, dtind = dtind, beamnum = 0)

            feature = cl.submit(search.calc_features, st, ims_thresh, search_coords, pure=True, workers=workers, allow_other_workers=allow_other_workers)
#            features.append(feature)

#    logger.info('Saving candidates...')
#    cands = cl.submit(search.collect_cands, features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
            saved = cl.submit(search.save_cands, st, cands, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers)

    return saved
