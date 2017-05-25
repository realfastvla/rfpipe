from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

from . import state, source, search, util, metadata
import distributed
from dask import delayed
from functools import partial
from astropy import time
from time import sleep
import numpy as np


def pipeline_vys(wait, nsegment=1, nant=3, nspw=1, nchan=64, npol=1, inttime_micros=1e5, host='cbe-node-01', preffile=None, cfile=None):
    """ Start nsegment vysmaw jobs reading a segment each after time wait
    """

    assert nsegment > 0

    cl = distributed.Client('{0}:{1}'.format(host, '8786'))
    data_delayed = [delayed(source.read_vys_seg)(st, seg, cfile=cfile) for seg in range(nsegment)]

    data_fut = []

    while len(data_delayed):
        if len([df for df in data_fut if not df.done()]) < 2:
            dd = data_delayed.pop()
            data_prep = cl.compute(dd)
            data_fut.append(data_prep)
        else:
            sleep(0.1)

    return data_fut


def pipeline_vys2(wait, nsegment=1, nant=3, nspw=1, nchan=64, npol=1, inttime_micros=1e5, host='cbe-node-01', preffile=None, cfile=None, workers=None):
    """ Start nsegment vysmaw jobs reading a segment each after time wait
    """

    assert nsegment > 0
    allow_other_workers = workers != None

    cl = distributed.Client('{0}:{1}'.format(host, '8786'))
    t0 = time.Time.now() + time.TimeDelta(wait, format='sec')

    meta = cl.submit(metadata.mock_metadata, t0.mjd, nant, nspw, nchan, npol, inttime_micros)
    st = cl.submit(state.State, preffile=preffile, inmeta=meta, inprefs={'nsegment':nsegment})
    data_delayed = [delayed(source.read_vys_seg)(st, seg, cfile=cfile) for seg in range(nsegment)]

    features = []
    saved = []
    data_fut = []

    while len(data_delayed):
        if len([df for df in data_fut if not df.done()]) < 2:
            dd = data_delayed.pop()
            data_prep = cl.compute(dd)
            data_fut.append(data_prep)

            wisdom = cl.submit(search.set_wisdom, st.npixx, st.npixy, pure=True, workers=workers, allow_other_workers=allow_other_workers)

            for dmind in range(len(st.dmarr)):
                delay = cl.submit(util.calc_delay, st.freq, st.freq[-1], st.dmarr[dmind], st.metadata.inttime, pure=True, workers=workers, allow_other_workers=allow_other_workers)
                data_dm = cl.submit(search.dedisperse, data_prep, delay, pure=True, workers=workers, allow_other_workers=allow_other_workers)

                for dtind in range(len(st.dtarr)):
                    uvw = st.get_uvw_segment(segment)
                    ims_thresh = cl.submit(search.resample_image, data_dm, st.dtarr[dtind], uvw, st.freq, st.npixx, st.npixy, st.uvres, st.prefs.sigma_image1, wisdom, pure=True, workers=workers, allow_other_workers=allow_other_workers)

                    feature = cl.submit(search.calc_features, ims_thresh, dmind, st.dtarr[dtind], dtind, segment, st.features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
                    features.append(feature)

            cands = cl.submit(search.collect_cands, features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
            saved.append(cl.submit(search.save_cands, st, cands, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers))

        else:
            sleep(0.1)

    return saved


def pipeline_seg(st, segment, cl, workers=None):
    """ Run segment pipelne with cl.submit calls """

    features = []
    allow_other_workers = workers != None

    # plan fft
    logger.info('Planning FFT...')
    wisdom = cl.submit(search.set_wisdom, st.npixx, st.npixy, pure=True, workers=workers, allow_other_workers=allow_other_workers)

    logger.info('reading data...')
    data_prep = cl.submit(source.dataprep, st, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers)
#    cl.replicate([data_prep, uvw, wisdom])  # spread data around to search faster

    for dmind in range(len(st.dmarr)):
        delay = cl.submit(util.calc_delay, st.freq, st.freq[-1], st.dmarr[dmind], st.metadata.inttime, pure=True, workers=workers, allow_other_workers=allow_other_workers)
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
            feature = cl.submit(search.calc_features, ims_thresh, dmind, st.dtarr[dtind], dtind, segment, st.features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
            features.append(feature)

    cands = cl.submit(search.collect_cands, features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
    saved = cl.submit(search.save_cands, st, cands, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers)
    return saved


def pipeline_seg_delayed(st, segment, cl, workers=None):
    """ Run segment pipelne with cl.submit calls """

    from dask import delayed

    features = []
    allow_other_workers = workers != None

    # plan fft
    logger.info('Planning FFT...')
    wisdom = delayed(search.set_wisdom)(st.npixx, st.npixy)

    logger.info('reading data...')
    data_prep = delayed(source.dataprep)(st, segment)
#    cl.replicate([data_prep, uvw, wisdom])  # spread data around to search faster

    for dmind in range(len(st.dmarr)):
        delay = delayed(util.calc_delay)(st.freq, st.freq[-1], st.dmarr[dmind], st.metadata.inttime)
        data_dm = delayed(search.dedisperse)(data_prep, delay)

        for dtind in range(len(st.dtarr)):
            # schedule stages separately
#            data_resampled = cl.submit(search.resample, data_dm, st['dtarr'][dtind])
#            grids = cl.submit(search.grid_visibilities, data_resampled, uvw, st['freq'], st['npixx'], st['npixy'], st['uvres'])
#            images = cl.submit(search.image_fftw, grids, wisdom=wisdom)
#            ims_thresh = cl.submit(search.threshold_images, images, st['sigma_image1'])
            # schedule them as single call
            uvw = st.get_uvw_segment(segment)
            ims_thresh = delayed(search.resample_image)(data_dm, st.dtarr[dtind], uvw, st.freq, st.npixx, st.npixy, st.uvres, st.prefs.sigma_image1, wisdom)

#            candplot = cl.submit(search.candplot, ims_thresh, data_dm)
            feature = delayed(search.calc_features)(ims_thresh, dmind, st.dtarr[dtind], dtind, segment, st.features)
            features.append(feature)

    cands = delayed(search.collect_cands)(features)
    saved = delayed(search.save_cands)(st, cands, segment)
    return cl.persist(saved)


def pipeline_scan(st, host='nmpost-master'):
    """ Given rfpipe state and dask distributed client, run search pipline """

    cl = distributed.Client('{0}:{1}'.format(host, '8786'))

    logger.debug('submitting segments')

    saved = []
    for segment in range(st.nsegment):
        saved.append(pipeline_seg_delayed(st, segment, cl))

    return saved
