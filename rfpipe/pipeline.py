import logging
logger = logging.getLogger(__name__)

from . import state, source, search, util
import distributed
from functools import partial

##
# testing dask distributed and numba
##


def pipeline_seg(st, segment, cl, workers=None):
    """ Run segment pipelne with cl.submit calls """

    features = []
    allow_other_workers = workers != None

    # plan fft
    logger.info('Planning FFT...')
    wisdom = cl.submit(search.set_wisdom, st.npixx, st.npixy, pure=True, workers=workers, allow_other_workers=allow_other_workers)

    logger.info('reading data...')
    data_prep = cl.submit(source.dataprep, st, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers)
#    uvw = cl.submit(source.calc_uvw, st, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers) # now built into state
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
            ims_thresh = cl.submit(search.resample_image, data_dm, st.dtarr[dtind], uvw, st.freq, st.npixx, st.npixy, st.uvres, st.parameters.sigma_image1, wisdom, pure=True, workers=workers, allow_other_workers=allow_other_workers)

#            candplot = cl.submit(search.candplot, ims_thresh, data_dm)
            feature = cl.submit(search.calc_features, ims_thresh, dmind, st.dtarr[dtind], dtind, segment, st.features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
            features.append(feature)

    cands = cl.submit(search.collect_cands, features, pure=True, workers=workers, allow_other_workers=allow_other_workers)
    saved = cl.submit(search.save_cands, st, cands, segment, pure=True, workers=workers, allow_other_workers=allow_other_workers)
    return saved


def pipeline_scan(st, host='nmpost-master'):
    """ Given rfpipe state and dask distributed client, run search pipline """

    cl = distributed.Client('{0}:{1}'.format(host, '8786'))

    logger.debug('submitting segments')

    saved = []
    for segment in range(st.nsegments):
        saved.append(pipeline_seg(st, segment, cl))

    return saved
