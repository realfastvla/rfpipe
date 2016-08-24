import logging
from . import state, source, search
import distributed
import rtpipe
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger('rfpipe')

# need to design state initialization as an avalanch of decisions set by initial state
# 1) initial, minimal state defines either parameters for later use or fixes final state
# 2) read metadata from observation for given scan (if final value not yet set)
# 3) run functions to set state (sets hashable state)
# 4) may also set convenience attibutes
# 5) run data processing for given segment

# these should be modified to change state based on input
# - nsegments or dmarr + memory_limit + metadata => segmenttimes
# - dmarr or dm_parameters + metadata => dmarr
# - uvoversample + npix_max + metadata => npixx, npixy

# maybe this all goes in state.py outside of class?


# def run(datasource, paramfile, version=2):
#     """ Run whole pipeline """

#     st = state.State(paramfile=paramfile, verison=version)

#     if datatype(datasource) == 'sdm':
#         apply_metadata(st, sdmfile)

#     # learn distributed for this part
#     if 'image' in st.searchtype:
#         search.imaging(st)


def apply_metadata(st, sdmfile):
    """ Run all functions to apply metadata to transform initial state """

    source.parsesdm(sdmfile, st)
    state.set_dmgrid(st)
    state.set_imagegrid(st)
    state.set_segments(st)


##
# testing dask distributed and numba
##


def pipeline_seg(st, segment, ex):
    """ Run segment pipelne with ex.submit calls """

    features = []
    st['segment'] = segment

    # plan fft
    logger.info('Planning FFT...')
    wisdom = ex.submit(search.set_wisdom, st['npixx'], st['npixy'])

    logger.info('reading data...')
    data_prep = ex.submit(source.dataprep, st, segment)
    uvw = ex.submit(source.calc_uvw, st, segment)
    ex.replicate([data_prep, uvw, wisdom])  # spread data around to search faster

    for dmind in range(len(st['dmarr'])):
        delay = ex.submit(search.calc_delay, st['freq'], st['freq'][-1], st['dmarr'][dmind], st['inttime'], pure=True)
        data_dm = ex.submit(search.dedisperse, data_prep, delay, pure=True)

#        ims_thresh = ex.map(search.resample_image, st['dtarr'], data=data_dm, uvw=uvw, freqs=st['freq'], npixx=st['npixx'], npixy=st['npixy'], uvres=st['uvres'], threshold=st['sigma_image1'], wisdom=wisdom)
#        dtind=0
#        feature = ex.map(search.calc_features, ims_thresh, dmind=dmind, dt=st['dtarr'][dtind], dtind=dtind, segment=st['segment'], featurelist=st['features'])
#        features.append(feature)

        for dtind in range(len(st['dtarr'])):
            ims_thresh = ex.submit(search.resample_image, data_dm, st['dtarr'][dtind], uvw, st['freq'], st['npixx'], st['npixy'], st['uvres'], st['sigma_image1'], wisdom)            # resample and search with one function
#            candplot = ex.submit(search.candplot, ims_thresh, data_dm)
            feature = ex.submit(search.calc_features, ims_thresh, dmind, st['dtarr'][dtind], dtind, st['segment'], st['features'])
            features.append(feature)

    cands = ex.submit(search.collect_cands, features)
    saved = ex.submit(search.save_cands, st, cands)
    return saved


def pipeline_scan(st, host='nmpost-master'):
    """ Given rfpipe state and dask distributed executor, run search pipline """

    ex = distributed.Executor('{0}:{1}'.format(host, '8786'))

    logger.debug('submitting segments')
    for segment in range(st['nsegments']):
        yield pipeline_seg(st, segment, ex)
