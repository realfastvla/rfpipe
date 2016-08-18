import logging
from . import state, source, search
import distributed
import rtpipe

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


def dataprep(st, segment):
    data = rtpipe.parsesdm.read_bdf_segment(st, segment)
#    sols = rtpipe.parsecal.telcal_sol(st['gainfile'])   # parse gainfile
#    sols.set_selection(st['segmenttimes'][segment].mean(), st['freq']*1e9, rtlib.calc_blarr(st), calname='', pols=st['pols'], radec=(), spwind=[])
#    sols.apply(data)
#    rtpipe.RT.dataflag(st, data)
    data = search.meantsub(data)
    return data


def calc_uvw(st, segment):
    return rtpipe.parsesdm.get_uvw_segment(st, segment)


def pipeline_seg(st, segment, ex):
    """ Run segment pipelne with ex.submit calls """

    features = []
    st['segment'] = segment

    logger.info('reading...')
    data_read = ex.submit(dataprep, st, segment)
    uvw = ex.submit(calc_uvw, st, segment)
    ex.replicate([data_read, uvw])  # spread data around to get ready for many core imaging

    for dmind in range(len(st['dmarr'])):
        data_dm = ex.submit(search.dedisperse, data_read, st['freq'], st['inttime'], st['dmarr'][dmind])

        for dtind in range(len(st['dtarr'])):
            # resample and search with one function
            ims_thresh = ex.submit(search.resample_image, data_dm, st['dtarr'][dtind], uvw, st['freq'], st['npixx'], st['npixy'], st['uvres'], st['sigma_image1'])

            # resample and search in four functions
#            data_dt = ex.submit(search.resample, data_dm, st['dtarr'][dtind])
#            uvgrid = ex.submit(search.grid_visibilities, data_dt, uvw, st['freq'], st['npixx'], st['npixy'], st['uvres'])
#            ims = ex.submit(search.image_fftw, uvgrid, st['nthread'])
#            ims_thresh = ex.submit(search.threshold_images, ims, st['sigma_image1'])
            
            feature = ex.submit(search.calc_features, ims_thresh, dmind, st['dtarr'][dtind], dtind, st['segment'], st['features'])
            features.append(feature)

    cands = ex.submit(search.collect_cands, features)
    result = ex.submit(search.save_cands, st, cands)
    return result
#    return features


def pipeline_scan(st, host='nmpost-master'):
    """ Given rfpipe state and dask distributed executor, run search pipline """

    futures = []
    ex = distributed.Executor('{0}:{1}'.format(host, '8786'))

    logger.debug('submitting segments')
    for segment in range(st['nsegments']):
        futures.append(pipeline_seg(st, segment, ex))

    return futures
