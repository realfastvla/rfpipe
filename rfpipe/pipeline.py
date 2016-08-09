from . import state, source, search
import distributed
import rtpipe
from functools import partial
import rtlib_cython as rtlib
import numpy as np
import logging

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


def run(datasource, paramfile, version=2):
    """ Run whole pipeline """

    st = state.State(paramfile=paramfile, verison=version)

    if datatype(datasource) == 'sdm':
        apply_metadata(st, sdmfile)

    # learn distributed for this part
    if 'image' in st.searchtype:
        search.imaging(st)


def apply_metadata(st, sdmfile):
    """ Run all functions to apply metadata to transform initial state """

    source.parsesdm(sdmfile, st)
    state.set_dmgrid(st)
    state.set_imagegrid(st)
    state.set_segments(st)


def dataprep(st, segment):
    data = rtpipe.parsesdm.read_bdf_segment(st, segment)
    sols = rtpipe.parsecal.telcal_sol(st['gainfile'])   # parse gainfile
    sols.set_selection(st['segmenttimes'][segment].mean(), st['freq']*1e9, rtlib.calc_blarr(st), calname='', pols=st['pols'], radec=(), spwind=[])
    sols.apply(data)
    rtpipe.RT.dataflag(st, data)
    rtlib.meantsub(data, [0, st['nbl']])
    return data


def image1(st, data, uvw):
    u, v, w, = uvw
    ims,snr,candints = rtlib.imgallfullfilterxyflux(np.outer(u, st['freq']/st['freq_orig'][0]), np.outer(v, st['freq']/st['freq_orig'][0]), data, st['npixx'], st['npixy'], st['uvres'], st['sigma_image1'])
    return candints, snr


def correct_dm(st, data, dm):
    data_resamp = data.copy()
    rtlib.dedisperse_par(data_resamp, st['freq'], st['inttime'], dm, [0, st['nbl']], verbose=0)        # dedisperses data.
    return data_resamp


def correct_dt(st, data, dt):
    rtlib.resample_par(data, st['freq'], st['inttime'], dt, [0, st['nbl']], verbose=0)        # dedisperses data.
    return data


def get_executor(hostname, port='8786'):
    return distributed.Executor('{0}:{1}'.format(hostname, port))


def get_state(sdmfile, scan):
    return rtpipe.RT.set_pipeline(sdmfile, scan, memory_limit=3, logfile=False, timesub='mean')


def pipeline(st, ex):
    """ Given rfpipe state and dask distributed executor, run search pipline """

    # run pipeline
    future_dict = {}
    scan = str(st['scan'])

    logger.debug('submitting segments')
    for segment in range(st['nsegments']):
        data_read = ex.submit(dataprep, st, segment)
        uvw = ex.submit(rtpipe.parsesdm.get_uvw_segment, st, segment)

        logger.debug('submitting dedispersion')
        for dmind in range(len(st['dmarr'])):
            data_dm = ex.submit(correct_dm, st, data_read, st['dmarr'][dmind])

            dtind = 0
            im_dt = ex.submit(image1, st, data_dm, uvw)
            key ='{0}-{1}-{2}-{3}'.format(scan, segment, dmind, dtind)
            future_dict[key] = im_dt

            data_dt = data_dm
            print('submitting reampling and imaging')
            for dtind in range(1, len(st['dtarr'])):
                data_dt = ex.submit(correct_dt, st, data_dt, 2)
                im_dt = ex.submit(image1, st, data_dt, uvw)
                key ='{0}-{1}-{2}-{3}'.format(scan, segment, dmind, dtind)
                future_dict[key] = im_dt

    return future_dict
