from . import state, source, search
import distributed
import rtpipe
from functools import partial
import rtlib_cython as rtlib
import numpy as np

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


def imthresh(sigma, im):
    snr = im.max()/im.std()
    if snr > sigma:
        return im

def image1(st, data, uvw):
    u, v, w, = uvw
    ims,snr,candints = rtlib.imgallfullfilterxyflux(np.outer(u, st['freq']/st['freq_orig'][0]), np.outer(v, st['freq']/st['freq_orig'][0]), data, st['npixx'], st['npixy'], st['uvres'], st['sigma_image1'])
    return candints


def correct_dm(st, data, dm):
    data_resamp = data.copy()
    rtlib.dedisperse_par(data_resamp, st['freq'], st['inttime'], dm, [0, st['nbl']], verbose=0)        # dedisperses data.
    return data_resamp


def correct_dt(st, datadt):
    data, dt = datadt
    rtlib.resample_par(data, st['freq'], st['inttime'], dt, [0, st['nbl']], verbose=0)        # dedisperses data.
    return data


def get_scheduler(hostname, port='8786'):
    return distributed.Executor('{0}:{1}'.format(hostname, port))


def pipeline(st, ex):
    """ Given rfpipe state and dask distributed executor, run search pipline """

    # run pipeline
    data = ex.map(readdata, range(st['nsegments']))
    uvw = ex.map(readuvw, range(st['nsegments']))

    imsig = []
    for i in range(len(data)):
        imsig_dm = []
        dd = data[i]
        for dm in st['dmarr']:
            data_dm = ex.submit(correct_dm, st, dd, dm)
            imsig_dm.append(ex.submit(image1, st, data_dm, uvw[i]))
        imsig.append(imsig_dm)

#        while True:
#            resample = partial(correct_dt, st, data_dedisp)
#            data_resamp = ex.submit(resample, 2)   # every resample relative to previous iteration (2x resample each loop)
#            if dtind == len(st['dtarr']):
#                break

    return imsig
