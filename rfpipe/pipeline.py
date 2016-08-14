import logging, os, pickle
from functools import partial
import numpy as np
from scipy.stats import mstats
from . import state, source, search
import dask
import distributed
import rtpipe
import rtlib_cython as rtlib

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
# testing dask distributed
##

@dask.delayed(pure=True)
def dataprep(st, segment):
#    st['segment'] = segment # allowed?

    data = rtpipe.parsesdm.read_bdf_segment(st, segment)
    sols = rtpipe.parsecal.telcal_sol(st['gainfile'])   # parse gainfile
    sols.set_selection(st['segmenttimes'][segment].mean(), st['freq']*1e9, rtlib.calc_blarr(st), calname='', pols=st['pols'], radec=(), spwind=[])
    sols.apply(data)
    rtpipe.RT.dataflag(st, data)
    rtlib.meantsub(data, [0, st['nbl']])
    return data

@dask.delayed(pure=True)
def calc_uvw(st, segment):
    return rtpipe.parsesdm.get_uvw_segment(st, segment)

@dask.delayed(pure=True)
def correct_dm(st, data, dm):
    data_resamp = data.copy()
    rtlib.dedisperse_par(data_resamp, st['freq'], st['inttime'], dm, [0, st['nbl']], verbose=0)        # dedisperses data.
    return data_resamp


@dask.delayed(pure=True)
def correct_dt(st, data, dt):
    rtlib.resample_par(data, st['freq'], st['inttime'], dt, [0, st['nbl']], verbose=0)        # dedisperses data.
    return data


@dask.delayed(pure=True)
def image1(st, data, uvw):
    u, v, w, = uvw
    return rtlib.imgallfullfilterxyflux(np.outer(u, st['freq']/st['freq_orig'][0]), np.outer(v, st['freq']/st['freq_orig'][0]), data, st['npixx'], st['npixy'], st['uvres'], st['sigma_image1'])


@dask.delayed(pure=True)
def correct_image(st, data, uvw, dm, dt):
    u, v, w, = uvw
    data_resamp = data.copy()

    rtlib.dedisperse_par(data_resamp, st['freq'], st['inttime'], dm, [0, st['nbl']], verbose=0)        # dedisperses data.
    if dt > 1:
        rtlib.resample_par(data_resamp, st['freq'], st['inttime'], dt, [0, st['nbl']], verbose=0)        # dedisperses data.

    return rtlib.imgallfullfilterxyflux(np.outer(u, st['freq']/st['freq_orig'][0]), np.outer(v, st['freq']/st['freq_orig'][0]), data_resamp, st['npixx'], st['npixy'], st['uvres'], st['sigma_image1'])


def correct_image2(st, data, uvw, dm, dt):
    u, v, w, = uvw
    data_resamp = data.copy()

    rtlib.dedisperse_par(data_resamp, st['freq'], st['inttime'], dm, [0, st['nbl']], verbose=0)        # dedisperses data.
    if dt > 1:
        rtlib.resample_par(data_resamp, st['freq'], st['inttime'], dt, [0, st['nbl']], verbose=0)        # dedisperses data.

    return rtlib.imgallfullfilterxyflux(np.outer(u, st['freq']/st['freq_orig'][0]), np.outer(v, st['freq']/st['freq_orig'][0]), data_resamp, st['npixx'], st['npixy'], st['uvres'], st['sigma_image1'])


@dask.delayed(pure=True)
def calc_features(imgall, dmind, dt, dtind, segment, featurelist):
    ims, snr, candints = imgall
    beamnum = 0

    feat = {}
    for i in xrange(len(candints)):
        candid =  (segment, candints[i]*dt, dmind, dtind, beamnum)

        # assemble feature in requested order
        ff = []
        for feature in featurelist:
            if feature == 'snr1':
                ff.append(snr[i])
            elif feature == 'immax1':
                if snr[i] > 0:
                    ff.append(ims[i].max())
                else:
                    ff.append(ims[i].min())

        feat[candid] = list(ff)
    return feat


@dask.delayed(pure=True)
def savecands(st, cands):
    """ Save all candidates in pkl file for later aggregation and filtering.
    domock is option to save simulated cands file
    """

    candsfile = os.path.join(st['workdir'], 'cands_' + st['fileroot'] + '_sc' + str(st['scan']) + 'seg' + str(st['segment']) + '.pkl')
    with open(candsfile, 'w') as pkl:
        pickle.dump(st, pkl)
        pickle.dump(cands, pkl)

    return True


@dask.delayed(pure=True)
def collectcands(feature_list):

    cands = {}
    for features in feature_list:
        for kk in features.iterkeys():
            cands[kk] = features[kk]
                
    return cands


@dask.delayed(pure=True)
def collectsegs(segfutures):
    return [fut for fut in segfutures]


def get_executor(hostname, port='8786'):
    return distributed.Executor('{0}:{1}'.format(hostname, port))


def get_state(sdmfile, scan, **kwargs):
    return rtpipe.RT.set_pipeline(sdmfile, scan, searchtype='image1', memory_limit=3, logfile=False, timesub='mean', **kwargs)


def pipeline_scan(st):
    """ Given rfpipe state and dask distributed executor, run search pipline """

    segfutures = []
    logger.debug('submitting segments')
    for segment in range(st['nsegments']):
        segfutures.append(pipeline_seg(st, 0))

    return collectsegs(segfutures)


def pipeline_seg(st, segment):
    """ Given state and segment, run search pipeline """

    # run pipeline
    scan = str(st['scan'])
    feature_list = []
    st['segment'] = segment

    logger.info('reading...')
    data_read = dataprep(st, segment)
    uvw = calc_uvw(st, segment)

    for dmind in range(len(st['dmarr'])):
#        data_dm = correct_dm(st, data_read, st['dmarr'][dmind])
#        data_dt = data_dm
#        dtind = 0
#        im_dt = image1(st, data_dt, uvw)
        for dtind in range(len(st['dtarr'])):
            logger.info('submitting reampling and imaging')
            im_dt = correct_image(st, data_read, uvw, st['dmarr'][dmind], st['dtarr'][dtind])

            feature_list.append(calc_features(im_dt, dmind, st['dtarr'][dtind], dtind, st['segment'], st['features']))

#        for dtind in range(1, len(st['dtarr'])):
#            data_dt = correct_dt(st, data_dt, 2)
#            im_dt = image1(st, data_dt, uvw)
#            feature_list.append(calc_features(im_dt, dmind, st['dtarr'][dtind], dtind, st['segment'], st['features']))

    cands = collectcands(feature_list)
    return savecands(st, cands)


#def run(sdmfile, scan, host, **kwargs):
#    st = get_state(sdmfile, scan, **kwargs)
def run(st, segment, host):
    with distributed.Executor('{0}:{1}'.format(host, '8786')) as ex:
        with dask.set_options(get=ex.get):
            status = pipeline_seg(st, segment).compute()

    return status


def run2(st, data, uvw, host):
    ex = distributed.Executor('{0}:{1}'.format(host, '8786'))

    im_dt = []
    for dmind in range(len(st['dmarr'])):
        for dtind in range(len(st['dtarr'])):
            im_dt.append(ex.submit(correct_image2, st, data, uvw, st['dmarr'][dmind], st['dtarr'][dtind]))

    return im_dt


from functools import partial
def adder(data, nn):
    return data + nn


def run3(sh, num, host):
    ex = distributed.Executor('{0}:{1}'.format(host, '8786'))

    arr = np.zeros(shape=sh, dtype='complex64')
    arr.real = np.random.normal(size=sh)
    arr.imag = np.random.normal(size=sh)
    adderp = partial(adder, arr)

#    fut = []
#    for nn in range(num):
#        fut.append(ex.submit(lambda x: x+nn, arr, pure=False))
    fut = ex.map(adderp, range(num))

    return fut
