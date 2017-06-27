from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging, os, math, pickle
logger = logging.getLogger(__name__)

import numba
from numba import cuda
from numba import jit, vectorize, guvectorize, int32, int64, float_, complex64, bool_
import numpy as np
import pyfftw
# import pycuda?


@jit(nopython=True, nogil=True)
def uvcell(uv, freq, freqref, uvres):
    """ Given a u or v coordinate, scale by freq and round to units of uvres """

    cell = np.zeros(len(freq), dtype=np.int32)
    for i in range(len(freq)):
        cell[i] = np.round(uv*freq[i]/freqref/uvres, 0)

    return cell


@vectorize(nopython=True)
def get_mask(x):
    """ Returns equal sized array of 0/1 """
    
    return x != 0j


def runcuda(func, arr, threadsperblock, *args, **kwargs):
    """ Function to run cuda kernels while defining threads/blocks """

    blockspergrid = []
    for tpb, sh in threadsperblock, arr.shape:
        blockspergrid.append = int32(math.ceil(sh / tpb))
    func[tuple(blockspergrid), threadsperblock](arr, *args, **kwargs)


@jit(nogil=True, nopython=True)
def dedisperse(data, delay):
    """ Dispersion shift to new array """

    sh = data.shape
    newsh = (sh[0]-delay.max(), sh[1], sh[2], sh[3])
    result = np.zeros(shape=newsh, dtype=data.dtype)

    if delay.max() > 0:
        for k in range(sh[2]):
            for i in range(newsh[0]):
                iprime = i + delay[k]
                for l in range(sh[3]):
                    for j in range(sh[1]):
                        result[i,j,k,l] = data[iprime,j,k,l]
        return result
    else:
        return data


@jit(nogil=True, nopython=True)
def resample(data, dt):
    """ Resample (integrate) in place by factor dt """

    sh = data.shape
    newsh = (int64(sh[0]/dt), sh[1], sh[2], sh[3])

    if dt > 1:
        result = np.zeros(shape=newsh, dtype=data.dtype)

        for j in range(sh[1]):
            for k in range(sh[2]):
                for l in range(sh[3]):
                    for i in range(newsh[0]):
                        iprime = int64(i*dt)
                        for r in range(dt):
                            result[i,j,k,l] = result[i,j,k,l] + data[iprime+r,j,k,l]
                        result[i,j,k,l] = result[i,j,k,l]/dt

        return result
    else:
        return data


##
## CUDA 
##


## 
## fft and imaging
##


#@jit
def resample_image(data, dt, uvw, freqs, npixx, npixy, uvres, threshold, wisdom=None):
    """ All stages of analysis for a given dt image grid """

    data_resampled = resample(data, dt)
    grids = grid_visibilities(data_resampled, uvw, freqs, npixx, npixy, uvres)
    images = image_fftw(grids, wisdom=wisdom)
    images_thresh = threshold_images(images, threshold)

    return images_thresh


#@jit(nogil=True, nopython=True)
def grid_visibilities(visdata, uvw, freqs, npixx, npixy, uvres):
    """ Grid visibilities into rounded uv coordinates """

    us, vs, ws = uvw
    nint, nbl, nchan, npol = visdata.shape

    grids = np.zeros(shape=(nint, npixx, npixy), dtype=np.complex64)

    for j in range(nbl):
        ubl = uvcell(us[j], freqs, freqs[-1], uvres)
        vbl = uvcell(vs[j], freqs, freqs[-1], uvres)

#        if np.logical_and((np.abs(ubl) < npixx/2), (np.abs(vbl) < npixy/2)):
        for k in range(nchan):
            if (ubl[k] < npixx/2) and (np.abs(vbl[k]) < npixy/2):
                u = int64(np.mod(ubl[k], npixx))
                v = int64(np.mod(vbl[k], npixy))
                for i in range(nint):
                    for l in xrange(npol):
                        grids[i, u, v] = grids[i, u, v] + visdata[i, j, k, l]

    return grids


def set_wisdom(npixx, npixy):
    """ Run single 2d ifft like image to prep fftw wisdom in worker cache """

    arr = pyfftw.empty_aligned((npixx, npixy), dtype='complex64', n=16)
    arr[:] = np.random.randn(*arr.shape) + 1j*np.random.randn(*arr.shape)
    fft_arr = pyfftw.interfaces.numpy_fft.ifft2(arr, auto_align_input=True, auto_contiguous=True,  planner_effort='FFTW_MEASURE')
    return pyfftw.export_wisdom()


#@jit    # no point?
def image_fftw(grids, wisdom=None):
    """ Plan pyfftw ifft2 and run it on uv grids (time, npixx, npixy)
    Returns time images.
    """

    if wisdom:
        logger.debug('Importing wisdom...')
        pyfftw.import_wisdom(wisdom)
    images = pyfftw.interfaces.numpy_fft.ifft2(grids, auto_align_input=True, auto_contiguous=True,  planner_effort='FFTW_MEASURE')

    return images.real


#@jit(nopython=True)  # not working. lowering error?
def threshold_images(images, threshold):
    """ Take time images and return subset above threshold """

    ims = []
    snrs = []
    ints = []
    for i in range(len(images)):
        im = images[i]
        snr = im.max()/im.std()
        if snr > threshold:
            ims.append(im)
            snrs.append(snr)
            ints.append(i)

    return (ims, snrs, ints)


def image_arm():
    """ Takes visibilities and images arms of VLA """

    pass


##
## candidates and features
##

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


def candplot(imgall, data_dm):
    """ Takes results of imaging threshold operation and data to make candidate plot """

    pass


def collect_cands(feature_list):

    cands = {}
    for features in feature_list:
        for kk in features.iterkeys():
            cands[kk] = features[kk]
                
    return cands


def save_cands(st, cands, segment):
    """ Save all candidates in pkl file for later aggregation and filtering.
    domock is option to save simulated cands file
    """

    candsfile = os.path.join(st.metadata.workdir, 'cands_' + st.fileroot + '_sc' + str(st.metadata.scan) + 'seg' + str(segment) + '.pkl')
    with open(candsfile, 'wb') as pkl:
        pickle.dump(st, pkl)
        pickle.dump(cands, pkl)

    return cands
