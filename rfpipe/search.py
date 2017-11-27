from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
from numba import jit, guvectorize, int64
import pyfftw
from rfpipe import util, candidates

import logging
logger = logging.getLogger(__name__)


def dedisperse(data, delay, mode='single'):
    """ Shift data in time (axis=0) by channel-dependent value given in
    delay. Returns new array with time length shortened by max delay in
    integrations. wraps _dedisperse to add logging.
    Can set mode to "single" or "multi" to use different functions.
    Changes memory in place, so forces writability
    """

    if not np.any(data):
        return np.array([])

    logger.info('Dedispersing up to delay shift of {0} integrations'
                .format(delay.max()))

    if delay.max() > 0:
        nint, nbl, nchan, npol = data.shape
        newsh = (nint-delay.max(), nbl, nchan, npol)
        result = np.zeros(shape=newsh, dtype=data.dtype)

        if mode == 'single':
            _dedisperse_jit(np.require(data, requirements='W'), delay, result)
            return result
        elif mode == 'multi':
            _ = _dedisperse_gu(np.swapaxes(np.require(data, requirements='W'), 0, 1), delay)
            return data[0:len(data)-delay.max()]
        else:
            logger.error('No such dedispersion mode.')
    else:
        return data


@jit(nogil=True, nopython=True)
def _dedisperse_jit(data, delay, result):

    nint, nbl, nchan, npol = data.shape
    for k in range(nchan):
        for i in range(nint-delay.max()):
            iprime = i + delay[k]
            for l in range(npol):
                for j in range(nbl):
                    result[i, j, k, l] = data[iprime, j, k, l]


@guvectorize(["void(complex64[:,:,:], int64[:])"], '(n,m,l),(m)',
             target='parallel', nopython=True)
def _dedisperse_gu(data, delay):
    """ Multicore dedispersion via numpy broadcasting.
    Requires that data be in axis order (nbl, nint, nchan, npol), so typical
    input visibility array must have view from "np.swapaxis(data, 0, 1)".
    """

    for i in range(data.shape[0]-delay.max()):
        for j in range(data.shape[1]):
            iprime = i + delay[j]
            for k in range(data.shape[2]):
                data[i, j, k] = data[iprime, j, k]


def resample(data, dt, mode='single'):
    """ Resample (integrate) by factor dt and return new data structure
    wraps _resample to add logging.
    Can set mode to "single" or "multi" to use different functions.
    Changes memory in place, so forces writability
    """

    if not np.any(data):
        return np.array([])

    len0 = data.shape[0]
    logger.info('Resampling data of length {0} by a factor of {1}'
                .format(len0, dt))

    if dt > 1:
        nint, nbl, nchan, npol = data.shape
        newsh = (int64(nint//dt), nbl, nchan, npol)
        result = np.zeros(shape=newsh, dtype=data.dtype)

        if mode == 'single':
            _resample_jit(np.require(data, requirements='W'), dt, result)
            return result
        elif mode == 'multi':
            _ = _resample_gu(np.swapaxes(np.require(data, requirements='W'), 0, 3), dt)
            return data[:len0//dt]
        else:
            logger.error('No such resample mode.')
    else:
        return data


@jit(nogil=True, nopython=True)
def _resample_jit(data, dt, result):

    nint, nbl, nchan, npol = data.shape
    for j in range(nbl):
        for k in range(nchan):
            for l in range(npol):
                for i in range(int64(nint//dt)):
                    iprime = int64(i*dt)
                    result[i, j, k, l] = data[iprime, j, k, l]
                    for r in range(1, dt):
                        result[i, j, k, l] += data[iprime+r, j, k, l]
                    result[i, j, k, l] = result[i, j, k, l]/dt


@guvectorize(["void(complex64[:], int64)"], '(n),()', target='parallel',
             nopython=True)
def _resample_gu(data, dt):
    """ Multicore resampling via numpy broadcasting.
    Requires that data be in nint axisto be last, so input
    visibility array must have view from "np.swapaxis(data, 0, 3)".
    *modifies original memory space* (unlike _resample_jit)
    """

    if dt > 1:
        for i in range(data.shape[0]//dt):
            iprime = int64(i*dt)
            data[i] = data[iprime]
            for r in range(1, dt):
                data[i] += data[iprime+r]
            data[i] = data[i]/dt


def dedisperseresample(data, delay, dt, mode='single'):
    """ Dedisperse and resample in single function.
    Can set mode to "single" or "multi" to use different functions.
    Changes memory in place, so forces writability
    """

    if not np.any(data):
        return np.array([])

    logger.info('Correcting by delay/resampling {0}/{1} ints in mode {2}'
                .format(delay.max(), dt, mode))

    if delay.max() > 0 or dt > 1:
        nint, nbl, nchan, npol = data.shape
        newsh = (int64(nint-delay.max())//dt, nbl, nchan, npol)
        result = np.zeros(shape=newsh, dtype=data.dtype)

        if mode == 'single':
            _dedisperseresample_jit(data, delay, dt, result)
            return result
        elif mode == 'multi':
            _ = _dedisperseresample_gu(np.swapaxes(np.require(data, requirements='W'), 0, 1), delay, dt)
            return data[0:(len(data)-delay.max())//dt]
        else:
            logger.error('No such dedispersion mode.')
    else:
        return data


@jit(nogil=True, nopython=True)
def _dedisperseresample_jit(data, delay, dt, result):

    nint, nbl, nchan, npol = data.shape
    nintout = int64(len(result))

    for j in range(nbl):
        for l in range(npol):
            for k in range(nchan):
                for i in range(nintout):
                    weight = int64(0)
                    for r in range(dt):
                        iprime = int64(i*dt + delay[k] + r)
                        val = data[iprime, j, k, l]
                        result[i, j, k, l] += val
                        if val != 0j:
                            weight += 1

                    if weight > 0:
                        result[i, j, k, l] = result[i, j, k, l]/weight
                    else:
                        result[i, j, k, l] = weight

    return result


@guvectorize(["void(complex64[:,:,:], int64[:], int64)"], '(n,m,l),(m),()',
             target='parallel', nopython=True)
def _dedisperseresample_gu(data, delay, dt):
    if delay.max() > 0 or dt > 1:
        nint, nchan, npol = data.shape
        for l in range(npol):
            for k in range(nchan):
                for i in range((nint-delay.max())//dt):
                    iprime = int64(i + delay[k])
                    data[i, k, l] = data[iprime, k, l]
                    for r in range(1, dt):
                        data[i, k, l] += data[iprime+r, k, l]
                    data[i, k, l] = data[i, k, l]/dt


#
# searching, imaging, thresholding
#

def search_thresh(st, data, segment, dmind, dtind, integrations=None,
                  beamnum=0, wisdom=None):
    """ High-level wrapper for search algorithms.
    Expects dedispersed, resampled data as input and data state.
    Returns list of CandData objects that define candidates with
    candloc, image, and phased visibility data.

    ** only supports threshold > image max (no min)
    ** dmind, dtind, beamnum assumed to represent current state of data
    """

    if not np.any(data):
        return []

    if integrations is None:
        integrations = list(range(len(data)))
    elif isinstance(integrations, int):
        integrations = [integrations]

    assert isinstance(integrations, list), "integrations should be int, list of ints, or None."

    logger.info('Imaging {0} ints for DM {1} and dt {2}. Size {3}x{4} '
                '(uvres {5}) with mode {6} and {7} threads (seg {8}).'
                .format(len(integrations), st.dmarr[dmind], st.dtarr[dtind],
                        st.npixx, st.npixy, st.uvres, st.fftmode,
                        st.prefs.nthread, segment))

    if 'image1' in st.prefs.searchtype:
        uvw = st.get_uvw_segment(segment)
        images = image(data, uvw, st.npixx,
                       st.npixy, st.uvres, st.fftmode,
                       st.prefs.nthread, wisdom=wisdom)

        logger.debug('Thresholding at {0} sigma.'
                    .format(st.prefs.sigma_image1))

        # TODO: the following is really slow
        canddatalist = []
        for i in range(len(images)):
            peak_snr = images[i].max()/util.madtostd(images[i])
            if peak_snr > st.prefs.sigma_image1:
                candloc = (segment, integrations[i], dmind, dtind, beamnum)
                candim = images[i]
#                logger.info("i {0} shape {1}, loc {2}".format(i, candim.shape, candloc))
#                logger.info("max {0}".format(np.where(candim == candim.max())))
                l, m = st.pixtolm(np.where(candim == candim.max()))
#                logger.info("image peak at l, m: {0}, {1}".format(l, m))
                util.phase_shift(data, uvw, l, m)
#                logger.info("phasing data from: {0}, {1}"
#                            .format(max(0, i-st.prefs.timewindow//2),
#                                    min(i+st.prefs.timewindow//2, len(data))))
                dataph = data[max(0, i-st.prefs.timewindow//2):
                              min(i+st.prefs.timewindow//2, len(data))].mean(axis=1)
                util.phase_shift(data, uvw, -l, -m)
                canddatalist.append(candidates.CandData(state=st, loc=candloc,
                                                        image=candim, data=dataph))
    else:
        raise NotImplemented("only searchtype=image1 implemented")

    # tuple(list(int), list(ndarray), list(ndarray))
#    return (ints, images_thresh, dataph)
    logger.info("{0} candidates returned for (seg, dmind, dtind) = "
                "({1}, {2}, {3})".format(len(canddatalist), segment, dmind,
                                         dtind))

    return canddatalist


def correct_search_thresh(st, segment, data, dmind, dtind, mode='single',
                          wisdom=None):
    """ Fuse the dediserpse, resample, search, threshold functions.
    """

    delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind],
                            st.inttime)

    data_corr = dedisperseresample(data, delay, st.dtarr[dtind], mode=mode)

    canddatalist = search_thresh(st, data_corr, segment, dmind, dtind,
                                 wisdom=wisdom)

    return canddatalist


def image(data, uvw, npixx, npixy, uvres, fftmode, nthread, wisdom=None,
          integrations=None):
    """ Grid and image data.
    Optionally image integrations in list i.
    fftmode can be fftw or cuda.
    nthread is number of threads to use
    """

    mode = 'single' if nthread == 1 else 'multi'

    if integrations is None:
        integrations = list(range(len(data)))
    elif isinstance(integrations, int):
        integrations = [integrations]

    grids = grid_visibilities(data.take(integrations, axis=0), uvw, npixx, npixy, uvres, mode=mode)

    if fftmode == 'fftw':
        logger.debug("Imaging with fftw on {0} threads".format(nthread))
        images = image_fftw(grids, nthread=nthread, wisdom=wisdom)
    elif fftmode == 'cuda':
        logger.debug("Imaging with cuda.")
        images = image_cuda(grids)
    else:
        logger.warn("Imaging fftmode {0} not supported.".format(fftmode))

    return images


def image_cuda(grids):
    """ Run 2d FFT to image each plane of grid array
    """

    try:
        from pyfft.cuda import Plan
        from pycuda.tools import make_default_context
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda
    except ImportError:
        logger.error('ImportError for pycuda or pyfft. Use pyfftw instead.')

    nints, npixx, npixy = grids.shape

    cuda.init()
    context = make_default_context()
    stream = cuda.Stream()

    plan = Plan((npixx, npixy), stream=stream)

    grid_gpu = gpuarray.to_gpu(grids)
    for i in range(0, nints):
        plan.execute(grid_gpu[i], inverse=True)
    grids = grid_gpu.get()

    context.pop()
    return recenter(grids.real, (npixx//2, npixy//2))


def image_fftw(grids, nthread=1, wisdom=None):
    """ Plan pyfftw ifft2 and run it on uv grids (time, npixx, npixy)
    Returns time images.
    """

    if wisdom:
        logger.debug('Importing wisdom...')
        pyfftw.import_wisdom(wisdom)

    logger.debug("Starting pyfftw ifft2")

    images = pyfftw.interfaces.numpy_fft.ifft2(grids, auto_align_input=True,
                                               auto_contiguous=True,
                                               planner_effort='FFTW_MEASURE',
                                               overwrite_input=True,
                                               threads=nthread)

    nints, npixx, npixy = images.shape
    logger.debug('Recentering fft\'d images...')

    return recenter(images.real, (npixx//2, npixy//2))


def grid_visibilities(data, uvw, npixx, npixy, uvres, mode='single'):
    """ Grid visibilities into rounded uv coordinates """

    logger.debug('Gridding {0} ints at ({1}, {2}) pix and {3} '
                'resolution with mode {4}.'.format(len(data), npixx, npixy,
                                                   uvres, mode))
    u, v, w = uvw
    grids = np.zeros(shape=(data.shape[0], npixx, npixy),
                     dtype=np.complex64)

    if mode == 'single':
        _grid_visibilities_jit(data, u, v, w, npixx, npixy, uvres, grids)
    elif mode == 'multi':
        _ = _grid_visibilities_gu(data, u, v, w, npixx, npixy, uvres, grids)
    else:
        logger.error('No such resample mode.')

    return grids


@jit(nogil=True, nopython=True)
def _grid_visibilities_jit(data, u, v, w, npixx, npixy, uvres, grids):
    """ Grid visibilities into rounded uv coordinates using jit on single core.
    Rounding not working here, so minor differences with original and
    guvectorized versions.
    """

    nint, nbl, nchan, npol = data.shape

# rounding not available in numba
#    ubl = np.round(us/uvres, 0).astype(np.int32)
#    vbl = np.round(vs/uvres, 0).astype(np.int32)

    for j in range(nbl):
        for k in range(nchan):
            ubl = int64(u[j, k]/uvres)
            vbl = int64(v[j, k]/uvres)
            if (np.abs(ubl < npixx//2)) and (np.abs(vbl < npixy//2)):
                umod = int64(np.mod(ubl, npixx))
                vmod = int64(np.mod(vbl, npixy))
                for i in range(nint):
                    for l in xrange(npol):
                        grids[i, umod, vmod] += data[i, j, k, l]

    return grids


@guvectorize(["void(complex64[:,:,:], float32[:,:], float32[:,:], float32[:,:], int64, int64, int64, complex64[:,:])"],
             '(n,m,l),(n,m),(n,m),(n,m),(),(),(),(o,p)', target='parallel',
             nopython=True)
def _grid_visibilities_gu(data, us, vs, ws, npixx, npixy, uvres, grid):
    """ Grid visibilities into rounded uv coordinates for multiple cores"""

    ubl = np.zeros(us.shape, dtype=int64)
    vbl = np.zeros(vs.shape, dtype=int64)

    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            ubl[j, k] = int64(np.round(us[j, k]/uvres, 0))
            vbl[j, k] = int64(np.round(vs[j, k]/uvres, 0))
            if (np.abs(ubl[j, k]) < npixx//2) and \
               (np.abs(vbl[j, k]) < npixy//2):
                u = np.mod(ubl[j, k], npixx)
                v = np.mod(vbl[j, k], npixy)
                for l in range(data.shape[2]):
                    grid[u, v] += data[j, k, l]


def recenter(array, center):
    """ Recenters images in array to location center (x, y)
    Array can be either 2d (x, y) or 3d array (time, x, y).
    """

    assert len(center) == 2

    if len(array.shape) == 2:
        return np.roll(np.roll(array, center[0], axis=0), center[1], axis=1)
    elif len(array.shape) == 3:
        return np.roll(np.roll(array, center[0], axis=1), center[1], axis=2)

#    s = a[1:].shape
#    c = (c[0] % s[0], c[1] % s[1])
#    a1 = np.concatenate([a[:, c[0]:], a[:, :c[0]]], axis=1)
#    a2 = np.concatenate([a1[:, :,c[1]:], a1[:, :,:c[1]]], axis=2)
#    return a2


def image_arm():
    """ Takes visibilities and images arms of VLA """

    pass


def set_wisdom(npixx, npixy):
    """ Run single 2d ifft like image to prep fftw wisdom in worker cache """

    logger.info('Calculating FFT wisdom...')
    arr = pyfftw.empty_aligned((npixx, npixy), dtype='complex64', n=16)
    arr[:] = np.random.randn(*arr.shape) + 1j*np.random.randn(*arr.shape)
    fft_arr = pyfftw.interfaces.numpy_fft.ifft2(arr, auto_align_input=True,
                                                auto_contiguous=True,
                                                planner_effort='FFTW_MEASURE')
    return pyfftw.export_wisdom()
