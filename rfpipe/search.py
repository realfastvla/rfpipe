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


def dedisperse(data, delay, mode='multi'):
    """ Shift data in time (axis=0) by channel-dependent value given in
    delay. Returns new array with time length shortened by max delay in
    integrations. wraps _dedisperse to add logging.
    Can set mode to "single" or "multi" to use different functions.
    """

    if not np.any(data):
        return np.array([])

    # changes memory in place, so need to force writability
    data = np.require(data, requirements='W')

    logger.info('Dedispersing up to delay shift of {0} integrations'
                .format(delay.max()))

    if mode == 'single':
        return _dedisperse_jit(data, delay)
    elif mode == 'multi':
        _ = _dedisperse_gu(np.swapaxes(data, 0, 1), delay)
        return data[0:len(data)-delay.max()]
    else:
        logger.error('No such dedispersion mode.')


@jit(nogil=True, nopython=True)
def _dedisperse_jit(data, delay):

    if delay.max() > 0:
        nint, nbl, nchan, npol = data.shape
        newsh = (nint-delay.max(), nbl, nchan, npol)
        result = np.zeros(shape=newsh, dtype=data.dtype)
        for k in range(nchan):
            for i in range(nint-delay.max()):
                iprime = i + delay[k]
                for l in range(npol):
                    for j in range(nbl):
                        result[i, j, k, l] = data[iprime, j, k, l]
        return result
    else:
        return data


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


def resample(data, dt, mode='multi'):
    """ Resample (integrate) by factor dt and return new data structure
    wraps _resample to add logging.
    Can set mode to "single" or "multi" to use different functions.
    """

    if not np.any(data):
        return np.array([])

    # changes memory in place, so need to force writability
    data = np.require(data, requirements='W')

    len0 = data.shape[0]
    logger.info('Resampling data of length {0} by a factor of {1}'
                .format(len0, dt))

    if mode == 'single':
        return _resample_jit(data, dt)
    elif mode == 'multi':
        _ = _resample_gu(np.swapaxes(data, 0, 3), dt)
        return data[:len0//dt]
    else:
        logger.error('No such resample mode.')


@jit(nogil=True, nopython=True)
def _resample_jit(data, dt):

    if dt > 1:
        nint, nbl, nchan, npol = data.shape
        newsh = (int64(nint//dt), nbl, nchan, npol)
        result = np.zeros(shape=newsh, dtype=data.dtype)

        for j in range(nbl):
            for k in range(nchan):
                for l in range(npol):
                    for i in range(int64(nint//dt)):
                        iprime = int64(i*dt)
                        result[i, j, k, l] = data[iprime, j, k, l]
                        for r in range(1, dt):
                            result[i, j, k, l] += data[iprime+r, j, k, l]
                        result[i, j, k, l] = result[i, j, k, l]/dt

        return result
    else:
        return data


@guvectorize(["void(complex64[:], int64)"], '(n),()', target='parallel', nopython=True)
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
            for r in range(1,dt):
#                print(i0, i, iprime, r)
                data[i] += data[iprime+r]
            data[i] = data[i]/dt

#
# searching, imaging, thresholding
#


def search_thresh(st, data, segment, dmind, dtind, integrations=None, beamnum=0, wisdom=None):
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

    data = np.require(data, requirements='W')
    uvw = st.get_uvw_segment(segment)

    logger.info('Imaging {0} ints for DM {1} and dt {2}. Size {3}x{4} '
                '(uvres {5}) with mode {6}.'
                .format(len(integrations), st.dmarr[dmind], st.dtarr[dtind],
                        st.npixx, st.npixy, st.uvres, st.fftmode))

    if 'image1' in st.prefs.searchtype:
        images = image(data, uvw, st.npixx, st.npixy, st.uvres, st.fftmode,
                       st.prefs.nthread, integrations=integrations,
                       wisdom=wisdom)

        logger.debug('Thresholding at {2} sigma.'
                     .format(st.prefs.sigma_image1))

        # TODO: the following is really slow
        canddatalist = []
        for i in range(len(images)):
            if (images[i].max()/util.madtostd(images[i]) >
               st.prefs.sigma_image1):
                candloc = (segment, i, dmind, dtind, beamnum)
                candim = images[i]
                l, m = st.pixtolm(np.where(candim == candim.max()))
                logger.debug("image peak at l, m: {0}, {1}".format(l, m))
                util.phase_shift(data, uvw, l, m)
                logger.debug("phasing data from: {0}, {1}"
                             .format(max(0, i-st.prefs.timewindow//2),
                                     min(i+st.prefs.timewindow//2, len(data))))
                dataph = data[max(0, i-st.prefs.timewindow//2):
                              min(i+st.prefs.timewindow//2, len(data))].mean(axis=1)
                util.phase_shift(data, uvw, -l, -m)
                canddatalist.append(candidates.CandData(state=st, loc=candloc,
                                                        image=candim, data=dataph))
    else:
        raise NotImplemented("only searchtype=image1 implemented")

    # tuple(list(int), list(ndarray), list(ndarray))
#    return (ints, images_thresh, dataph)
    logger.info("Returning data for {0} candidates".format(len(canddatalist)))

    return canddatalist


def image(data, uvw, npixx, npixy, uvres, fftmode, nthread, wisdom=None,
          integrations=None):
    """ Grid and image data.
    Optionally image integrations in list i.
    fftmode can be fftw or cuda.
    nthread is number of threads to use
    """

    mode = 'single' if nthread == 1 else 'multi'

    grids = grid_visibilities(data.take(integrations, axis=0), uvw, npixx,
                              npixy, uvres, mode=mode)

    if fftmode == 'fftw':
#        nthread = 1
        logger.debug("Imaging with fftw on {0} threads".format(nthread))
        images = image_fftw(grids, nthread=nthread, wisdom=wisdom)  # why unstable?
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

    images = pyfftw.interfaces.numpy_fft.ifft2(grids, auto_align_input=True,
                                               auto_contiguous=True,
                                               planner_effort='FFTW_MEASURE',
                                               overwrite_input=True,
                                               threads=nthread)

    npixx, npixy = images[0].shape

    return recenter(images.real, (npixx//2, npixy//2))


def grid_visibilities(data, uvw, npixx, npixy, uvres, mode='multi'):
    """ Grid visibilities into rounded uv coordinates """

    data = np.require(data, requirements='W')
    logger.debug('Gridding visibilities for grid of ({0}, {1}) pix and {2} '
                 'resolution.'.format(npixx, npixy, uvres))

    if mode == 'single':
        return _grid_visibilities_jit(data, uvw, npixx, npixy, uvres)
    elif mode == 'multi':
        grid = np.zeros(shape=(data.shape[0], npixx, npixy),
                        dtype=np.complex64)
        u, v, w = uvw
        _ = _grid_visibilities_gu(data, u, v, w, npixx, npixy, uvres, grid)
        return grid
    else:
        logger.error('No such resample mode.')


@jit(nogil=True, nopython=True)
def _grid_visibilities_jit(data, uvw, npixx, npixy, uvres):
    """ Grid visibilities into rounded uv coordinates using jit on single core.
    Rounding not working here, so minor differences with original and
    guvectorized versions.
    """

    us, vs, ws = uvw
    nint, nbl, nchan, npol = data.shape

# rounding not available in numba
#    ubl = np.round(us/uvres, 0).astype(np.int32)
#    vbl = np.round(vs/uvres, 0).astype(np.int32)
    ubl = (us/uvres).astype(np.int32)
    vbl = (vs/uvres).astype(np.int32)

    grids = np.zeros(shape=(nint, npixx, npixy), dtype=np.complex64)

    for j in range(nbl):
        for k in range(nchan):
            if (np.abs(ubl[j, k]) < npixx//2) and \
               (np.abs(vbl[j, k]) < npixy//2):
                u = int64(np.mod(ubl[j, k], npixx))
                v = int64(np.mod(vbl[j, k], npixy))
                for i in range(nint):
                    for l in xrange(npol):
                        grids[i, u, v] = grids[i, u, v] + data[i, j, k, l]

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
