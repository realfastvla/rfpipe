from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
from numba import jit, guvectorize, int64
import pyfftw
from rfpipe import util, candidates, source

import logging
logger = logging.getLogger(__name__)


def dedisperse(data, delay, parallel=False):
    """ Shift data in time (axis=0) by channel-dependent value given in
    delay. Returns new array with time length shortened by max delay in
    integrations. wraps _dedisperse to add logging.
    Can set mode to "single" or "multi" to use different functions.
    """

    if not np.any(data):
        return np.array([])

    logger.info('Dedispersing up to delay shift of {0} integrations'
                .format(delay.max()))

    nint, nbl, nchan, npol = data.shape
    newsh = (nint-delay.max(), nbl, nchan, npol)
    if parallel:
        data = data.copy()
        _ = _dedisperse_gu(np.swapaxes(data, 0, 1), delay)
        return data[0:len(data)-delay.max()]
    else:
        result = np.zeros(shape=newsh, dtype=data.dtype)
        _dedisperse_jit(np.require(data, requirements='W'), delay, result)
        return result


@jit(nogil=True, nopython=True)
def _dedisperse_jit(data, delay, result):

    nint, nbl, nchan, npol = data.shape
    for k in range(nchan):
        for i in range(nint-delay.max()):
            iprime = i + delay[k]
            for l in range(npol):
                for j in range(nbl):
                    result[i, j, k, l] = data[iprime, j, k, l]


@guvectorize([str("void(complex64[:,:,:], int64[:])")], str("(n,m,l),(m)"),
             target='parallel', nopython=True)
def _dedisperse_gu(data, delay):
    b""" Multicore dedispersion via numpy broadcasting.
    Requires that data be in axis order (nbl, nint, nchan, npol), so typical
    input visibility array must have view from "np.swapaxis(data, 0, 1)".
    """

    if delay.max() > 0:
        for i in range(data.shape[0]-delay.max()):
            for j in range(data.shape[1]):
                iprime = i + delay[j]
                for k in range(data.shape[2]):
                    data[i, j, k] = data[iprime, j, k]


def resample(data, dt, parallel=False):
    """ Resample (integrate) by factor dt and return new data structure
    wraps _resample to add logging.
    Can set mode to "single" or "multi" to use different functions.
    """

    if not np.any(data):
        return np.array([])

    len0 = data.shape[0]
    logger.info('Resampling data of length {0} by a factor of {1}'
                .format(len0, dt))

    nint, nbl, nchan, npol = data.shape
    newsh = (int64(nint//dt), nbl, nchan, npol)

    if parallel:
        data = data.copy()
        _ = _resample_gu(np.swapaxes(data, 0, 3), dt)
        return data[:len0//dt]
    else:
        result = np.zeros(shape=newsh, dtype=data.dtype)
        _resample_jit(np.require(data, requirements='W'), dt, result)
        return result


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


@guvectorize([str("void(complex64[:], int64)")], str("(n),()"),
             target="parallel", nopython=True)
def _resample_gu(data, dt):
    b""" Multicore resampling via numpy broadcasting.
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


def dedisperseresample(data, delay, dt, parallel=False):
    """ Dedisperse and resample in single function.
    parallel controls use of multicore versions of algorithms.
    """

    if not np.any(data):
        return np.array([])

    logger.info('Correcting by delay/resampling {0}/{1} ints in {2} mode'
                .format(delay.max(), dt, ['single', 'parallel'][parallel]))

    nint, nbl, nchan, npol = data.shape
    newsh = (int64(nint-delay.max())//dt, nbl, nchan, npol)

    if parallel:
        data = data.copy()
        _ = _dedisperseresample_gu(np.swapaxes(data, 0, 1),
                                   delay, dt)
        return data[0:(len(data)-delay.max())//dt]
    else:
        result = np.zeros(shape=newsh, dtype=data.dtype)
        _dedisperseresample_jit(data, delay, dt, result)
        return result


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


@guvectorize([str("void(complex64[:,:,:], int64[:], int64)")],
             str("(n,m,l),(m),()"), target="parallel", nopython=True)
def _dedisperseresample_gu(data, delay, dt):

    if delay.max() > 0 or dt > 1:
        nint, nchan, npol = data.shape
        for l in range(npol):
            for k in range(nchan):
                for i in range((nint-delay.max())//dt):
                    weight = int64(0)
                    for r in range(dt):
                        iprime = int64(i*dt + delay[k] + r)
                        val = data[iprime, k, l]
                        if r == 0:
                            data[i, k, l] = val
                        else:
                            data[i, k, l] += val
                        if val != 0j:
                            weight += 1
                    if weight > 0:
                        data[i, k, l] = data[i, k, l]/weight
                    else:
                        data[i, k, l] = weight


#
# searching, imaging, thresholding
#

def prep_and_search(st, segment, data):
    """ Bundles prep and search functions to improve performance in distributed.
    """

    data_prep = source.data_prep(st, segment, data)

    if st.prefs.fftmode == "cuda":
        candcollection = dedisperse_image_cuda(st, segment, data_prep)
    elif st.prefs.fftmode == "fftw":
        candcollection = dedisperse_image_fftw(st, segment, data_prep)
    else:
        logger.warn("fftmode {0} not recognized (cuda, fftw allowed)"
                    .format(st.prefs.fftmode))

    candidates.save_cands(st, candcollection)

    return candcollection


def dedisperse_image_cuda(st, segment, data, devicenum=None):
    """ Run dedispersion, resample for all dm and dt.
    Grid and image on GPU.
    rfgpu is built from separate repo.
    Uses state to define integrations to image based on segment, dm, and dt.
    devicenum can force the gpu to use, but can be inferred via distributed.
    """

    assert st.dtarr[0] == 1, "st.dtarr[0] assumed to be 1"
    assert all([st.dtarr[dtind]*2 == st.dtarr[dtind+1]
                for dtind in range(len(st.dtarr)-1)]), ("dtarr must increase "
                                                        "by factors of 2")

    try:
        import rfgpu
    except ImportError:
        logger.warn('ImportError for rfgpu. Exiting.')
        return []

    if not np.any(data):
        logger.info("Data is all zeros. Skipping search.")
        return []

    if devicenum is None:
        # assume first gpu, but try to infer from worker name
        devicenum = 0
        try:
            from distributed import get_worker
            name = get_worker().name
            devicenum = int(name.split('g')[1])
            logger.debug("Using name {0} to set GPU devicenum to {1}"
                         .format(name, devicenum))
        except IndexError:
            logger.warn("Could not parse worker name {0}. Using default GPU devicenum {1}"
                        .format(name, devicenum))
        except ValueError:
            logger.warn("No worker found. Using default GPU devicenum {0}"
                        .format(devicenum))
        except ImportError:
            logger.warn("distributed not available. Using default GPU devicenum {0}"
                        .format(devicenum))

    rfgpu.cudaSetDevice(devicenum)

    bytespercd = 8*(st.npixx*st.npixy + st.prefs.timewindow*st.nchan*st.npol)

    beamnum = 0
    uvw = util.get_uvw_segment(st, segment)

    upix = st.npixx
    vpix = st.npixy//2 + 1

    grid = rfgpu.Grid(st.nbl, st.nchan, st.readints, upix, vpix)
    image = rfgpu.Image(st.npixx, st.npixy)
    image.add_stat('rms')
    image.add_stat('max')

    # Data buffers on GPU
    vis_raw = rfgpu.GPUArrayComplex((st.nbl, st.nchan, st.readints))
    vis_grid = rfgpu.GPUArrayComplex((upix, vpix))
    img_grid = rfgpu.GPUArrayReal((st.npixx, st.npixy))

    # Convert uv from lambda to us
    u, v, w = uvw
    u_us = 1e6*u[:, 0]/(1e9*st.freq[0])
    v_us = 1e6*v[:, 0]/(1e9*st.freq[0])

    # Q: set input units to be uv (lambda), freq in GHz?
    grid.set_uv(u_us, v_us)  # u, v in us
    grid.set_freq(st.freq*1e3)  # freq in MHz
    grid.set_cell(st.uvres)  # uv cell size in wavelengths (== 1/FoV(radians))

    # Compute gridding transform
    grid.compute()

    # move Stokes I data in (assumes dual pol data)
    vis_raw.data[:] = np.rollaxis(data.mean(axis=3), 0, 3)
    vis_raw.h2d()  # Send it to GPU memory

    grid.conjugate(vis_raw)

    canddatalist = []
    candcollection = candidates.CandCollection(prefs=st.prefs,
                                               metadata=st.metadata)
    for dtind in range(len(st.dtarr)):
        for dmind in range(len(st.dmarr)):
            delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind],
                                    st.inttime)

            if dtind > 0:
                grid.downsample(vis_raw)

            grid.set_shift(delay >> dtind)  # dispersion shift per chan in samples

            integrations = st.get_search_ints(segment, dmind, dtind)
            minint = min(integrations)
            maxint = max(integrations)

            logger.info('Imaging {0} ints ({1}-{2}) in seg {3} at DM/dt {4:.1f}/{5}'
                        ' with image {6}x{7} (uvres {8}) with gpu {9}'
                        .format(len(integrations), minint, maxint, segment,
                                st.dmarr[dmind], st.dtarr[dtind], st.npixx,
                                st.npixy, st.uvres, devicenum))

            for i in integrations:
                # grid and FFT
                grid.operate(vis_raw, vis_grid, i)
                image.operate(vis_grid, img_grid)

                # calc snr
                stats = image.stats(img_grid)
                try:
                    peak_snr = stats['max']/stats['rms']
                except ZeroDivisionError:
                    peak_snr = 0.

                # threshold image on GPU and optionally save it
                if peak_snr > st.prefs.sigma_image1:
                    img_grid.d2h()
                    img_data = np.fft.fftshift(img_grid.data)  # shift zero pixel in middle
                    l, m = st.pixtolm(np.where(img_data == img_data.max()))
                    candloc = (segment, i, dmind, dtind, beamnum)

                    logger.info("Got one! SNR {0:.1f} candidate at {1} and (l,m) = ({2},{3})"
                                .format(peak_snr, candloc, l, m))

                    data_corr = dedisperseresample(data, delay,
                                                   st.dtarr[dtind],
                                                   parallel=st.prefs.nthread > 1)
                    data_corr = data_corr[max(0, i-st.prefs.timewindow//2):
                                          min(i+st.prefs.timewindow//2,
                                          len(data))]
                    util.phase_shift(data_corr, uvw, l, m)
                    data_corr = data_corr.mean(axis=1)
                    canddatalist.append(candidates.CandData(state=st,
                                                            loc=candloc,
                                                            image=img_data,
                                                            data=data_corr))

                    # TODO: add safety against triggering return of all images
                    if len(canddatalist)*bytespercd/1000**3 > st.prefs.memory_limit:
                        logger.info("Accumulated CandData size is {0:.1f} GB, "
                                    "which exceeds memory limit of {1:.1f}. "
                                    "Running calc_features..."
                                    .format(len(canddatalist)*bytespercd/1000**3,
                                            st.prefs.memory_limit))
                        candcollection += candidates.calc_features(canddatalist)
                        canddatalist = []

    candcollection += candidates.calc_features(canddatalist)

    logger.info("{0} candidates returned for seg {1}"
                .format(len(candcollection), segment))

    return candcollection


def dedisperse_image_fftw(st, segment, data, wisdom=None, integrations=None):
    """ Fuse the dediserpse, resample, search, threshold functions.
    """

    candcollection = candidates.CandCollection(prefs=st.prefs,
                                               metadata=st.metadata)
    for dtind in range(len(st.dtarr)):
        for dmind in range(len(st.dmarr)):
            delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind],
                                    st.inttime)

            data_corr = dedisperseresample(data, delay, st.dtarr[dtind],
                                           parallel=st.prefs.nthread > 1)

            candcollection += search_thresh_fftw(st, segment, data_corr, dmind,
                                                 dtind, wisdom=wisdom,
                                                 integrations=integrations)

    logger.info("{0} candidates returned for seg {1}"
                .format(len(candcollection), segment))

    return candcollection


def search_thresh_fftw(st, segment, data, dmind, dtind, integrations=None,
                       beamnum=0, wisdom=None):
    """ Take dedispersed, resampled data, image with fftw and threshold.
    Returns list of CandData objects that define candidates with
    candloc, image, and phased visibility data.
    Integrations can define subset of all available in data to search.
    Default will take integrations not searched in neighboring segments.

    ** only supports threshold > image max (no min)
    ** dmind, dtind, beamnum assumed to represent current state of data
    """

    if not np.any(data):
        logger.info("Data is all zeros. Skipping search.")
        return []

    bytespercd = 8*(st.npixx*st.npixy + st.prefs.timewindow*st.nchan*st.npol)

    # assumes dedispersed/resampled data has only back end trimmed off
    if integrations is None:
        integrations = st.get_search_ints(segment, dmind, dtind)
    elif isinstance(integrations, int):
        integrations = [integrations]

    assert isinstance(integrations, list), ("integrations should be int, list "
                                            "of ints, or None.")
    minint = min(integrations)
    maxint = max(integrations)

    logger.info('Imaging {0} ints ({1}-{2}) in seg {3} at DM/dt {4:.1f}/{5} with '
                'image {6}x{7} (uvres {8}) with fftw'
                .format(len(integrations), minint, maxint, segment,
                        st.dmarr[dmind], st.dtarr[dtind], st.npixx, st.npixy,
                        st.uvres))

    uvw = util.get_uvw_segment(st, segment)

    if 'image1' in st.prefs.searchtype:
        images = grid_image(data, uvw, st.npixx, st.npixy, st.uvres,
                            'fftw', st.prefs.nthread, wisdom=wisdom,
                            integrations=integrations)

        logger.debug('Thresholding at {0} sigma.'
                     .format(st.prefs.sigma_image1))

        # TODO: the following is really slow
        canddatalist = []
        candcollection = candidates.CandCollection(prefs=st.prefs,
                                                   metadata=st.metadata)
        for i in range(len(images)):
            peak_snr = images[i].max()/util.madtostd(images[i])
            if peak_snr > st.prefs.sigma_image1:
                candloc = (segment, integrations[i], dmind, dtind, beamnum)
                candim = images[i]
                l, m = st.pixtolm(np.where(candim == candim.max()))
                logger.info("Got one! SNR {0:.1f} candidate at {1} and (l,m) = ({2},{3})"
                            .format(peak_snr, candloc, l, m))
                util.phase_shift(data, uvw, l, m)
                dataph = data[max(0, integrations[i]-st.prefs.timewindow//2):
                              min(integrations[i]+st.prefs.timewindow//2, len(data))].mean(axis=1)
                util.phase_shift(data, uvw, -l, -m)
                canddatalist.append(candidates.CandData(state=st, loc=candloc,
                                                        image=candim,
                                                        data=dataph))

                if len(canddatalist)*bytespercd/1000**3 > st.prefs.memory_limit:
                    logger.info("Accumulated CandData size is {0:.1f} GB, "
                                "which exceeds memory limit of {1:.1f}. "
                                "Running calc_features..."
                                .format(len(canddatalist)*bytespercd/1000**3,
                                        st.prefs.memory_limit))
                    candcollection += candidates.calc_features(canddatalist)
                    canddatalist = []

        candcollection += candidates.calc_features(canddatalist)

    else:
        raise NotImplemented("only searchtype=image1 implemented")

    logger.info("{0} candidates returned for (seg, dmind, dtind) = "
                "({1}, {2}, {3})".format(len(candcollection), segment, dmind,
                                         dtind))

    return candcollection


def grid_image(data, uvw, npixx, npixy, uvres, fftmode, nthread, wisdom=None,
               integrations=None):
    """ Grid and image data.
    Optionally image integrations in list i.
    fftmode can be fftw or cuda.
    nthread is number of threads to use
    """

    if integrations is None:
        integrations = list(range(len(data)))
    elif isinstance(integrations, int):
        integrations = [integrations]

    if fftmode == 'fftw':
        logger.debug("Imaging with fftw on {0} threads".format(nthread))
        grids = grid_visibilities(data.take(integrations, axis=0), uvw, npixx,
                                  npixy, uvres, parallel=nthread > 1)
        images = image_fftw(grids, nthread=nthread, wisdom=wisdom)
    elif fftmode == 'cuda':
        logger.warn("Imaging with cuda not yet supported.")
        images = image_cuda()
    else:
        logger.warn("Imaging fftmode {0} not supported.".format(fftmode))

    return images


def image_cuda():
    """ Run grid and image with rfgpu
    TODO: update to use rfgpu
    """

    pass


def image_fftw(grids, nthread=1, wisdom=None):
    """ Plan pyfftw ifft2 and run it on uv grids (time, npixx, npixy)
    Returns time images.
    """

    if wisdom:
        logger.debug('Importing wisdom...')
        pyfftw.import_wisdom(wisdom)

    logger.debug("Starting pyfftw ifft2")
    images = np.zeros_like(grids)

#    images = pyfftw.interfaces.numpy_fft.ifft2(grids, auto_align_input=True,
#                                               auto_contiguous=True,
#                                               planner_effort='FFTW_MEASURE',
#                                               overwrite_input=True,
#                                               threads=nthread)
#    nints, npixx, npixy = images.shape
#
#   return np.fft.fftshift(images.real, (npixx//2, npixy//2))

    fft_obj = pyfftw.FFTW(grids, images, axes=(1, 2), direction="FFTW_BACKWARD")
    fft_obj.execute()

    logger.debug('Recentering fft\'d images...')

    return np.fft.fftshift(images.real, axes=(1, 2))


def grid_visibilities(data, uvw, npixx, npixy, uvres, parallel=False):
    """ Grid visibilities into rounded uv coordinates """

    logger.debug('Gridding {0} ints at ({1}, {2}) pix and {3} '
                 'resolution in {4} mode.'.format(len(data), npixx, npixy,
                                                  uvres,
                                                  ['single', 'parallel'][parallel]))
    u, v, w = uvw
    grids = np.zeros(shape=(data.shape[0], npixx, npixy),
                     dtype=np.complex64)

    if parallel:
        _ = _grid_visibilities_gu(data, u, v, w, npixx, npixy, uvres, grids)
    else:
        _grid_visibilities_jit(data, u, v, w, npixx, npixy, uvres, grids)

    return grids


@jit(nogil=True, nopython=True)
def _grid_visibilities_jit(data, u, v, w, npixx, npixy, uvres, grids):
    b""" Grid visibilities into rounded uv coordinates using jit on single core.
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
                    for l in range(npol):
                        grids[i, umod, vmod] += data[i, j, k, l]

    return grids


@guvectorize([str("void(complex64[:,:,:], float32[:,:], float32[:,:], float32[:,:], int64, int64, int64, complex64[:,:])")],
             str("(n,m,l),(n,m),(n,m),(n,m),(),(),(),(o,p)"),
             target='parallel', nopython=True)
def _grid_visibilities_gu(data, us, vs, ws, npixx, npixy, uvres, grid):
    b""" Grid visibilities into rounded uv coordinates for multiple cores"""

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


def image_arm():
    """ Takes visibilities and images arms of VLA """

    pass


def set_wisdom(npixx, npixy):
    """ Run single 2d ifft like image to prep fftw wisdom in worker cache """

    logger.info('Calculating FFT wisdom...')
    arr = pyfftw.empty_aligned((npixx, npixy), dtype='complex64', n=16)
    fft_arr = pyfftw.interfaces.numpy_fft.ifft2(arr, auto_align_input=True,
                                                auto_contiguous=True,
                                                planner_effort='FFTW_MEASURE')
    return pyfftw.export_wisdom()
