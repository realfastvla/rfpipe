from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
from numba import jit, guvectorize, int64
import pyfftw
from rfpipe import util, candidates, source
import scipy.stats

import logging
logger = logging.getLogger(__name__)

try:
    import rfgpu
except ImportError:
    pass


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

    data = source.data_prep(st, segment, data)

    if st.prefs.fftmode == "cuda":
        candcollection = dedisperse_image_cuda(st, segment, data)
    elif st.prefs.fftmode == "fftw":
        candcollection = dedisperse_image_fftw(st, segment, data)
    else:
        logger.warn("fftmode {0} not recognized (cuda, fftw allowed)"
                    .format(st.prefs.fftmode))

    candidates.save_cands(st, candcollection=candcollection)

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

    if not np.any(data):
        logger.info("Data is all zeros. Skipping search.")
        return candidates.CandCollection(prefs=st.prefs,
                                         metadata=st.metadata)

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

    candcollection = candidates.CandCollection(prefs=st.prefs,
                                               metadata=st.metadata)

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

    # some prep if kalman filter is to be applied
    if st.prefs.searchtype in ['imagek']:
        # TODO: check that this is ok if pointing at bright source
        spec_std = data.real.mean(axis=1).mean(axis=2).std(axis=0)
        sig_ts, kalman_coeffs = kalman_prepare_coeffs(spec_std)

    canddatalist = []
    for dtind in range(len(st.dtarr)):
        for dmind in range(len(st.dmarr)):
            delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind],
                                    st.inttime)

            if dtind > 0:
                grid.downsample(vis_raw)

            grid.set_shift(delay >> dtind)  # dispersion shift per chan in samples

            integrations = st.get_search_ints(segment, dmind, dtind)
            if len(integrations) == 0:
                continue
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
                if stats['rms'] != 0.:
                    peak_snr = stats['max']/stats['rms']
                    # TODO: add lm calc to rfgpu
                    # l, m = stats['peak_l'], stats['peak_m']
                else:
                    peak_snr = 0.

                # TODO: collect candlocs/snr and close loops here to find peaks
                #       then return to create canddata per peak or island

                # threshold image on GPU and optionally save it
                if peak_snr > st.prefs.sigma_image1:
                    img_grid.d2h()  # TODO: implement l,m and phasing on GPU
                    img_data = np.fft.fftshift(img_grid.data)  # shift zero pixel in middle
                    l, m = st.pixtolm(np.where(img_data == img_data.max()))
                    candloc = (segment, i, dmind, dtind, beamnum)
                    data_corr = dedisperseresample(data, delay,
                                                   st.dtarr[dtind],
                                                   parallel=st.prefs.nthread > 1)

                    if st.prefs.searchtype == 'image':
                        logger.info("Got one! SNR {0:.1f} candidate at {1} and (l,m) = ({2},{3})"
                                    .format(peak_snr, candloc, l, m))

                        data_corr = data_corr[max(0, i-st.prefs.timewindow//2):
                                              min(i+st.prefs.timewindow//2,
                                              len(data))]
                        util.phase_shift(data_corr, uvw, l, m)
                        data_corr = data_corr.mean(axis=1)
                        canddatalist.append(candidates.CandData(state=st,
                                                                loc=candloc,
                                                                image=img_data,
                                                                data=data_corr))

                    elif st.prefs.searchtype == 'imagek':
                        spec = data_corr.take([i], axis=0)
                        util.phase_shift(spec, uvw, l, m)
                        spec = spec[0].real.mean(axis=2).mean(axis=0)

                        # TODO: this significance can be biased low if averaging in long baselines that are not phased well
                        # TODO: spec should be calculated from baselines used to measure l,m?
                        significance_kalman = kalman_significance(spec,
                                                                  spec_std,
                                                                  sig_ts=sig_ts,
                                                                  coeffs=kalman_coeffs)
                        snrk = (2*significance_kalman)**0.5
                        snrtot = (snrk**2 + peak_snr**2)**0.5
                        if snrtot > (st.prefs.sigma_kalman**2 + st.prefs.sigma_image1**2)**0.5:
                            logger.info("Got one! SNR1 {0:.1f} and SNRk {1:.1f} candidate at {2} and (l,m) = ({3},{4})"
                                        .format(peak_snr, snrk, candloc, l, m))

                            data_corr = data_corr[max(0, i-st.prefs.timewindow//2):
                                                  min(i+st.prefs.timewindow//2,
                                                  len(data))]
                            util.phase_shift(data_corr, uvw, l, m)
                            data_corr = data_corr.mean(axis=1)
                            canddatalist.append(candidates.CandData(state=st,
                                                                    loc=candloc,
                                                                    image=img_data,
                                                                    data=data_corr,
                                                                    snrk=snrk))
                    elif st.prefs.searchtype == 'armkimage':
                        raise NotImplementedError
                    elif st.prefs.searchtype == 'armk':
                        raise NotImplementedError
                    else:
                        logger.warn("searchtype {0} not recognized"
                                    .format(st.prefs.searchtype))

                    # TODO: add safety against triggering return of all images
                    if len(canddatalist)*bytespercd/1000**3 > st.prefs.memory_limit:
                        logger.info("Accumulated CandData size exceeds "
                                    "memory limit of {0:.1f}. "
                                    "Running calc_features..."
                                    .format(st.prefs.memory_limit))
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

    candcollection = candidates.CandCollection(prefs=st.prefs,
                                               metadata=st.metadata)

    if not np.any(data):
        logger.info("Data is all zeros. Skipping search.")
        return candcollection

    bytespercd = 8*(st.npixx*st.npixy + st.prefs.timewindow*st.nchan*st.npol)

    # assumes dedispersed/resampled data has only back end trimmed off
    if integrations is None:
        integrations = st.get_search_ints(segment, dmind, dtind)
    elif isinstance(integrations, int):
        integrations = [integrations]

    assert isinstance(integrations, list), ("integrations should be int, list "
                                            "of ints, or None.")
    if len(integrations) == 0:
        return candcollection
    minint = min(integrations)
    maxint = max(integrations)

    # some prep if kalman filter is to be applied
    if st.prefs.searchtype in ['imagek', 'armkimage']:
        # TODO: check that this is ok if pointing at bright source
        spec_std = data.real.mean(axis=1).mean(axis=2).std(axis=0)
        sig_ts, kalman_coeffs = kalman_prepare_coeffs(spec_std)

    # TODO: add check that manually set integrations are safe for given dt

    logger.info('{0} search of {1} ints ({2}-{3}) in seg {4} at DM/dt {5:.1f}/{6} with '
                'image {7}x{8} (uvres {9}) with fftw'
                .format(st.prefs.searchtype, len(integrations), minint, maxint,
                        segment, st.dmarr[dmind], st.dtarr[dtind], st.npixx,
                        st.npixy, st.uvres))

    uvw = util.get_uvw_segment(st, segment)

    if st.prefs.searchtype in ['image', 'imagek']:
        images = grid_image(data, uvw, st.npixx, st.npixy, st.uvres,
                            'fftw', st.prefs.nthread, wisdom=wisdom,
                            integrations=integrations)

        canddatalist = []
        for i, image in enumerate(images):
            candloc = (segment, integrations[i], dmind, dtind, beamnum)
#            peak_snr = image.max()/util.madtostd(image)
            peak_snr = image.max()/image.std()
            if peak_snr > st.prefs.sigma_image1:
                l, m = st.pixtolm(np.where(image == image.max()))

                # if set, use sigma_kalman as second stage filter
                if st.prefs.searchtype == 'imagek':
                    spec = data.take([integrations[i]], axis=0)
                    util.phase_shift(spec, uvw, l, m)
                    spec = spec[0].real.mean(axis=2).mean(axis=0)
                    # TODO: this significance can be biased low if averaging in long baselines that are not phased well
                    # TODO: spec should be calculated from baselines used to measure l,m?
                    significance_kalman = kalman_significance(spec, spec_std,
                                                              sig_ts=sig_ts,
                                                              coeffs=kalman_coeffs)
                    snrk = (2*significance_kalman)**0.5
                    snrtot = (snrk**2 + peak_snr**2)**0.5
                    if snrtot > (st.prefs.sigma_kalman**2 + st.prefs.sigma_image1**2)**0.5:
                        logger.info("Got one! SNR1 {0:.1f} and SNRk {1:.1f} candidate at {2} and (l,m) = ({3},{4})"
                                    .format(peak_snr, snrk, candloc, l, m))
                        dataph = data[max(0, integrations[i]-st.prefs.timewindow//2):
                                      min(integrations[i]+st.prefs.timewindow//2, len(data))].copy()
                        util.phase_shift(dataph, uvw, l, m)
                        dataph = dataph.mean(axis=1)
                        canddatalist.append(candidates.CandData(state=st,
                                                                loc=candloc,
                                                                image=image,
                                                                data=dataph,
                                                                snrk=snrk))
                elif st.prefs.searchtype == 'image':
                    logger.info("Got one! SNR1 {0:.1f} candidate at {1} and (l, m) = ({2},{3})"
                                .format(peak_snr, candloc, l, m))
                    dataph = data[max(0, integrations[i]-st.prefs.timewindow//2):
                                  min(integrations[i]+st.prefs.timewindow//2, len(data))].copy()
                    util.phase_shift(dataph, uvw, l, m)
                    dataph = dataph.mean(axis=1)
                    canddatalist.append(candidates.CandData(state=st,
                                                            loc=candloc,
                                                            image=image,
                                                            data=dataph))
                else:
                    logger.warn("searchtype {0} not recognized"
                                .format(st.prefs.searchtype))

                if len(canddatalist)*bytespercd/1000**3 > st.prefs.memory_limit:
                    logger.info("Accumulated CandData size is {0:.1f} GB, "
                                "which exceeds memory limit of {1:.1f}. "
                                "Running calc_features..."
                                .format(len(canddatalist)*bytespercd/1000**3,
                                        st.prefs.memory_limit))
                    candcollection += candidates.calc_features(canddatalist)
                    canddatalist = []

        candcollection += candidates.calc_features(canddatalist)

    elif st.prefs.searchtype in ['armkimage', 'armk']:
        armk_candidates = search_thresh_armk(st, data, uvw,
                                             integrations=integrations,
                                             spec_std=spec_std,
                                             sig_ts=sig_ts,
                                             coeffs=kalman_coeffs)
        canddatalist = []
#        for candind, snrarm, snrk in armk_candidates:
        for candind, snrarm, snrk, armloc, peakxy, lm in armk_candidates:
            candloc = (segment, candind, dmind, dtind, beamnum)
            # if set, use sigma_kalman as second stage filter
            if st.prefs.searchtype == 'armkimage':
                image = grid_image(data, uvw, st.npixx_full, st.npixy_full,
                                   st.uvres, 'fftw', st.prefs.nthread,
                                   wisdom=wisdom, integrations=candind)
                peakx, peaky = np.where(image[0] == image[0].max())
                l, m = st.calclm(st.npixx_full, st.npixy_full, st.uvres,
                                 peakx[0], peaky[0])
#                peak_snr = image.max()/util.madtostd(image)
                peak_snr = image.max()/image.std()
                if peak_snr > st.prefs.sigma_image1:
                    logger.info("Got one! SNRarm {0:.1f} and SNRk {1:.1f} and SNR1 {2:.1f} candidate at {3} and (l,m) = ({4},{5})"
                                .format(snrarm, snrk, peak_snr, candloc, l, m))
                    dataph = data[max(0, candind-st.prefs.timewindow//2):
                                  min(candind+st.prefs.timewindow//2, len(data))].copy()
                    util.phase_shift(dataph, uvw, l, m)
                    dataph = dataph.mean(axis=1)
                    canddatalist.append(candidates.CandData(state=st,
                                                            loc=candloc,
                                                            image=image[0],
                                                            data=dataph,
                                                            snrarm=snrarm,
                                                            snrk=snrk))
            elif st.prefs.searchtype == 'armk':
                raise NotImplementedError
            else:
                logger.warn("searchtype {0} not recognized"
                            .format(st.prefs.searchtype))

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
        raise NotImplemented("only searchtype=image, imagek, armkimage implemented")

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


def image_fftw(grids, nthread=1, wisdom=None, axes=(1, 2)):
    """ Plan pyfftw inverse fft and run it on input grids.
    Allows fft on 1d (time, npix) or 2d (time, npixx, npixy) grids.
    axes refers to dimensions of fft, so (1, 2) will do 2d fft on
    last two axes of (time, npixx, nipxy) data, while (1) will do
    1d fft on last axis of (time, npix) data.
    Returns recentered fftoutput for each integration.
    """

    if wisdom is not None:
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

    fft_obj = pyfftw.FFTW(grids, images, axes=axes, direction="FFTW_BACKWARD")
    fft_obj.execute()

    logger.debug('Recentering fft output...')

    return np.fft.fftshift(images.real, axes=axes)


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


###
# cascading 3arm imaging with kalman filter
###

def search_thresh_armk(st, data, uvw, integrations=None, spec_std=None,
                       sig_ts=[], coeffs=[]):
    """
    """

    if integrations is None:
        integrations = list(range(len(data)))
    elif isinstance(integrations, int):
        integrations = [integrations]

    if spec_std is None:
        spec_std = data.real.mean(axis=1).mean(axis=2).std(axis=0)

    if not len(sig_ts):
        sig_ts = [x*np.median(spec_std) for x in [0.3, 0.1, 0.03, 0.01]]

    if not len(coeffs):
        sig_ts, coeffs = kalman_prepare_coeffs(spec_std, sig_ts)

    n_max_cands = 10  # TODO set with function of sigma_arms

    u, v, w = uvw
    ch0 = 0
    u0 = u[:, ch0]
    v0 = v[:, ch0]
    w0 = w[:, ch0]

    order = ['N', 'E', 'W']
    T012 = maparms(st=st, u0=u0, v0=v0, order=order)
    arm0, arm1, arm2 = image_arms(st, data.take(integrations, axis=0), uvw,
                                  order=order)
    candinds, armlocs, snrarms = thresh_arms(arm0, arm1, arm2, T012,
                                             st.prefs.sigma_arm,
                                             st.prefs.sigma_arms,
                                             n_max_cands)

    # kalman filter integrated for now
    T01U = maparms(st=st, u0=u0, v0=v0, order=[order[0], order[1]],
                   e2=(1., 0.))
    T01V = maparms(st=st, u0=u0, v0=v0, order=[order[0], order[1]],
                   e2=(0., 1.))
    T12U = maparms(st=st, u0=u0, v0=v0, order=[order[1], order[2]],
                   e2=(1., 0.))
    T12V = maparms(st=st, u0=u0, v0=v0, order=[order[1], order[2]],
                   e2=(0., 1.))
    T20U = maparms(st=st, u0=u0, v0=v0, order=[order[2], order[0]],
                   e2=(1., 0.))
    T20V = maparms(st=st, u0=u0, v0=v0, order=[order[2], order[0]],
                   e2=(0., 1.))
    npix = max(st.npixx_full, st.npixy_full)
    kpeaks = []
    for i in range(len(candinds)):
        kpeak = ()
        snrlast = 0.  # initialize snr to find max per i
        for j in range(n_max_cands):
            if snrarms[i, j] > 0.:
                spec = data.take([integrations[candinds[i, j]]], axis=0).copy()
                armloc0, armloc1, armloc2 = armlocs[i, j]

                # find x,y loc from common loc inferred from each arm pair
                peakx01 = projectarms(armloc0-npix//2, armloc1-npix//2, T01U,
                                      st.npixx_full)
                peaky01 = projectarms(armloc0-npix//2, armloc1-npix//2, T01V,
                                      st.npixy_full)
                peakx12 = projectarms(armloc1-npix//2, armloc2-npix//2, T12U,
                                      st.npixx_full)
                peaky12 = projectarms(armloc1-npix//2, armloc2-npix//2, T12V,
                                      st.npixy_full)
                peakx20 = projectarms(armloc2-npix//2, armloc0-npix//2, T20U,
                                      st.npixx_full)
                peaky20 = projectarms(armloc2-npix//2, armloc0-npix//2, T20V,
                                      st.npixy_full)
                peakx = np.sort([peakx01, peakx12, peakx20])[1]
                peaky = np.sort([peaky01, peaky12, peaky20])[1]
                l, m = st.calclm(st.npixx_full, st.npixy_full, st.uvres, peakx,
                                 peaky)
                util.phase_shift(spec, uvw, l, m)
                spec = spec[0].real.mean(axis=2).mean(axis=0)
                significance_kalman = kalman_significance(spec, spec_std,
                                                          sig_ts=sig_ts,
                                                          coeffs=coeffs)
                snrk = (2*significance_kalman)**0.5
                snrtot = (snrk**2 + snrarms[i, j]**2)**0.5
                if (snrtot > (st.prefs.sigma_kalman**2 + st.prefs.sigma_arm**2)**0.5) and (snrk > snrlast):
                    kpeak = (integrations[candinds[i, j]], snrarms[i, j],
                             snrk, (armloc0, armloc1, armloc2), (peakx, peaky),
                             (l, m))
                    snrlast = snrtot
        if len(kpeak):
            kpeaks.append(kpeak)

    return kpeaks


def image_arms(st, data, uvw, wisdom=None, order=['N', 'E', 'W']):
    """ Calculate grids for all three arms of VLA.
    Uses maximum of ideal number of pixels on side of image.
    """

    npix = max(st.npixx_full, st.npixy_full)

    # TODO: check if there is a center ant that can be counted in all arms
    grids_arm0 = grid_arm(data, uvw, st.blind_arm(order[0]), npix, st.uvres)
    arm0 = image_fftw(grids_arm0, axes=(1,), wisdom=wisdom)

    grids_arm1 = grid_arm(data, uvw, st.blind_arm(order[1]), npix, st.uvres)
    arm1 = image_fftw(grids_arm1, axes=(1,), wisdom=wisdom)

    grids_arm2 = grid_arm(data, uvw, st.blind_arm(order[2]), npix, st.uvres)
    arm2 = image_fftw(grids_arm2, axes=(1,), wisdom=wisdom)

    return arm0, arm1, arm2


def grid_arm(data, uvw, arminds, npix, uvres):
    """ Grids visibilities along 1d arms of array.
    arminds defines a subset of baselines that for a linear array.
    grids as radius with sign of the u coordinate.
    defines a convention of uv distance as positive in u direction.
    Returns FFT output (time vs pixel) from gridded 1d visibilities.
    """

    u, v, w = uvw
    # TODO: check colinearity and "w"
    # TODO: integrate with unit vector approach in mapper function?
    sign = np.sign(u.take(arminds, axis=0))
    uvd = sign*(u.take(arminds, axis=0)**2 + v.take(arminds, axis=0)**2)**0.5

    grids = np.zeros(shape=(data.shape[0], npix), dtype=np.complex64)
    grid_visibilities_arm_jit(data.take(arminds, axis=1), uvd, npix,
                              uvres, grids)

    return grids


def maparms(st=None, u0=None, v0=None, e0=None, e1=None, e2=None,
            order=['N', 'E', 'W']):
    """ Generates a function for geometric mapping between three unit vectors.
    0,1,2 indiced are marking the order of the vectors.
    They can be measured with (st, u0, v0) or given with e0, e1, e2.
    dot(T012,(A0,A1)) = A2, where A0,A1 are locations on arms 0,1
    and A2 is the location on arm 2.
    Convention defined in gridding for vectors to be positive in u direction.
    u,v are 1d of length nbl chosen at channel 0
    order can be arm names N, E, W
    """

    assert all([o in ['N', 'E', 'W'] for o in order])

    if e0 is None:
        e0 = get_uvunit(st.blind_arm(order[0]), u0, v0)
    if e1 is None:
        e1 = get_uvunit(st.blind_arm(order[1]), u0, v0)
    if e2 is None:
        e2 = get_uvunit(st.blind_arm(order[2]), u0, v0)

    # they should be unit vectors (within rounding errors)
    assert np.linalg.norm(e0) > 0.99, "Problem with unit vector"
    assert np.linalg.norm(e1) > 0.99, "Problem with unit vector"
    assert np.linalg.norm(e2) > 0.99, "Problem with unit vector"

    T012 = np.dot(e2, np.linalg.inv(np.array((e0, e1))))
    return T012


def get_uvunit(blind, u, v):
    """ Calculate uv unit vector for indices blind of u/v.
    """
    # positive u convention

    ind = blind[np.argmax(u.take(blind, axis=0)**2 + v.take(blind, axis=0)**2)]
    l = (u[ind]**2 + v[ind]**2)**0.5
    e = (u[ind]/l * np.sign(u[ind]), v[ind]/l * np.sign(u[ind]))

    return e


@jit(nopython=True)
def projectarms(dpix0, dpix1, T012, npix2):
    """ Take any two locations relative to center and project in a new direction.
    npix2 is size of direction2.
    """

    return int(round(np.dot(np.array([float(dpix0), float(dpix1)]),
                            T012) + npix2//2))


@jit(nopython=True)
def thresh_arms(arm0, arm1, arm2, T012, sigma_arm, sigma_trigger, n_max_cands):
    """ Run 3-arm search with sigma_arm per arm and sigma_trigger overall.
    arm0/1/2 are the 1d arm "images" and T012 is the coefficients to map arm0/1
    positions to arm2.
    Number of candidates is limit to n_max_cands per integration.
    Highest snrarm candidates returned up to n_max_cands per integration.
    """

    assert len(arm0[0]) == len(arm1[0])
    assert len(arm2[0]) == len(arm1[0])

    # TODO: assure stds is calculated over larger sample than 1 int
    std_arm0 = arm0.std()  # over all ints and pixels
    std_arm1 = arm1.std()
    std_arm2 = arm2.std()

    nint = len(arm0)
    npix = len(arm0[0])

    effective_3arm_sigma = (std_arm0**2 + std_arm1**2 + std_arm2**2)**0.5
    effective_eta_trigger = sigma_trigger * effective_3arm_sigma

    candinds = np.zeros(shape=(nint, n_max_cands), dtype=np.int64)
    armlocs = np.zeros(shape=(nint, n_max_cands, 3), dtype=np.int64)
    snrarms = np.zeros(shape=(nint, n_max_cands), dtype=np.float64)
    for i in range(len(arm0)):
        success_counter = 0
        indices_arr0 = np.nonzero(arm0[i] > sigma_arm*std_arm0)[0]
        indices_arr1 = np.nonzero(arm1[i] > sigma_arm*std_arm1)[0]
        for ind0 in indices_arr0:
            for ind1 in indices_arr1:
                ind2 = projectarms(ind0-npix//2, ind1-npix//2, T012, npix)
                score = arm0[i, ind0] + arm1[i, ind1] + arm2[i, ind2]
                if score > effective_eta_trigger:
                    snr_3arm = score/effective_3arm_sigma

                    # TODO find better logic (heap?)
                    success_counter0 = success_counter
                    while snrarms[i, success_counter] > snr_3arm:
                        success_counter += 1
                        if success_counter >= n_max_cands:
                            success_counter = 0
                        if success_counter == success_counter0:
                            break
                    if snrarms[i, success_counter] < snr_3arm:
                        snrarms[i, success_counter] = snr_3arm
                        armlocs[i, success_counter] = (ind0, ind1, ind2)
                        candinds[i, success_counter] = i
                        success_counter += 1
                        if success_counter >= n_max_cands:
                            success_counter = 0

    return candinds, armlocs, snrarms


@jit(nogil=True, nopython=True)
def grid_visibilities_arm_jit(data, uvd, npix, uvres, grids):
    b""" Grid visibilities into rounded uvd coordinates using jit on single core.
    data/uvd are selected for a single arm
    """

    nint, nbl, nchan, npol = data.shape

# rounding not available in numba
#    ubl = np.round(us/uvres, 0).astype(np.int32)
#    vbl = np.round(vs/uvres, 0).astype(np.int32)

    for j in range(nbl):
        for k in range(nchan):
            uvbl = int64(uvd[j, k]/uvres)
            if (np.abs(uvbl < npix//2)):
                uvmod = int64(np.mod(uvbl, npix))
                for i in range(nint):
                    for l in range(npol):
                        grids[i, uvmod] += data[i, j, k, l]

    return grids


###
# Kalman significance
###

def kalman_significance(spec, spec_std, sig_ts=[], coeffs=[]):
    """ Calculate kalman significance for given 1d spec and per-channel error.
    If no coeffs input, it will calculate with random number generation.
    From Barak Zackay
    """

    if not len(sig_ts):
        sig_ts = [x*np.median(spec_std) for x in [0.3, 0.1, 0.03, 0.01]]
    if not len(coeffs):
        sig_ts, coeffs = kalman_prepare_coeffs(spec_std, sig_ts)

    assert len(sig_ts) == len(coeffs)
    logger.debug("Calculating max Kalman significance for {0} channel spectrum"
                 .format(len(spec)))

    significances = []
    for i, sig_t in enumerate(sig_ts):
        score = kalman_filter_detector(spec, spec_std, sig_t)
        coeff = coeffs[i]
        x_coeff, const_coeff = coeff
        significances.append(x_coeff * score + const_coeff)

    # return prob in units of nats. ignore negative probs
    return max(0, np.max(significances) * np.log(2))


@jit(nopython=True)
def kalman_filter_detector(spec, spec_std, sig_t, A_0=None, sig_0=None):
    """ Core calculation of Kalman estimator of input 1d spectrum data.
    spec/spec_std are 1d spectra in same units.
    sig_t sets the smoothness scale of model (A) change.
    Number of changes is sqrt(nchan)*sig_t/mean(spec_std).
    Frequency scale is 1/sig_t**2
    A_0/sig_0 are initial guesses of model value in first channel.
    Returns score, which is the likelihood of presence of signal.
    From Barak Zackay
    """

    if A_0 is None:
        A_0 = spec.mean()
    if sig_0 is None:
        sig_0 = np.median(spec_std)

    spec = spec - np.mean(spec)  # likelihood calc expects zero mean spec

    cur_mu, cur_state_v = A_0, sig_0**2
    cur_log_l = 0
    for i in range(len(spec)):
        cur_z = spec[i]
        cur_spec_v = spec_std[i]**2
        # computing consistency with the data
        cur_log_l += -(cur_z-cur_mu)**2 / (cur_state_v + cur_spec_v + sig_t**2)/2 - 0.5*np.log(2*np.pi*(cur_state_v + cur_spec_v + sig_t**2))

        # computing the best state estimate
        cur_mu = (cur_mu / cur_state_v + cur_z/cur_spec_v) / (1/cur_state_v + 1/cur_spec_v)
        cur_state_v = cur_spec_v * cur_state_v / (cur_spec_v + cur_state_v) + sig_t**2

    H_0_log_likelihood = -np.sum(spec**2 / spec_std**2 / 2) - np.sum(0.5*np.log(2*np.pi * spec_std**2))
    return cur_log_l - H_0_log_likelihood


def kalman_prepare_coeffs(data_std, sig_ts=None, n_trial=10000):
    """ Measure kalman significance distribution in random data.
    data_std is the noise vector per channel.
    sig_ts can be single float or list of values.
    returns tuple (sig_ts, coeffs)
    From Barak Zackay
    """

    if sig_ts is None:
        sig_ts = np.array([x*np.median(data_std) for x in [0.3, 0.1, 0.03, 0.01]])
    elif isinstance(sig_ts, float):
        sig_ts = np.array([sig_ts])
    elif isinstance(sig_ts, list):
        sig_ts = np.array(sig_ts)
    else:
        logger.warn("Not sure what to do with sig_ts {0}".format(sig_ts))

    assert isinstance(sig_ts, np.ndarray)

    logger.info("Measuring Kalman significance distribution for sig_ts {0}".format(sig_ts))

    coeffs = []
    for sig_t in sig_ts:
        nchan = len(data_std)
        random_scores = []
        for i in range(n_trial):
            normaldist = np.random.normal(0, data_std, size=nchan)
            normaldist -= normaldist.mean()
            random_scores.append(kalman_filter_detector(normaldist, data_std, sig_t))

        # Approximating the tail of the distribution as an  exponential tail (probably is justified)
        coeffs.append(np.polyfit([np.percentile(random_scores, 100 * (1 - 2 ** (-i))) for i in range(3, 10)], range(3, 10), 1))
        # TODO: check distribution out to 1/1e6

    return sig_ts, coeffs


def kalman_significance_canddata(canddata, sig_ts=[]):
    """ Go from canddata to total ignificance with kalman significance
    Calculates coefficients from data and then adds significance to image snr.
    From Barak Zackay
    """

    # TODO check how to automate candidate selection of on/off integrations
    onint = 15
    offints = list(range(0, 10))+list(range(20, 30))
    spec_std = canddata.data.real.mean(axis=2).take(offints, axis=0).std(axis=0)
    spec = canddata.data.real.mean(axis=2)[onint]

    sig_ts, coeffs = kalman_prepare_coeffs(spec_std)
    significance_kalman = kalman_significance(spec, spec_std, sig_ts=sig_ts,
                                              coeffs=coeffs)
    snrk = (2*significance_kalman)**0.5

#    snr_image = canddata.image.max()/util.madtostd(canddata.image)
    snr_image = canddata.image.max()/canddata.image.std()
    significance_image = -scipy.stats.norm.logsf(snr_image)

    snr_total = (2*(snrk + significance_image))**0.5
    return snr_total


def set_wisdom(npixx, npixy=None):
    """ Run single ifft to prep fftw wisdom in worker cache
    Supports 1d and 2d ifft.
    """

    logger.info('Calculating FFT wisdom...')

    if npixy is not None:
        arr = pyfftw.empty_aligned((npixx, npixy), dtype='complex64', n=16)
        fft_arr = pyfftw.interfaces.numpy_fft.ifft2(arr, auto_align_input=True,
                                                    auto_contiguous=True,
                                                    planner_effort='FFTW_MEASURE')
    else:
        arr = pyfftw.empty_aligned((npixx), dtype='complex64', n=16)
        fft_arr = pyfftw.interfaces.numpy_fft.ifft(arr, auto_align_input=True,
                                                   auto_contiguous=True,
                                                   planner_effort='FFTW_MEASURE')
    return pyfftw.export_wisdom()
