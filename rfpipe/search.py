from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import os
import math
import pickle
import numpy as np
from numba import jit, guvectorize, vectorize, int32, int64
from collections import OrderedDict
import pandas as pd
import pyfftw
import matplotlib.pyplot as plt
from rfpipe import fileLock, util, version

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

    logger.info('Dedispersing up to delay shift of {0} integrations'
                .format(delay.max()))
    if mode == 'single':
        return _dedisperse_jit(np.require(data, requirements='W'), delay)
    elif mode == 'multi':
        _ = _dedisperse_gu(np.require(np.swapaxes(data, 0, 1), requirements='W'), delay)
        return data[:-delay.max()]
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


@guvectorize(["void(complex64[:,:,:], int64[:])"], '(n,m,l),(m)', target='parallel', nopython=True)
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

    len0 = data.shape[0]
    logger.info('Resampling data of length {0} by a factor of {1}'
                .format(len0, dt))

    if mode == 'single':
        return _resample_jit(np.require(data, requirements='W'), dt)
    elif mode == 'multi':
        _ = _resample_gu(np.require(np.swapaxes(data, 0, 3), requirements='W'), dt)
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


def search_thresh(st, data, segment, dmind, dtind, beamnum=0, wisdom=None):
    """ High-level wrapper for search algorithms.
    Expects dedispersed, resampled data as input and data state.
    Returns list of CandData objects that define candidates with
    candloc, image, and phased visibility data.

    ** only supports threshold > image max (no min)
    ** dmind, dtind, beamnum assumed to represent current state of data
    """

    if not np.any(data):
        return []

    logger.info('Imaging {0}x{1} pix with uvres of {2}.'
                .format(st.npixx, st.npixy, st.uvres))

    if 'image1' in st.prefs.searchtype:
        uvw = st.get_uvw_segment(segment)
        images = image(data, uvw, st.npixx, st.npixy,
                       st.uvres, wisdom=wisdom)

        logger.info('Thresholding images at {0} sigma.'
                    .format(st.prefs.sigma_image1))

        canddatalist = []
        for i in range(len(images)):
            if (images[i].max()/util.madtostd(images[i]) >
               st.prefs.sigma_image1):
                candloc = (segment, i, dmind, dtind, beamnum)
                candim = images[i]
                l, m = st.pixtolm(np.where(candim == candim.max()))
                phase_shift(data, uvw, l, m)
                dataph = data[max(0, i-st.prefs.timewindow//2):
                              min(i+st.prefs.timewindow//2, len(data))].mean(axis=1)
                phase_shift(data, uvw, -l, -m)
                canddatalist.append(CandData(state=st, loc=candloc,
                                             image=candim, data=dataph))
    else:
        raise NotImplemented("only searchtype=image1 implemented")

    # tuple(list(int), list(ndarray), list(ndarray))
#    return (ints, images_thresh, dataph)
    logger.info("Returning data for {0} candidates".format(len(canddatalist)))

    return canddatalist


def image(data, uvw, npixx, npixy, uvres, wisdom=None, integrations=None, mode='cuda'):
    """ Grid and image data.
    Optionally image integrations in list i.
    mode can be fftw or cuda.
    """

    if not integrations:
        integrations = range(len(data))
    else:
        assert isinstance(integrations, list)
        logger.info('Imaging int{0} {1}'
                    .format('s'[not len(integrations)-1:],
                            ','.join([str(i) for i in integrations])))

    grids = grid_visibilities(data.take(integrations, axis=0), uvw, npixx,
                              npixy, uvres)
    if mode == 'fftw':
        logger.info("Imaging with fftw.")
        images = image_fftw(grids, wisdom=wisdom)
    elif mode == 'cuda':
        logger.info("Imaging with cuda.")
        images = image_cuda(grids)
    else:
        logger.warn("Imaging mode {0} not supported.".format(mode))

    return images


def image_cuda(grids):
    """ Run 2d FFT to image each plane of grid array
    """

    from pyfft.cuda import Plan
    from pycuda.tools import make_default_context
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda

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


def grid_visibilities(data, uvw, npixx, npixy, uvres, mode='multi'):
    """ Grid visibilities into rounded uv coordinates """

    logger.info('Gridding visibilities for grid of ({0}, {1}) pix and {2} '
                'resolution.'.format(npixx, npixy, uvres))

    if mode == 'single':
        return _grid_visibilities_jit(np.require(data, requirements='W'), uvw,
                                      npixx, npixy, uvres)
    elif mode == 'multi':
        grid = np.zeros(shape=(data.shape[0], npixx, npixy),
                        dtype=np.complex64)
        u, v, w = uvw
        _ = _grid_visibilities_gu(np.require(data, requirements='W'), u, v, w,
                                  npixx, npixy, uvres, grid)
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
             '(n,m,l),(n,m),(n,m),(n,m),(),(),(),(o,p)', target='parallel', nopython=True)
def _grid_visibilities_gu(data, us, vs, ws, npixx, npixy, uvres, grid):
    """ Grid visibilities into rounded uv coordinates for multiple cores"""

    ubl = np.zeros(us.shape, dtype=int64)
    vbl = np.zeros(vs.shape, dtype=int64)

    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            ubl[j,k] = int64(np.round(us[j,k]/uvres, 0))
            vbl[j,k] = int64(np.round(vs[j,k]/uvres, 0))
            if (np.abs(ubl[j, k]) < npixx//2) and \
               (np.abs(vbl[j, k]) < npixy//2):
                u = np.mod(ubl[j, k], npixx)
                v = np.mod(vbl[j, k], npixy)
                for l in range(data.shape[2]):
                    grid[u, v] += data[j, k, l]


def grid_visibilities_image(st, data, segment):
    """ Grid and image using multiple cores"""

    if st.prefs.nthread == 1:
        mode = 'single'
    else:
        mode = 'multi'

    grid = grid_visibilities(data, st.get_uvw_segment(segment), st.npixx, st.npixy, st.uvres, mode=mode)

    _ = pyfftw.interfaces.numpy_fft.ifft2(grid, overwrite_input=True, 
                                          auto_align_input=True,
                                          auto_contiguous=True,
                                          planner_effort='FFTW_MEASURE',
                                          threads=st.prefs.nthread)

    return recenter(grid.real, (st.npixx//2, st.npixy//2))


def image_fftw(grids, wisdom=None):
    """ Plan pyfftw ifft2 and run it on uv grids (time, npixx, npixy)
    Returns time images.
    """

    if wisdom:
        logger.debug('Importing wisdom...')
        pyfftw.import_wisdom(wisdom)
    images = pyfftw.interfaces.numpy_fft.ifft2(grids, auto_align_input=True,
                                               auto_contiguous=True,
                                               planner_effort='FFTW_MEASURE')

    npixx, npixy = images[0].shape

    return recenter(images.real, (npixx//2, npixy//2))


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


@jit(nogil=True, nopython=True)
def phase_shift(data, uvw, dl, dm):
    """ Applies a phase shift to data for a given (dl, dm).
    """

    sh = data.shape
    u, v, w = uvw

    if (dl != 0.) or (dm != 0.):
        for j in xrange(sh[1]):
            for k in xrange(sh[2]):
                for i in xrange(sh[0]):    # iterate over pols
                    for l in xrange(sh[3]):
                        # phasor unwraps phase at (dl, dm) per (bl, chan)
                        frot = np.exp(-2j*np.pi*(dl*u[j, k] + dm*v[j, k]))
                        data[i, j, k, l] = data[i, j, k, l] * frot


#
# CUDA
#


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


#
# candidates and features
#


class CandData(object):
    """ Object that bundles data from search stage to candidate visualization.
    Provides some properties for the state of the phased data and candidate.
    """

    def __init__(self, state, loc, image, data):
        """ Instantiate with pipeline state, candidate location tuple,
        image, and resampled data phased to candidate.
        TODO: Need to use search_dimensions to infer candloc meaning
        """

        self.state = state
        self.loc = tuple(loc)
        self.image = image
        self.data = data

        assert len(loc) == len(state.search_dimensions), ("candidate location "
                                                          "should set each of "
                                                          "the st.search_dimensions")

    @property
    def peak_lm(self):
        """
        """

        return self.state.pixtolm(self.peak_xy)

    @property
    def peak_xy(self):
        """ Peak pixel in image
        Only supports positive peaks for now.
        """

        return np.where(self.image == self.image.max())

    @property
    def time_top(self):
        """ Time in mjd where burst is at top of band
        """

        return (self.state.metadata.starttime_mjd +
                (self.loc[1]*self.state.inttime)/(24*3600))

    @property
    def time_infinite(self):
        """ Time in mjd where burst is at infinite frequency
        """

        delay = util.calc_delay(1e5, self.time_top, self.dmarr[self.loc[2]],
                                self.state.inttime)
        return self.time_top - delay


def calc_features(canddatalist):
    """ Calculates the candidate features for CandData instance(s).
    Returns dictionary of candidate features with keys as defined in
    st.search_dimensions.
    """

    if not len(canddatalist):
        return {}

    if not isinstance(canddatalist, list):
        logger.debug('Wrapping solo CandData object')
        canddatalist = [canddatalist]

    logger.info('Calculating features for {0} candidates.'
                .format(len(canddatalist)))

    candidates = {}
    for i in xrange(len(canddatalist)):
        canddata = canddatalist[i]
        st = canddata.state
        candloc = canddata.loc
        image = canddata.image
        dataph = canddata.data

        # assemble feature in requested order
        ff = []
        for feat in st.features:
            if feat == 'snr1':
                imstd = util.madtostd(image)
                snrmax = image.max()/imstd
                snrmin = image.min()/imstd
                snr = snrmax if snrmax >= snrmin else snrmin
                ff.append(snr)
            elif feat == 'immax1':
                if snr > 0:
                    ff.append(image.max())
                else:
                    ff.append(image.min())
            elif feat == 'l1':
                l1, m1 = st.pixtolm(np.where(image == image.max()))
                ff.append(float(l1))
            elif feat == 'm1':
                l1, m1 = st.pixtolm(np.where(image == image.max()))
                ff.append(float(m1))
            else:
                print(feat)
                raise NotImplementedError("Feature {0} calculation not ready"
                                          .format(feat))

        candidates[candloc] = list(ff)

    return candidates


def save_cands(st, candidates, canddatalist):
    """ Save candidates in reproducible form.
    Saves as DataFrame with metadata and preferences attached.
    Writes to location defined by state using a file lock to allow multiple
    writers.
    """

    if st.prefs.savecands and len(candidates):
        logger.info('Saving {0} candidates to {1}.'.format(len(candidates),
                                                           st.candsfile))

        df = pd.DataFrame(OrderedDict(zip(st.search_dimensions,
                                          np.transpose(candidates.keys()))))
        df2 = pd.DataFrame(OrderedDict(zip(st.features,
                                           np.transpose(candidates.values()))))
        df3 = pd.concat([df, df2], axis=1)

        cdf = CandidateDF(df3, prefs=st.prefs, metadata=st.metadata)

        try:
            with fileLock.FileLock(st.candsfile+'.lock', timeout=10):
                with open(st.candsfile, 'ab+') as pkl:
                    pickle.dump(cdf, pkl)
        except fileLock.FileLock.FileLockException:
            scan = st.metadata.scan
            loc, prop = candidates.popitem()
            segment = loc[0]
            newcandsfile = ('{0}_sc{0}_seg{1}.pkl'
                            .format(st.candsfile.rstrip('.pkl'),
                                    scan, segment))
            logger.warn('Candidate file writing timeout. '
                        'Spilling to new file {0}.'.format(newcandsfile))
            with open(newcandsfile, 'ab+') as pkl:
                pickle.dump(cdf, pkl)

        if len(cdf.df):
            snrs = cdf.df['snr1'].values
            candplot(canddatalist, snrs=snrs)

        return st.candsfile
    elif st.prefs.savecands and not len(candidates):
        logger.info('No candidates to save to {0}.'.format(st.candsfile))
        return None
    elif not st.prefs.savecands:
        logger.info('Not saving candidates.')
        return None


def candplot(canddatalist, snrs=[], outname=''):
    """ Takes output of search_thresh (CandData objects) to make
    candidate plots.
    Expects pipeline state, candidate location, image, and
    phased, dedispersed data (cut out in time, dual-pol).

    snrs is array for an (optional) SNR histogram plot.
    Written by Bridget Andersen and modified by Casey for rfpipe.
    """

    if not isinstance(canddatalist, list):
        logger.debug('Wrapping solo CandData object')
        canddatalist = [canddatalist]

    logger.info('Making {0} candidate plots.'.format(len(canddatalist)))

    for i in xrange(len(canddatalist)):
        canddata = canddatalist[i]
        st = canddata.state
        candloc = canddata.loc
        im = canddata.image
        data = canddata.data

        logger.info('Plotting for (image, data) shapes: ({0}, {1})'
                    .format(str(im.shape), str(data.shape)))

        scan = st.metadata.scan
        segment, candint, dmind, dtind, beamnum = candloc

        # calc source location
        imstd = util.madtostd(im)
        snrmin = im.min()/imstd
        snrmax = im.max()/imstd
        if snrmax > -1*snrmin:
            l1, m1 = st.pixtolm(np.where(im == im.max()))
            snrobs = snrmax
        else:
            l1, m1 = st.pixtolm(np.where(im == im.min()))
            snrobs = snrmin
        pt_ra, pt_dec = st.metadata.radec
        src_ra, src_dec = source_location(pt_ra, pt_dec, l1, m1)
        logger.info('Peak (RA, Dec): %s, %s' % (src_ra, src_dec))

        # convert l1 and m1 from radians to arcminutes
        l1arcm = np.degrees(l1)*60
        m1arcm = np.degrees(m1)*60

        # build overall plot
        fig = plt.Figure(figsize=(12.75, 8))

        # add metadata in subfigure
        ax = fig.add_subplot(2, 3, 1, axisbg='white')

        # calculate the overall dispersion delay: dd
        f1 = st.metadata.freq_orig[0]
        f2 = st.metadata.freq_orig[-1]
        dd = 4.15*st.dmarr[dmind]*(f1**(-2)-f2**(-2))

        # add annotating info
        # set spacing and location of the annotating information
        start = 1.1
        space = 0.07
        left = 0.0
        ax.text(left, start, st.fileroot, fontname='sans-serif',
                transform=ax.transAxes, fontsize='small')
        ax.text(left, start-space, 'Peak (arcmin): ('
                + str(np.round(l1arcm, 3)) + ', '
                + str(np.round(m1arcm, 3)) + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        # split the RA and Dec and display in a nice format
        ra = src_ra.split()
        dec = src_dec.split()
        ax.text(left, start-2*space, 'Peak (RA, Dec): (' + ra[0] + ':' + ra[1]
                + ':' + ra[2][0:4] + ', ' + dec[0] + ':' + dec[1] + ':'
                + dec[2][0:4] + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-3*space, 'Source: ' + str(st.metadata.source),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-4*space, 'scan: ' + str(scan),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-5*space, 'segment: ' + str(segment),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-6*space, 'integration: ' + str(candint),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-7*space, 'DM = ' + str(st.dmarr[dmind])
                + ' (index ' + str(dmind) + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-8*space, 'dt = '
                + str(np.round(st.inttime*st.dtarr[dtind], 3)*1e3)
                + ' ms' + ' (index ' + str(dtind) + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-9*space, 'disp delay = ' + str(np.round(dd, 1))
                + ' ms',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-10*space, 'SNR: ' + str(np.round(snrobs, 1)),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')

        # set the plot invisible so that it doesn't interfere with annotations
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')

        # plot full dynamic spectra
        left, width = 0.75, 0.2*2./3.
        bottom, height = 0.2, 0.7
        # three rectangles for each panel of the spectrum (RR, RR+LL, LL)
        rect_dynsp1 = [left, bottom, width/3., height]
        rect_dynsp2 = [left+width/3., bottom, width/3., height]
        rect_dynsp3 = [left+2.*width/3., bottom, width/3., height]
        rect_lc1 = [left, bottom-0.1, width/3., 0.1]
        rect_lc2 = [left+width/3., bottom-0.1, width/3., 0.1]
        rect_lc3 = [left+2.*width/3., bottom-0.1, width/3., 0.1]
        rect_sp = [left+width, bottom, 0.1*2./3., height]
        ax_dynsp1 = fig.add_axes(rect_dynsp1)
        # sharey so that axes line up
        ax_dynsp2 = fig.add_axes(rect_dynsp2, sharey=ax_dynsp1)
        ax_dynsp3 = fig.add_axes(rect_dynsp3, sharey=ax_dynsp1)
        # hide RR+LL and LL dynamic spectra y labels to avoid overlap
        [label.set_visible(False) for label in ax_dynsp2.get_yticklabels()]
        [label.set_visible(False) for label in ax_dynsp3.get_yticklabels()]
        ax_sp = fig.add_axes(rect_sp, sharey=ax_dynsp3)
        [label.set_visible(False) for label in ax_sp.get_yticklabels()]
        ax_lc1 = fig.add_axes(rect_lc1)
        ax_lc2 = fig.add_axes(rect_lc2, sharey=ax_lc1)
        ax_lc3 = fig.add_axes(rect_lc3, sharey=ax_lc1)
        [label.set_visible(False) for label in ax_lc2.get_yticklabels()]
        [label.set_visible(False) for label in ax_lc3.get_yticklabels()]

        # now actually plot the data
        spectra = np.swapaxes(data.real, 0, 1)
        dd1 = spectra[..., 0]
        dd2 = spectra[..., 0] + spectra[..., 1]
        dd3 = spectra[..., 1]
        colormap = 'viridis'
        logger.debug('{0}'.format(dd1.shape))
        logger.debug('{0}'.format(dd2.shape))
        logger.debug('{0}'.format(dd3.shape))
        _ = ax_dynsp1.imshow(dd1, origin='lower', interpolation='nearest',
                             aspect='auto', cmap=plt.get_cmap(colormap))
        _ = ax_dynsp2.imshow(dd2, origin='lower', interpolation='nearest',
                             aspect='auto', cmap=plt.get_cmap(colormap))
        _ = ax_dynsp3.imshow(dd3, origin='lower', interpolation='nearest',
                             aspect='auto', cmap=plt.get_cmap(colormap))
        ax_dynsp1.set_yticks(range(0, len(st.freq), 30))
        ax_dynsp1.set_yticklabels(st.freq[::30])
        ax_dynsp1.set_ylabel('Freq (GHz)')
        ax_dynsp1.set_xlabel('RR')
        ax_dynsp1.xaxis.set_label_position('top')
        ax_dynsp2.set_xlabel('RR+LL')
        ax_dynsp2.xaxis.set_label_position('top')
        ax_dynsp3.set_xlabel('LL')
        ax_dynsp3.xaxis.set_label_position('top')
        # hide xlabels invisible so that they don't interefere with lc plots
        [label.set_visible(False) for label in ax_dynsp1.get_xticklabels()]
        # This one y label was getting in the way
        ax_dynsp1.get_yticklabels()[0].set_visible(False)
        # plot stokes I spectrum of the candidate pulse (assume middle bin)
        # select stokes I middle bin
        spectrum = spectra[:, len(spectra[0])//2].mean(axis=1)
        ax_sp.plot(spectrum, range(len(spectrum)), 'k.')
        # plot 0 Jy dotted line
        ax_sp.plot(np.zeros(len(spectrum)), range(len(spectrum)), 'r:')
        xmin, xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(np.linspace(xmin, xmax, 3).round(2))
        ax_sp.set_xlabel('Flux (Jy)')

        # plot mean flux values for each time bin
        lc1 = dd1.mean(axis=0)
        lc2 = dd2.mean(axis=0)
        lc3 = dd3.mean(axis=0)
        lenlc = len(data)
        ax_lc1.plot(range(0, lenlc), list(lc1)[:lenlc], 'k.')
        ax_lc2.plot(range(0, lenlc), list(lc2)[:lenlc], 'k.')
        ax_lc3.plot(range(0, lenlc), list(lc3)[:lenlc], 'k.')
        # plot 0 Jy dotted line for each plot
        ax_lc1.plot(range(0, lenlc), list(np.zeros(lenlc)), 'r:')
        ax_lc2.plot(range(0, lenlc), list(np.zeros(lenlc)), 'r:')
        ax_lc3.plot(range(0, lenlc), list(np.zeros(lenlc)), 'r:')
        ax_lc2.set_xlabel('Integration (rel)')
        ax_lc1.set_ylabel('Flux (Jy)')
        ax_lc1.set_xticks([0, 0.5*lenlc, lenlc])
        # only show the '0' label for one of the plots to avoid messy overlap
        ax_lc1.set_xticklabels(['0', str(lenlc//2), str(lenlc)])
        ax_lc2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_lc2.set_xticklabels(['', str(lenlc//2), str(lenlc)])
        ax_lc3.set_xticks([0, 0.5*lenlc, lenlc])
        ax_lc3.set_xticklabels(['', str(lenlc//2), str(lenlc)])
        ymin, ymax = ax_lc1.get_ylim()
        ax_lc1.set_yticks(np.linspace(ymin, ymax, 3).round(2))

        # adjust the x tick marks to line up with the lc plots
        ax_dynsp1.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp3.set_xticks([0, 0.5*lenlc, lenlc])

        # plot second set of dynamic spectra
        left, width = 0.45, 0.1333
        bottom, height = 0.1, 0.4
        rect_dynsp1 = [left, bottom, width/3., height]
        rect_dynsp2 = [left+width/3., bottom, width/3., height]
        rect_dynsp3 = [left+2.*width/3., bottom, width/3., height]
        rect_sp = [left+width, bottom, 0.1*2./3., height]
        ax_dynsp1 = fig.add_axes(rect_dynsp1)
        ax_dynsp2 = fig.add_axes(rect_dynsp2, sharey=ax_dynsp1)
        ax_dynsp3 = fig.add_axes(rect_dynsp3, sharey=ax_dynsp1)
        # hide RR+LL and LL dynamic spectra y labels
        [label.set_visible(False) for label in ax_dynsp2.get_yticklabels()]
        [label.set_visible(False) for label in ax_dynsp3.get_yticklabels()]
        ax_sp = fig.add_axes(rect_sp, sharey=ax_dynsp3)
        [label.set_visible(False) for label in ax_sp.get_yticklabels()]

        # calculate the channels to average together for SNR=2
        n = int((2.*(len(spectra))**0.5/snrobs)**2)
        if n == 0:  # if n==0 then don't average
            dd1avg = dd1
            dd3avg = dd3
        else:
            # otherwise, add zeros onto the data so that it's length is cleanly
            # divisible by n (makes it easier to average over)
            dd1zerotemp = np.concatenate((np.zeros((n-len(spectra) % n,
                                                    len(spectra[0])),
                                         dtype=dd1.dtype), dd1), axis=0)
            dd3zerotemp = np.concatenate((np.zeros((n-len(spectra) % n,
                                                    len(spectra[0])),
                                         dtype=dd3.dtype), dd3), axis=0)
            # make masked arrays so appended zeros do not affect average
            zeros = np.zeros((len(dd1), len(dd1[0])))
            ones = np.ones((n-len(spectra) % n, len(dd1[0])))
            masktemp = np.concatenate((ones, zeros), axis=0)
            dd1zero = np.ma.masked_array(dd1zerotemp, mask=masktemp)
            dd3zero = np.ma.masked_array(dd3zerotemp, mask=masktemp)
            # average together the data
            dd1avg = np.array([], dtype=dd1.dtype)
            for i in range(len(spectra[0])):
                temp = dd1zero[:, i].reshape(-1, n)
                tempavg = np.reshape(np.mean(temp, axis=1), (len(temp), 1))
                # repeats the mean values to create more pixels
                # (easier to properly crop when it is finally displayed)
                temprep = np.repeat(tempavg, n, axis=0)
                if i == 0:
                    dd1avg = temprep
                else:
                    dd1avg = np.concatenate((dd1avg, temprep), axis=1)
            dd3avg = np.array([], dtype=dd3.dtype)
            for i in range(len(spectra[0])):
                temp = dd3zero[:, i].reshape(-1, n)
                tempavg = np.reshape(np.mean(temp, axis=1), (len(temp), 1))
                temprep = np.repeat(tempavg, n, axis=0)
                if i == 0:
                    dd3avg = temprep
                else:
                    dd3avg = np.concatenate((dd3avg, temprep), axis=1)
        dd2avg = dd1avg + dd3avg  # add together to get averaged RR+LL spectrum
        colormap = 'viridis'
        # if n==0 then don't crop the spectra because no zeroes were appended
        if n == 0:
            dd1avgcrop = dd1avg
            dd2avgcrop = dd2avg
            dd3avgcrop = dd3avg
        else:  # otherwise, crop off the appended zeroes
            dd1avgcrop = dd1avg[len(ones):len(dd1avg), :]
            dd2avgcrop = dd2avg[len(ones):len(dd2avg), :]
            dd3avgcrop = dd3avg[len(ones):len(dd3avg), :]
        logger.debug('{0}'.format(dd1avgcrop.shape))
        logger.debug('{0}'.format(dd2avgcrop.shape))
        logger.debug('{0}'.format(dd3avgcrop.shape))
        _ = ax_dynsp1.imshow(dd1avgcrop, origin='lower',
                             interpolation='nearest', aspect='auto',
                             cmap=plt.get_cmap(colormap))
        _ = ax_dynsp2.imshow(dd2avgcrop, origin='lower',
                             interpolation='nearest', aspect='auto',
                             cmap=plt.get_cmap(colormap))
        _ = ax_dynsp3.imshow(dd3avgcrop, origin='lower',
                             interpolation='nearest', aspect='auto',
                             cmap=plt.get_cmap(colormap))
        ax_dynsp1.set_yticks(range(0, len(st.freq), 30))
        ax_dynsp1.set_yticklabels(st.freq[::30])
        ax_dynsp1.set_ylabel('Freq (GHz)')
        ax_dynsp1.set_xlabel('RR')
        ax_dynsp1.xaxis.set_label_position('top')
        ax_dynsp2.set_xlabel('Integration (rel)')
        ax2 = ax_dynsp2.twiny()
        ax2.set_xlabel('RR+LL')
        [label.set_visible(False) for label in ax2.get_xticklabels()]
        ax_dynsp3.set_xlabel('LL')
        ax_dynsp3.xaxis.set_label_position('top')

        # plot stokes I spectrum of the candidate pulse from middle integration
        ax_sp.plot(dd2avgcrop[:, len(dd2avgcrop[0])//2]/2.,
                   range(len(dd2avgcrop)), 'k.')
        ax_sp.plot(np.zeros(len(dd2avgcrop)), range(len(dd2avgcrop)), 'r:')
        xmin, xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(np.linspace(xmin, xmax, 3).round(2))
        ax_sp.get_xticklabels()[0].set_visible(False)
        ax_sp.set_xlabel('Flux (Jy)')

        # readjust the x tick marks on the dynamic spectra
        ax_dynsp1.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp1.set_xticklabels(['0', str(lenlc//2), str(lenlc)])
        ax_dynsp2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp2.set_xticklabels(['', str(lenlc//2), str(lenlc)])
        ax_dynsp3.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp3.set_xticklabels(['', str(lenlc//2), str(lenlc)])

        # plot the image and zoomed cutout
        ax = fig.add_subplot(2, 3, 4)
        fov = np.degrees(1./st.uvres)*60.
        _ = ax.imshow(im.transpose(), aspect='equal', origin='upper',
                      interpolation='nearest',
                      extent=[fov/2, -fov/2, -fov/2, fov/2],
                      cmap=plt.get_cmap('viridis'), vmin=0,
                      vmax=0.5*im.max())
        ax.set_xlabel('RA Offset (arcmin)')
        ax.set_ylabel('Dec Offset (arcmin)')
        # to set scale when we plot the triangles that label the location
        ax.autoscale(False)
        # add markers on the axes at measured position of the candidate
        ax.scatter(x=[l1arcm], y=[-fov/2], c='#ffff00', s=60, marker='^',
                   clip_on=False)
        ax.scatter(x=[fov/2], y=[m1arcm], c='#ffff00', s=60, marker='>',
                   clip_on=False)
        # makes it so the axis does not intersect the location triangles
        ax.set_frame_on(False)

        # add a zoomed cutout image of the candidate (set width at 5*beam)
        sbeam = np.mean(st.beamsize_deg)*60
        # figure out the location to center the zoomed image on
        xratio = len(im[0])/fov  # pix/arcmin
        yratio = len(im)/fov  # pix/arcmin
        mult = 5  # sets how many times the synthesized beam the zoomed FOV is
        xmin = max(0, int(len(im[0])//2-(m1arcm+sbeam*mult)*xratio))
        xmax = int(len(im[0])//2-(m1arcm-sbeam*mult)*xratio)
        ymin = max(0, int(len(im)//2-(l1arcm+sbeam*mult)*yratio))
        ymax = int(len(im)//2-(l1arcm-sbeam*mult)*yratio)
        left, width = 0.231, 0.15
        bottom, height = 0.465, 0.15
        rect_imcrop = [left, bottom, width, height]
        ax_imcrop = fig.add_axes(rect_imcrop)
        logger.debug('{0}'.format(im.transpose()[xmin:xmax, ymin:ymax].shape))
        logger.debug('{0} {1} {2} {3}'.format(xmin, xmax, ymin, ymax))
        _ = ax_imcrop.imshow(im.transpose()[xmin:xmax,ymin:ymax], aspect=1,
                             origin='upper', interpolation='nearest',
                             extent=[-1, 1, -1, 1],
                             cmap=plt.get_cmap('viridis'), vmin=0,
                             vmax=0.5*im.max())
        # setup the axes
        ax_imcrop.set_ylabel('Dec (arcmin)')
        ax_imcrop.set_xlabel('RA (arcmin)')
        ax_imcrop.xaxis.set_label_position('top')
        ax_imcrop.xaxis.tick_top()
        xlabels = [str(np.round(l1arcm+sbeam*mult/2, 1)), '',
                   str(np.round(l1arcm, 1)), '',
                   str(np.round(l1arcm-sbeam*mult/2, 1))]
        ylabels = [str(np.round(m1arcm-sbeam*mult/2, 1)), '',
                   str(np.round(m1arcm, 1)), '',
                   str(np.round(m1arcm+sbeam*mult/2, 1))]
        ax_imcrop.set_xticklabels(xlabels)
        ax_imcrop.set_yticklabels(ylabels)
        # change axis label loc of inset to avoid the full picture
        ax_imcrop.get_yticklabels()[0].set_verticalalignment('bottom')

        # create SNR versus N histogram for the whole observation
        # (properties for each candidate in the observation given by prop)
        if len(snrs):
            left, width = 0.45, 0.2
            bottom, height = 0.6, 0.3
            rect_snr = [left, bottom, width, height]
            ax_snr = fig.add_axes(rect_snr)
            print(snrs, type(snrs))
            pos_snrs = snrs[snrs >= 0]
            neg_snrs = snrs[snrs < 0]
            if not len(neg_snrs):  # if working with subset and only pos snrs
                neg_snrs = pos_snrs
                nonegs = True
            else:
                nonegs = False
            minval = 5.5
            maxval = 8.0
            # determine the min and max values of the x axis
            if min(pos_snrs) < min(np.abs(neg_snrs)):
                minval = min(pos_snrs)
            else:
                minval = min(np.abs(neg_snrs))
            if max(pos_snrs) > max(np.abs(neg_snrs)):
                maxval = max(pos_snrs)
            else:
                maxval = max(np.abs(neg_snrs))

            # positive SNR bins are in blue
            # absolute values of negative SNR bins are taken and plotted as
            # red x's on top of positive blue bins for compactness
            n, b, patches = ax_snr.hist(pos_snrs, 50, (minval, maxval),
                                        facecolor='blue', zorder=1)
            vals, bin_edges = np.histogram(np.abs(neg_snrs), 50,
                                           (minval, maxval))
            bins = np.array([(bin_edges[i]+bin_edges[i+1])/2.
                             for i in range(len(vals))])
            vals = np.array(vals)
            if not nonegs:
                ax_snr.scatter(bins[vals > 0], vals[vals > 0], marker='x',
                               c='orangered', alpha=1.0, zorder=2)
            ax_snr.set_xlabel('SNR')
            ax_snr.set_xlim(left=minval-0.2)
            ax_snr.set_xlim(right=maxval+0.2)
            ax_snr.set_ylabel('N')
            ax_snr.set_yscale('log')
            # draw vertical line where the candidate SNR is
            ax_snr.axvline(x=np.abs(snrobs), linewidth=1, color='y', alpha=0.7)

        if not outname:
            outname = os.path.join(st.metadata.workdir,
                                   'cands_{}_sc{}-seg{}-i{}-dm{}-dt{}.png'
                                   .format(st.fileroot, scan, segment, candint,
                                           dmind, dtind))

        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(outname)
        except ValueError:
            logger.warn('Could not write figure to %s' % outname)


def source_location(pt_ra, pt_dec, l1, m1):
    """ Takes phase center and src l,m in radians to get ra,dec of source.
    Returns string ('hh mm ss', 'dd mm ss')
    """
    import math

    srcra = np.degrees(pt_ra + l1/math.cos(pt_dec))
    srcdec = np.degrees(pt_dec + m1)

    return deg2HMS(srcra, srcdec)


def deg2HMS(ra='', dec='', round=False):
    """ quick and dirty coord conversion. googled to find bdnyc.org.
    """
    RA, DEC, rs, ds = '', '', '', ''
    if dec:
        if str(dec)[0] == '-':
            ds, dec = '-', abs(dec)
        deg = int(dec)
        decM = abs(int((dec-deg)*60))
        if round:
            decS = int((abs((dec-deg)*60)-decM)*60)
        else:
            decS = (abs((dec-deg)*60)-decM)*60
        DEC = '{0}{1} {2} {3}'.format(ds, deg, decM, decS)
  
    if ra:
        if str(ra)[0] == '-':
            rs, ra = '-', abs(ra)
        raH = int(ra/15)
        raM = int(((ra/15)-raH)*60)
        if round:
            raS = int(((((ra/15)-raH)*60)-raM)*60)
        else:
            raS = ((((ra/15)-raH)*60)-raM)*60
        RA = '{0}{1} {2} {3}'.format(rs, raH, raM, raS)
  
    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC



class CandidateDF(object):
    """ Wrap pandas DataFrame that allows candidate metadata and prefs to be attached and pickled.
    """

    def __init__(self, df, prefs=None, metadata=None):
        self.df = df
        self.prefs = prefs
        self.metadata = metadata
        self.rfpipe_version = version.__version__

