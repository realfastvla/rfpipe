from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str (numba signature bug)
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
from numba import jit, guvectorize, int64
import pyfftw
from kalman_detector import kalman_prepare_coeffs, kalman_significance
from concurrent import futures
from itertools import cycle
from threading import Lock

import logging
logger = logging.getLogger(__name__)

try:
    import rfgpu
except ImportError:
    pass


###
# packaged searching functions
###

def dedisperse_search_cuda(st, segment, data, devicenum=None):
    """ Run dedispersion, resample for all dm and dt.
    Returns candcollection with optional clustering.
    Grid and image on GPU (uses rfgpu from separate repo).
    Uses state to define integrations to image based on segment, dm, and dt.
    devicenum is int or tuple of ints that set gpu(s) to use.
    If not set, then it can be inferred with distributed.
    """

    from rfpipe import candidates, util

    assert st.dtarr[0] == 1, "st.dtarr[0] assumed to be 1"
    assert all([st.dtarr[dtind]*2 == st.dtarr[dtind+1]
                for dtind in range(len(st.dtarr)-1)]), ("dtarr must increase "
                                                        "by factors of 2")
    anydata = np.any(data)
    if not anydata or st.prefs.searchtype is None:
        if not anydata:
            logger.info("Data is all zeros. Skipping search.")
        return candidates.CandCollection(prefs=st.prefs,
                                         metadata=st.metadata)

    if isinstance(devicenum, int):
        devicenums = (devicenum,)
    elif isinstance(devicenum, str):
        devicenums = (int(devicenum),)
    elif isinstance(devicenum, tuple):
        assert isinstance(devicenum[0], int)
        devicenums = devicenum
    elif devicenum is None:
        # assume first gpu, but try to infer from worker name
        devicenum = 0
        try:
            from distributed import get_worker
            name = get_worker().name
            devicenum = int(name.split('g')[1])
            devicenums = (devicenum, devicenum+1)  # TODO: smarter multi-GPU
            logger.debug("Using name {0} to set GPU devicenum to {1}"
                         .format(name, devicenum))
        except IndexError:
            devicenums = (devicenum,)
            logger.warning("Could not parse worker name {0}. Using default GPU devicenum {1}"
                           .format(name, devicenum))
        except ValueError:
            devicenums = (devicenum,)
            logger.warning("No worker found. Using default GPU devicenum {0}"
                           .format(devicenum))
        except ImportError:
            devicenums = (devicenum,)
            logger.warning("distributed not available. Using default GPU devicenum {0}"
                           .format(devicenum))

    assert isinstance(devicenums, tuple)
    logger.info("Using gpu devicenum(s): {0}".format(devicenums))

    uvw = util.get_uvw_segment(st, segment)

    upix = st.npixx
    vpix = st.npixy//2 + 1

    grids = [rfgpu.Grid(st.nbl, st.nchan, st.readints, upix, vpix, dn) for dn in devicenums]
    images = [rfgpu.Image(st.npixx, st.npixy, dn) for dn in devicenums]
    for image in images:
        image.add_stat('rms')
        image.add_stat('pix')

    # Data buffers on GPU
    # Vis buffers identical on all GPUs. image buffer unique.
    vis_raw = rfgpu.GPUArrayComplex((st.nbl, st.nchan, st.readints),
                                    devicenums)
    vis_grids = [rfgpu.GPUArrayComplex((upix, vpix), (dn,)) for dn in devicenums]
    img_grids = [rfgpu.GPUArrayReal((st.npixx, st.npixy), (dn,)) for dn in devicenums]
#    locks = [Lock() for dn in devicenums]

    # Convert uv from lambda to us
    u, v, w = uvw
    u_us = 1e6*u[:, 0]/(1e9*st.freq[0])
    v_us = 1e6*v[:, 0]/(1e9*st.freq[0])

    # move Stokes I data in (assumes dual pol data)
    vis_raw.data[:] = np.rollaxis(data.mean(axis=3), 0, 3)

    # uv filtering
    if st.prefs.uvmin is not None:
        uvd = np.sqrt(u[:,0]**2 + v[:,0]**2)
        short = uvd < st.prefs.uvmin
        vis_raw.data[short] = 0j
    
    vis_raw.h2d()  # Send it to GPU memory of all

    for grid in grids:
        grid.set_uv(u_us, v_us)  # u, v in us
        grid.set_freq(st.freq*1e3)  # freq in MHz
        grid.set_cell(st.uvres)  # uv cell size in wavelengths (== 1/FoV(radians))
        grid.compute()
        grid.conjugate(vis_raw)

    # calc fraction of data gridded (any grid will do)
    gridfrac = grid.get_nnz()/(st.nbl*st.nchan)
    logger.info("Gridded {0}% of all baselines and channels".format(100*gridfrac))

    # some prep if kalman significance is needed
    if st.prefs.searchtype in ['imagek', 'armkimage', 'armk']:
        # TODO: check that this is ok if pointing at bright source
        spec_std, sig_ts, kalman_coeffs = util.kalman_prep(data)

        if not np.all(sig_ts):
            logger.info("sig_ts all zeros. Skipping search.")
            return candidates.CandCollection(prefs=st.prefs,
                                             metadata=st.metadata)
    else:
        spec_std, sig_ts, kalman_coeffs = None, [], []

    # place to hold intermediate result lists
    canddict = {}
    canddict['candloc'] = []
    canddict['l1'] = []
    canddict['m1'] = []
    canddict['snr1'] = []
    canddict['immax1'] = []
    for feat in st.searchfeatures:
        canddict[feat] = []

    for dtind in range(len(st.dtarr)):
        if dtind > 0:
            for grid in grids:
                logger.info("Downsampling for dn {0}"
                            .format(devicenums[grids.index(grid)]))
                grid.downsample(vis_raw)

#        cy = cycle(devicenums)
        threads = []
        with futures.ThreadPoolExecutor(max_workers=2*len(devicenums)) as ex:
#            for dmind, i_dn in list(zip(range(len(st.dmarr)), cy)):
#                threads.append(ex.submit(rfgpu_gridimage, st, segment,
#                                         grids[i_dn], images[i_dn], vis_raw,
#                                         vis_grids[i_dn], img_grids[i_dn],
#                                         dmind, dtind, devicenums[i_dn],
#                                         locks[i_dn]))
            ndm = len(st.dmarr)
            ndn = len(devicenums)
            for i_dn in range(ndn):
#                dminds = list(range(ndm)[i_dn*ndm//ndn:(i_dn+1)*ndm//ndn]
                dminds = [list(range(0+i, ndm, ndn)) for i in range(ndn)]
                threads.append(ex.submit(rfgpu_gridimage, st, segment,
                                         grids[i_dn], images[i_dn], vis_raw,
                                         vis_grids[i_dn], img_grids[i_dn],
                                         dminds[i_dn], dtind, devicenums[i_dn],
                                         data=data, uvw=uvw, spec_std=spec_std,
                                         sig_ts=sig_ts, kalman_coeffs=kalman_coeffs))

            for thread in futures.as_completed(threads):
                candlocs, l1s, m1s, snr1s, immax1s, snrks = thread.result()
                canddict['candloc'] += candlocs
                canddict['l1'] += l1s
                canddict['m1'] += m1s
                canddict['snr1'] += snr1s
                canddict['immax1'] += immax1s

    cc = candidates.make_candcollection(st, **canddict)
    logger.info("First pass found {0} candidates in seg {1}."
                .format(len(cc), segment))

    # check whether too many candidates
    if st.prefs.max_candfrac:
        total_integrations = 0
        for dtind in range(len(st.dtarr)):
            for dmind in range(len(st.dmarr)):
                total_integrations += len(st.get_search_ints(segment, dmind,
                                                             dtind))
        if len(cc)/total_integrations > st.prefs.max_candfrac:
            logger.warning("Too many candidates ({0} in {1} images). Flagging."
                           .format(len(cc), total_integrations))
            cc = candidates.make_candcollection(st,
                                                candloc=[(0, -1, 0, 0, 0)],
                                                ncands=[len(cc)])

    # add cluster labels to candidates
    if st.prefs.clustercands:
        cc = candidates.cluster_candidates(cc)

    return cc


def rfgpu_gridimage(st, segment, grid, image, vis_raw, vis_grid, img_grid,
                    dminds, dtind, devicenum, data=None, uvw=None, spec_std=None,
                    sig_ts=[], kalman_coeffs=[]):
    """ Dedisperse, grid, image, threshold with rfgpu
    """

    from rfpipe import util

    beamnum = 0
    candlocs, l1s, m1s, snr1s, immax1s, snrks = [], [], [], [], [], []
    for dmind in dminds:
        delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind],
                                st.inttime)
        integrations = st.get_search_ints(segment, dmind, dtind)

        if len(integrations) != 0:
            minint = min(integrations)
            maxint = max(integrations)
            logger.info('Imaging {0} ints ({1}-{2}) in seg {3} at DM/dt {4:.1f}/{5}'
                        ' with image {6}x{7} (uvres {8}) on GPU {9}'
                        .format(len(integrations), minint, maxint, segment,
                                st.dmarr[dmind], st.dtarr[dtind], st.npixx,
                                st.npixy, st.uvres, devicenum))

            grid.set_shift(delay >> dtind)  # dispersion shift per chan in samples

            zeros = []
            for i in integrations:
                grid.operate(vis_raw, vis_grid, i)
                image.operate(vis_grid, img_grid)

                # calc snr
                stats = image.stats(img_grid)
                if stats['rms'] != 0.:
                    snr1 = stats['max']/stats['rms']
                else:
                    snr1 = 0.
                    zeros.append(i)

                # threshold image
                if snr1 > st.prefs.sigma_image1:
                    candloc = (segment, i, dmind, dtind, beamnum)

                    xpeak = stats['xpeak']
                    ypeak = stats['ypeak']
                    l1, m1 = st.pixtolm((xpeak+st.npixx//2, ypeak+st.npixy//2))

                    if st.prefs.searchtype == 'image':
                        logger.info("Got one! SNR1 {0:.1f} candidate at {1} and (l, m) = ({2:.5f}, {3:.5f})"
                                    .format(snr1, candloc, l1, m1))
                        candlocs.append(candloc)
                        l1s.append(l1)
                        m1s.append(m1)
                        snr1s.append(snr1)
                        immax1s.append(stats['max'])

                    elif st.prefs.searchtype == 'imagek':
                        # TODO: implement phasing on GPU
                        data_corr = dedisperseresample(data, delay,
                                                       st.dtarr[dtind],
                                                       parallel=st.prefs.nthread > 1,
                                                       resamplefirst=True)
                        spec = data_corr.take([i], axis=0)
                        util.phase_shift(spec, uvw, l1, m1)
                        spec = spec[0].real.mean(axis=2).mean(axis=0)

                        # TODO: this significance can be biased low if averaging in long baselines that are not phased well
                        # TODO: spec should be calculated from baselines used to measure l,m?
                        significance_kalman = -kalman_significance(spec,
                                                                   spec_std,
                                                                   sig_ts=sig_ts,
                                                                   coeffs=kalman_coeffs)
                        snrk = (2*significance_kalman)**0.5
                        snrtot = (snrk**2 + snr1**2)**0.5
                        if snrtot > (st.prefs.sigma_kalman**2 + st.prefs.sigma_image1**2)**0.5:
                            logger.info("Got one! SNR1 {0:.1f} and SNRk {1:.1f} candidate at {2} and (l,m) = ({3:.5f}, {4:.5f})"
                                        .format(snr1, snrk, candloc, l1, m1))
                            candlocs.append(candloc)
                            l1s.append(l1)
                            m1s.append(m1)
                            snr1s.append(snr1)
                            immax1s.append(stats['max'])
                            snrks.append(snrk)
                    elif st.prefs.searchtype == 'armkimage':
                        raise NotImplementedError
                    elif st.prefs.searchtype == 'armk':
                        raise NotImplementedError
                    elif st.prefs.searchtype is not None:
                        logger.warning("searchtype {0} not recognized"
                                       .format(st.prefs.searchtype))
            if zeros:
                logger.warning("rfgpu rms is 0 in ints {0}."
                               .format(zeros))

    return candlocs, l1s, m1s, snr1s, immax1s, snrks


def dedisperse_search_fftw(st, segment, data, wisdom=None):
    """     Run dedispersion, resample for all dm and dt.
    Returns candcollection with optional clustering.
    Integrations can define subset of all available in data to search.
    Default will take integrations not searched in neighboring segments.
    ** only supports threshold > image max (no min)
    ** dmind, dtind, beamnum assumed to represent current state of data
    """

    from rfpipe import candidates, util

    anydata = np.any(data)
    if not anydata or st.prefs.searchtype is None:
        if not anydata:
            logger.info("Data is all zeros. Skipping search.")
        return candidates.CandCollection(prefs=st.prefs,
                                         metadata=st.metadata)

    # some prep if kalman significance is needed
    if st.prefs.searchtype in ['imagek', 'armkimage', 'armk']:
        # TODO: check that this is ok if pointing at bright source
        spec_std, sig_ts, kalman_coeffs = util.kalman_prep(data)

        if not np.all(sig_ts):
            logger.info("sig_ts all zeros. Skipping search.")
            return candidates.CandCollection(prefs=st.prefs,
                                             metadata=st.metadata)
    else:
        spec_std, sig_ts, kalman_coeffs = None, [], []

    beamnum = 0
    uvw = util.get_uvw_segment(st, segment)

    # place to hold intermediate result lists
    canddict = {}
    canddict['candloc'] = []
    for feat in st.searchfeatures:
        canddict[feat] = []

    for dtind in range(len(st.dtarr)):
        for dmind in range(len(st.dmarr)):
            # set search integrations
            integrations = st.get_search_ints(segment, dmind, dtind)
            if len(integrations) == 0:
                continue
            minint = min(integrations)
            maxint = max(integrations)

            logger.info('{0} search of {1} ints ({2}-{3}) in seg {4} at DM/dt '
                        '{5:.1f}/{6} with image {7}x{8} (uvres {9}) with fftw'
                        .format(st.prefs.searchtype, len(integrations), minint,
                                maxint, segment, st.dmarr[dmind],
                                st.dtarr[dtind], st.npixx,
                                st.npixy, st.uvres))

            # correct data
            delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind],
                                    st.inttime)
            data_corr = dedisperseresample(data, delay, st.dtarr[dtind],
                                           parallel=st.prefs.nthread > 1,
                                           resamplefirst=False)

            # run search
            if st.prefs.searchtype in ['image', 'imagek']:
                images = grid_image(data_corr, uvw, st.npixx, st.npixy, st.uvres,
                                    'fftw', st.prefs.nthread, wisdom=wisdom,
                                    integrations=integrations)

                for i, image in enumerate(images):
                    immax1 = image.max()
                    snr1 = immax1/image.std()
                    if snr1 > st.prefs.sigma_image1:
                        candloc = (segment, integrations[i], dmind, dtind, beamnum)
                        l1, m1 = st.pixtolm(np.where(image == immax1))

                        # if set, use sigma_kalman as second stage filter
                        if st.prefs.searchtype == 'imagek':
                            spec = data_corr.take([integrations[i]], axis=0)
                            util.phase_shift(spec, uvw, l1, m1)
                            spec = spec[0].real.mean(axis=2).mean(axis=0)
                            # TODO: this significance can be biased low if averaging in long baselines that are not phased well
                            # TODO: spec should be calculated from baselines used to measure l,m?
                            significance_kalman = -kalman_significance(spec,
                                                                       spec_std,
                                                                       sig_ts=sig_ts,
                                                                       coeffs=kalman_coeffs)
                            snrk = (2*significance_kalman)**0.5
                            snrtot = (snrk**2 + snr1**2)**0.5
                            if snrtot > (st.prefs.sigma_kalman**2 + st.prefs.sigma_image1**2)**0.5:
                                logger.info("Got one! SNR1 {0:.1f} and SNRk {1:.1f} candidate at {2} and (l,m) = ({3:.5f}, {4:.5f})"
                                            .format(snr1, snrk, candloc, l1, m1))
                                canddict['candloc'].append(candloc)
                                canddict['l1'].append(l1)
                                canddict['m1'].append(m1)
                                canddict['snr1'].append(snr1)
                                canddict['immax1'].append(immax1)
                                canddict['snrk'].append(snrk)
                        elif st.prefs.searchtype == 'image':
                            logger.info("Got one! SNR1 {0:.1f} candidate at {1} and (l, m) = ({2:.5f}, {3:.5f})"
                                        .format(snr1, candloc, l1, m1))
                            canddict['candloc'].append(candloc)
                            canddict['l1'].append(l1)
                            canddict['m1'].append(m1)
                            canddict['snr1'].append(snr1)
                            canddict['immax1'].append(immax1)

            elif st.prefs.searchtype in ['armkimage', 'armk']:
                armk_candidates = search_thresh_armk(st, data_corr, uvw,
                                                     integrations=integrations,
                                                     spec_std=spec_std,
                                                     sig_ts=sig_ts,
                                                     coeffs=kalman_coeffs)

                for candind, snrarms, snrk, armloc, peakxy, lm in armk_candidates:
                    candloc = (segment, candind, dmind, dtind, beamnum)

                    # if set, use sigma_kalman as second stage filter
                    if st.prefs.searchtype == 'armkimage':
                        image = grid_image(data_corr, uvw, st.npixx_full,
                                           st.npixy_full, st.uvres, 'fftw',
                                           st.prefs.nthread,
                                           wisdom=wisdom, integrations=candind)
                        peakx, peaky = np.where(image[0] == image[0].max())
                        l1, m1 = st.calclm(st.npixx_full, st.npixy_full,
                                           st.uvres, peakx[0], peaky[0])
                        immax1 = image.max()
                        snr1 = immax1/image.std()
                        if snr1 > st.prefs.sigma_image1:
                            logger.info("Got one! SNRarms {0:.1f} and SNRk "
                                        "{1:.1f} and SNR1 {2:.1f} candidate at"
                                        " {3} and (l,m) = ({4:.5f}, {5:.5f})"
                                        .format(snrarms, snrk, snr1,
                                                candloc, l1, m1))
                            canddict['candloc'].append(candloc)
                            canddict['l1'].append(l1)
                            canddict['m1'].append(m1)
                            canddict['snrarms'].append(snrarms)
                            canddict['snrk'].append(snrk)
                            canddict['snr1'].append(snr1)
                            canddict['immax1'].append(immax1)

                    elif st.prefs.searchtype == 'armk':
                        l1, m1 = lm
                        logger.info("Got one! SNRarms {0:.1f} and SNRk {1:.1f} "
                                    "candidate at {2} and (l,m) = ({3:.5f}, {4:.5f})"
                                    .format(snrarms, snrk, candloc, l1, m1))
                        canddict['candloc'].append(candloc)
                        canddict['l1'].append(l1)
                        canddict['m1'].append(m1)
                        canddict['snrarms'].append(snrarms)
                        canddict['snrk'].append(snrk)
            elif st.prefs.searchtype is not None:
                raise NotImplemented("only searchtype=image, imagek, armk, armkimage implemented")

    # save search results and its features
    cc = candidates.make_candcollection(st, **canddict)
    logger.info("First pass found {0} candidates in seg {1}."
                .format(len(cc), segment))

    # check whether too many candidates
    if st.prefs.max_candfrac:
        total_integrations = 0
        for dtind in range(len(st.dtarr)):
            for dmind in range(len(st.dmarr)):
                total_integrations += len(st.get_search_ints(segment, dmind,
                                                             dtind))
        if len(cc)/total_integrations > st.prefs.max_candfrac:
            logger.warning("Too many candidates ({0} in {1} images). Flagging."
                           .format(len(cc), total_integrations))
            cc = candidates.make_candcollection(st,
                                                candloc=[(0, -1, 0, 0, 0)],
                                                ncands=[len(cc)])

    # add cluster labels to candidates
    if st.prefs.clustercands:
        cc = candidates.cluster_candidates(cc)

        # TODO: find a way to return values as systematic data quality test

    return cc


def grid_image(data, uvw, npixx, npixy, uvres, fftmode, nthread, wisdom=None,
               integrations=None):
    """ Grid and image data.
    Optionally image integrations in list i.
    fftmode can be fftw or cuda.
    nthread is number of threads to use
    """

    if integrations is None:
        integrations = list(range(len(data)))
    elif not isinstance(integrations, list):
        integrations = [integrations]

    if fftmode == 'fftw':
        logger.debug("Imaging with fftw on {0} threads".format(nthread))
        grids = grid_visibilities(data.take(integrations, axis=0), uvw, npixx,
                                  npixy, uvres, parallel=nthread > 1)
        images = image_fftw(grids, nthread=nthread, wisdom=wisdom)
    elif fftmode == 'cuda':
        logger.warning("Imaging with cuda not yet supported.")
        images = image_cuda()
    else:
        logger.warning("Imaging fftmode {0} not supported.".format(fftmode))

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


@jit(nogil=True, nopython=True, cache=True)
def _grid_visibilities_jit(data, u, v, w, npixx, npixy, uvres, grids):
    b""" Grid visibilities into rounded uv coordinates using jit on single core.
    Rounding not working here, so minor differences with original and
    guvectorized versions.
    """

    nint, nbl, nchan, npol = data.shape

    for j in range(nbl):
        for k in range(nchan):
            ubl = int64(np.round(u[j, k]/uvres, 0))
            vbl = int64(np.round(v[j, k]/uvres, 0))
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
# dedispersion and resampling
###

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

    assert nchan == len(delay), "Number of channels in delay must be same as in data"
    if parallel:
        data = data.copy()
        _ = _dedisperse_gu(np.swapaxes(data, 0, 1), delay)
        return data[0:len(data)-delay.max()]
    else:
        result = np.zeros(shape=newsh, dtype=data.dtype)
        _dedisperse_jit(np.require(data, requirements='W'), delay, result)
        return result


@jit(nogil=True, nopython=True, cache=True)
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


def dedisperse_roll(data, delay):
    """ Using numpy roll to dedisperse.
    This avoids trimming data to area with valid delays,
    which is appropriate for dm-time data generation.
    TODO: check that -delay is correct way
    """

    nf, nt = data.shape
    assert nf == len(delay), "Delay must be same length as data freq axis"

    dataout = np.vstack([np.roll(arr, -delay[i]) for i, arr in enumerate(data)])
    return dataout


def make_dmt(data, dmi, dmf, dmsteps, freqs, inttime, mode='GPU', devicenum=0):
    """ Disperse data to a range of dms.
    Good transients have characteristic shape in dm-time space.
    """

    if mode == 'GPU':
        dmt = gpu_dmtime(data, dmi, dmf, dmsteps, freqs, inttime,
                         devicenum=devicenum)
    else:
        dmt = cpu_dmtime(data, dmi, dmf, dmsteps, freqs, inttime)

    return dmt


def cpu_dmtime(data, dmi, dmf, dmsteps, freqs, inttime):
    """ Make dm-time plot. Called by make_dmt
    """

    from rfpipe import util
    dmt = np.zeros((dmsteps, data.shape[1]), dtype=np.float32)
    dm_list = np.linspace(dmi, dmf, dmsteps)
    for ii, dm in enumerate(dm_list):
        delay = util.calc_delay(freqs, freqs.max(), dm, inttime)
        dmt[ii, :] = dedisperse_roll(data, delay).sum(axis=0)
    return dmt


def gpu_dmtime(ft, dm_i, dm_f, dmsteps, freqs, inttime, devicenum=0):
    """ Make dm-time plot. Called by make_dmt
    """

    from numba import cuda
    import math
    from rfpipe import util
    import os

    os.environ['NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT'] = '1'
    dm_list = np.linspace(dm_i, dm_f, dmsteps)
    delays = np.zeros((dmsteps, ft.shape[0]), dtype=np.int32)
    for ii, dm in enumerate(dm_list):
        delays[ii,:] = util.calc_delay(freqs, freqs.max(), dm, inttime).astype('int32')
      
    cuda.select_device(devicenum)
    stream = cuda.stream()

    dm_time = np.zeros((delays.shape[0], int(ft.shape[1])), dtype=np.float32)

    @cuda.jit(fastmath=True)
    def gpu_dmt(cand_data_in, all_delays, cand_data_out):
        ii, jj, kk = cuda.grid(3)
        if ii < cand_data_in.shape[0] and jj < cand_data_out.shape[1] and kk < all_delays.shape[1]:
            cuda.atomic.add(cand_data_out, (kk, jj), cand_data_in[ii,
                                                                  (jj + all_delays[ii,kk])%cand_data_in.shape[1]])

    with cuda.defer_cleanup():
        all_delays = cuda.to_device(delays.T, stream=stream)
        dmt_return = cuda.device_array(dm_time.shape, dtype=np.float32, stream=stream)
        cand_data_in = cuda.to_device(np.array(ft, dtype=np.float32), stream=stream)

        threadsperblock = (16, 4, 16)
        blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(dm_time.shape[0] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        gpu_dmt[blockspergrid, threadsperblock, stream](cand_data_in, all_delays,  dmt_return)
        dm_time = dmt_return.copy_to_host(stream=stream)
    # cuda.close()

    return dm_time


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


@jit(nogil=True, nopython=True, cache=True)
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


def dedisperseresample(data, delay, dt, parallel=False, resamplefirst=True):
    """ Dedisperse and resample in single function.
    parallel controls use of multicore versions of algorithms.
    resamplefirst is parameter that reproduces rfgpu order.
    """

    if not np.any(data):
        return np.array([])

    logger.info('Correcting by delay/resampling {0}/{1} ints in {2} mode'
                .format(delay.max(), dt, ['single', 'parallel'][parallel]))

    nint, nbl, nchan, npol = data.shape
    newsh = (int64(nint-delay.max())//dt, nbl, nchan, npol)

    if resamplefirst:
        result = resample(data, dt, parallel=parallel)
        result = dedisperse(result, delay//dt, parallel=parallel)
        return result
    else:
        if parallel:
            data = data.copy()
            _ = _dedisperseresample_gu(np.swapaxes(data, 0, 1),
                                       delay, dt)
            return data[0:(len(data)-delay.max())//dt]
        else:
            result = np.zeros(shape=newsh, dtype=data.dtype)
            _dedisperseresample_jit(data, delay, dt, result)
            return result


@jit(nogil=True, nopython=True, cache=True)
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


###
# cascading 3arm imaging with kalman filter
###

def search_thresh_armk(st, data, uvw, integrations=None, spec_std=None,
                       sig_ts=[], coeffs=[]):
    """
    """

    from rfpipe import util

    if integrations is None:
        integrations = list(range(len(data)))
    elif isinstance(integrations, int):
        integrations = [integrations]

    if spec_std is None:
        if data.shape[0] > 1:
            spec_std = data.real.mean(axis=3).mean(axis=1).std(axis=0)
        else:
            spec_std = data[0].real.mean(axis=2).std(axis=0)

    if not len(sig_ts):
        sig_ts = [x*np.median(spec_std) for x in [0.3, 0.1, 0.03, 0.01]]

    if not len(coeffs):
        if not np.any(spec_std):
            logger.warning("spectrum std all zeros. Not estimating coeffs.")
            kalman_coeffs = []
        else:
            sig_ts, kalman_coeffs = kalman_prepare_coeffs(spec_std)

        if not np.all(np.nan_to_num(sig_ts)):
            kalman_coeffs = []

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

    # TODO: This is not returning bright simulated transients. Why?
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
                significance_kalman = -kalman_significance(spec, spec_std,
                                                           sig_ts=sig_ts,
                                                           coeffs=coeffs)
                snrk = (2*significance_kalman)**0.5
                snrtot = (snrk**2 + snrarms[i, j]**2)**0.5
                if (snrtot > (st.prefs.sigma_kalman**2 + st.prefs.sigma_arms**2)**0.5) and (snrtot > snrlast):
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
    assert (np.linalg.norm(e0) > 0.99) and (np.linalg.norm(e0) < 1.01), "Problem with unit vector e0: {0}".format(e0)
    assert (np.linalg.norm(e1) > 0.99) and (np.linalg.norm(e1) < 1.01), "Problem with unit vector e1: {1}".format(e1)
    assert (np.linalg.norm(e2) > 0.99) and (np.linalg.norm(e2) < 1.01), "Problem with unit vector e2: {2}".format(e2)

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


@jit(nopython=True, cache=True)
def projectarms(dpix0, dpix1, T012, npix2):
    """ Take any two locations relative to center and project in a new direction.
    npix2 is size of direction2.
    """

    newpix = int(round(np.dot(np.array([float(dpix0), float(dpix1)]),
                       T012) + npix2//2))

    return newpix


@jit(nopython=True, cache=True)
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
                # check score if intersections are all on grid
                if ind2 < npix:
                    score = arm0[i, ind0] + arm1[i, ind1] + arm2[i, ind2]
                else:
                    score = 0.

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


@jit(nogil=True, nopython=True, cache=True)
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
