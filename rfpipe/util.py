from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
import math
import random
from time import sleep
from numba import cuda, guvectorize
from numba import jit, complex64, int64, float32
from scipy import constants, interpolate
import sdmpy
from rfpipe import calibration
from astropy import time, coordinates
import logging
logger = logging.getLogger(__name__)
logger.setLevel(20)

# Manage the astropy cached IERS data.  The rfnodes do not have direct
# network access so we disable the check for out-of-date IERS files.
# This means the data need to be downloaded and put in the right
# spot via some machine that can get on the network.
import astropy.utils.iers
astropy.utils.iers.conf.auto_max_age = None

def update_iers_cache():
    """Update cached astropy IERS files.  Must be run from a system
    with internet access."""

    # Force update of IERS A file:
    import astropy.utils.data
    astropy.utils.data.download_file(astropy.utils.iers.IERS_A_URL, cache='update')

    # Run a dummy UVW calc to make sure astropy gets all 
    # other files it needs for this.
    dummy_xyz = np.array([
        [-1604008.7431 , -5042135.8194 ,  3553403.7084 ],
        [-1601315.9011 , -5041985.30447,  3554808.3081 ]
        ])
    calc_uvw_astropy(time.Time.now(), (0.0,0.0), dummy_xyz) 

def getsdm(*args, **kwargs):
    """ Wrap sdmpy.SDM to get around schema change error """

    try:
        sdm = sdmpy.SDM(*args, **kwargs)
    except:
        kwargs['use_xsd'] = False
        sdm = sdmpy.SDM(*args, **kwargs)

    return sdm


def phase_shift(data, uvw=None, dl=None, dm=None, dw=None, ints=None):
    """ Applies a phase shift to data for a given (dl, dm).
    """

    data = np.require(data, requirements='W')
    if ints is None:
        ints = list(range(len(data)))

    if (uvw is not None) and (dl is not None) and (dm is not None):
        assert data.shape[1] == uvw[0].shape[0]
        assert data.shape[2] == uvw[0].shape[1]
        u, v, w = uvw
        _phaseshiftlm_jit(data, u, v, w, dl, dm, ints=ints)
    elif dw is not None:
        assert data.shape[1] == dw.shape[0]
        assert data.shape[2] == dw.shape[1]
        _phaseshiftdw_jit(data, dw, ints=ints)
    else:
        logger.warn("phase_shift requires either uvw/dl/dm or dw")


@jit(nogil=True, nopython=True, cache=True)
def _phaseshiftlm_jit(data, u, v, w, dl, dm, ints):

    sh = data.shape

    if (dl != 0.) or (dm != 0.):
        for j in range(sh[1]):
            for k in range(sh[2]):
                # + np.sqrt(1-dl**2-dm**2)*w[j, k]))
                frot = np.exp(-2j*np.pi*(dl*u[j, k] + dm*v[j, k]))
                for i in ints:
                    for l in range(sh[3]):    # iterate over pols
                        # phasor unwraps phase at (dl, dm) per (bl, chan)
                        data[i, j, k, l] = data[i, j, k, l] * frot


@jit(nogil=True, nopython=True, cache=True)
def _phaseshiftdw_jit(data, dw, ints):

    sh = data.shape

    for j in range(sh[1]):
        for k in range(sh[2]):
            frot = np.exp(-2j*np.pi*dw[j, k])  # Q: which sign out front? needs 2pi?
            for i in ints:
                for l in range(sh[3]):    # iterate over pols
                    # phasor unwraps phase at (dl, dm) per (bl, chan)
                    data[i, j, k, l] = data[i, j, k, l] * frot


def meantsub(data, mode='mean'):
    """ Subtract mean visibility in time.
    mode can be set in prefs.timesub as None, 'mean', or '2pt'
    """

    # TODO: make outlier resistant to avoid oversubtraction
    if mode == None:
        logger.info('No visibility subtraction done.')
    elif mode == 'mean':
        logger.info('Subtracting mean visibility in time.')
        _meantsub_jit(np.require(data, requirements='W'))
    elif mode == '2pt':
        logger.info('Subtracting 2pt time trend in visibility.')
        _2ptsub_jit(np.require(data, requirements='W'))
    elif mode == 'cs':
        logger.info("Subtracting cubic spline time trend in visibility")
        assert len(data) > 4, "Too few integrations for spline sub"
        nint, nbl, nchan, npol = data.shape
        piece = nint//4
        dataavg = np.empty((4, nbl, nchan, npol), dtype=np.complex64)
        _cssub0_jit(np.require(data, requirements='W'), dataavg)
        spline = interpolate.interp1d(np.array([piece*(i+0.5) for i in range(4)]),
                                      dataavg, axis=0, fill_value='extrapolate')
        dataavginterp = spline(range(len(data)))
        _cssub1_jit(data, dataavginterp)
    else:
        logger.warn("meantsub mode not recognized")

    return data


@jit(nogil=True, nopython=True, cache=True)
def _meantsub_jit(data):
    """ Calculate mean in time (ignoring zeros) and subtract in place

    Could ultimately parallelize by computing only on subset of data.
    """

    nint, nbl, nchan, npol = data.shape

    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                ss = complex64(0)
                weight = int64(0)
                for l in range(nint):
                    ss += data[l, i, j, k]
                    if data[l, i, j, k] != 0j:
                        weight += 1
                if weight > 0:
                    mean = ss/weight
                else:
                    mean = complex64(0)

                if mean:
                    for l in range(nint):
                        if data[l, i, j, k] != 0j:
                            data[l, i, j, k] -= mean


@jit(nogil=True, nopython=True, cache=True)
def _2ptsub_jit(data):
    """ Calculate 2-pt time trend and evaluate to subtract at each time.
    """

    nint, nbl, nchan, npol = data.shape

    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                # first half mean
                ss1 = complex64(0)
                weight1 = int64(0)
                for l in range(0, nint//2):
                    ss1 += data[l, i, j, k]
                    if data[l, i, j, k] != 0j:
                        weight1 += 1
                if weight1 > 0:
                    mean1 = ss1/weight1
                else:
                    mean1 = complex64(0)

                # second half mean
                ss2 = complex64(0)
                weight2 = int64(0)
                for l in range(nint//2, nint):
                    ss2 += data[l, i, j, k]
                    if data[l, i, j, k] != 0j:
                        weight2 += 1
                if weight2 > 0:
                    mean2 = ss2/weight2
                else:
                    mean2 = complex64(0)

                # calc mean per int
                if mean1 and mean2:
                    slope = (mean2-mean1)/(nint//2)
                    mean0 = (mean2+mean1)/2
                    for l in range(nint):
                        if data[l, i, j, k] != 0j:
                            data[l, i, j, k] -= slope*(l-nint//2) + mean0
                else:  # or just blank data
                    for l in range(nint):
                        data[l, i, j, k] = 0j


@jit(nogil=True, nopython=True, cache=True)
def _2ptinterp_jit(data):
    """ Calculate 2-pt time trend and evaluate to subtract at each time.
    """

    nint, nbl, nchan, npol = data.shape

    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                # first half mean
                ss1 = complex64(0)
                weight1 = int64(0)
                for l in range(0, nint//2):
                    ss1 += data[l, i, j, k]
                    if data[l, i, j, k] != 0j:
                        weight1 += 1
                if weight1 > 0:
                    mean1 = ss1/weight1
                else:
                    mean1 = complex64(0)

                # second half mean
                ss2 = complex64(0)
                weight2 = int64(0)
                for l in range(nint//2, nint):
                    ss2 += data[l, i, j, k]
                    if data[l, i, j, k] != 0j:
                        weight2 += 1
                if weight2 > 0:
                    mean2 = ss2/weight2
                else:
                    mean2 = complex64(0)

                ff = interpolate.interp1d([nint//4, 3*nint//4], [mean1, mean2],
                                          fill_value='extrapolate')
                for l in range(nint):
                    if data[l, i, j, k] != 0j:
                        data[l, i, j, k] -= slope*(l-nint//2) + mean0


def _cssub(data):
    """ Use scipy interpolate to subtract cubic spline
    Superseded by _cssub0_jit and _cssub1_jit with interpolation call.
    """

    # use 4 windows to make interp1d cubic spline function
    nint = len(data)//4

    dataavg = np.concatenate([data[nint*i:nint*(i+1)].mean(axis=0)[None,:,:,:]
                              for i in range(4)], axis=0)
    spline = interpolate.interp1d(np.array([nint*(i+0.5) for i in range(4)]),
                              dataavg, axis=0, fill_value='extrapolate')

    data -= spline(range(len(data)))


@jit(nogil=True, nopython=True, cache=True)
def _cssub0_jit(data, dataavg):
    """ Use scipy calculate 4-pt mean as input to spline estimate.
    zeroed data is treated as flagged
    """

    nint, nbl, nchan, npol = data.shape
    piece = nint//4

    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                # mean in each piece
                for pp in range(4):
                    ss = complex64(0)
                    weight = int64(0)
                    for l in range(pp*piece, (pp+1)*piece):
                        ss += data[l, i, j, k]
                        if data[l, i, j, k] != 0j:
                            weight += 1
                    if weight > 0:
                        dataavg[pp, i, j, k] = ss/weight
                    else:
                        dataavg[pp, i, j, k] = complex64(0)  # TODO: instead use nearest?


@jit(nogil=True, nopython=True, cache=True)
def _cssub1_jit(data, dataavginterp):
    """ Use interpolated data to subtract while ignoring zeros
    """

    nint, nbl, nchan, npol = data.shape

    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                for l in range(nint):
                    if data[l, i, j, k] == 0j:
                        dataavginterp[l, i, j, k] = complex64(0)
                    data[l, i, j, k] -= dataavginterp[l, i, j, k]


@jit(nogil=True, nopython=True, cache=True)
def blstd(data, mask):
    """ Calculate std over baselines (ignoring zeros).
    Expects masked array as input
    """

    nint, nbl, nchan, npol = data.shape
    # getting "data type not understood" if this typed as float32
    blstd = np.zeros((nint, nchan, npol), dtype=complex64)

    for i in range(nint):
        for j in range(nchan):
            for k in range(npol):
                ss = complex64(0)
                weight = int64(0)
                for l in range(nbl):
                    ss += data[i, l, j, k]
                    if mask[i, l, j, k] is False:
                        weight += 1
                if weight > 0:
                    mean = ss/weight
                    ss = complex64(0)
                    for l in range(nbl):
                        if mask[i, l, j, k] is False:
                            ss += np.abs((data[i, l, j, k]-mean)**2)
                    blstd[i, j, k] = np.sqrt(ss/weight)
                else:
                    blstd[i, j, k] = complex64(0)

    return blstd.real


def calc_delay(freq, freqref, dm, inttime, scale=None):
    """ Calculates the delay in integration time bins due to dispersion delay.
    freq is array of frequencies. delay is relative to freqref.
    default scale is 4.1488e-3 as linear prefactor (reproducing for rtpipe<=1.54 requires 4.2e-3).
    Casts to int, so it uses a floor operation, not round.
    """

    # TODO: implement shri's latest on scale a: 4.1513913646
    scale = 4.1488e-3 if not scale else scale
    delay = np.zeros(len(freq), dtype=np.int32)

    for i in range(len(freq)):
        delay[i] = int(scale * dm * (1./freq[i]**2 - 1./freqref**2)/inttime)

    return delay


def calc_delay2(freq, freqref, dm, scale=None):
    """ Calculates the delay in seconds due to dispersion delay.
    freq is array of frequencies. delay is relative to freqref.
    default scale is 4.1488e-3 as linear prefactor (reproducing for rtpipe<=1.54 requires 4.2e-3).
    """

    scale = 4.1488e-3 if not scale else scale
    return scale*dm*(1./freq**2 - 1./freqref**2)


def calc_dmarr(state):
    """ Function to calculate the DM values for a given maximum sensitivity loss.
    dm_maxloss is sensitivity loss tolerated by dm bin width. dm_pulsewidth is
    assumed pulse width in microsec.
    """

    dm_maxloss = state.prefs.dm_maxloss
    dm_pulsewidth = state.prefs.dm_pulsewidth
    mindm = state.prefs.mindm
    maxdm = state.prefs.maxdm

    # parameters
    tsamp = state.inttime*1e6  # in microsec
    k = 8.3
    freq = state.freq.mean()  # central (mean) frequency in GHz
    bw = 1e3*(state.freq.max() - state.freq.min())  # in MHz
    ch = 1e-6*state.metadata.spw_chansize[0]  # in MHz ** first spw only

#    print(dm_maxloss, dm_pulsewidth, tsamp, freq, bw, ch)

    # width functions and loss factor
#    dt0 = lambda dm: np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2)
#    dt1 = lambda dm, ddm: np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2 + ((k*ddm*bw)/(freq**3.))**2)
#    loss = lambda dm, ddm: 1 - np.sqrt(dt0(dm)/dt1(dm, ddm))
#    loss_cordes = lambda ddm, dfreq, dm_pulsewidth, freq: 1 - (np.sqrt(np.pi) / (2 * 6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))) * erf(6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))  # not quite right for underresolved pulses

    if maxdm == 0:
        return [0]
    else:
        # iterate over dmgrid to find optimal dm values. go higher than maxdm to be sure final list includes full range.
        dmgrid = np.arange(mindm, maxdm, 0.05)
        dmgrid_final = [dmgrid[0]]
        for i in range(len(dmgrid)):
            ddm = (dmgrid[i] - dmgrid_final[-1])/2.
            ll = loss(dm_pulsewidth, tsamp, k, ch, freq, bw, dmgrid[i], ddm)
            if ll > dm_maxloss:
                dmgrid_final.append(dmgrid[i])
        if maxdm not in dmgrid_final:
            dmgrid_final.append(maxdm)

    return dmgrid_final


@jit(nopython=True)
def loss(dm_pulsewidth, tsamp, k, ch, freq, bw, dm, ddm):
    return 1 - np.sqrt(dt0(dm_pulsewidth, tsamp, k, ch, freq, dm)/dt1(dm_pulsewidth, tsamp, k, ch, freq, bw, dm, ddm))


@jit(nopython=True)
def dt0(dm_pulsewidth, tsamp, k, ch, freq, dm):
    return np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2)


@jit(nopython=True)
def dt1(dm_pulsewidth, tsamp, k, ch, freq, bw, dm, ddm):
    return np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2 + ((k*ddm*bw)/(freq**3.))**2)


def get_uvw_segment(st, segment, pc_mjd=None, pc_radec=None, raw=False):
    """ Returns uvw in units of baselines for a given segment.
    Tuple of u, v, w given with each a numpy array of (nbl, nchan) shape.
    The pc keywords are the (absolute within scan) phase center. One for radec and mjd calc.
    raw defines whether uvw calc for all channels (metadata.freq_orig) or selected (state.freq)
    """

    logger.debug("Getting uvw for segment {0}".format(segment))

    if st.prefs.excludeants:
        takeants = [st.metadata.antids.index(antname) for antname in st.ants]
    else:
        takeants = None

    radec = st.get_radec(pc=pc_radec)
    mjd = st.get_mjd(segment=segment, pc=pc_mjd)
    mjdastropy = time.Time(mjd, format='mjd')

    (ur, vr, wr) = calc_uvw_astropy(datetime=mjdastropy, radec=radec,
                                    xyz=st.metadata.xyz, telescope=st.metadata.telescope,
                                    takeants=takeants)

    if raw:
        u = np.outer(ur, st.metadata.freq_orig * (1e9/constants.c) * (-1))
        v = np.outer(vr, st.metadata.freq_orig * (1e9/constants.c) * (-1))
        w = np.outer(wr, st.metadata.freq_orig * (1e9/constants.c) * (-1))
    else:
        u = np.outer(ur, st.freq * (1e9/constants.c) * (-1))
        v = np.outer(vr, st.freq * (1e9/constants.c) * (-1))
        w = np.outer(wr, st.freq * (1e9/constants.c) * (-1))

    return u.astype('float32'), v.astype('float32'), w.astype('float32')


def calc_uvw_astropy(datetime, radec, xyz, telescope='VLA', takeants=None):
    """ Calculates and returns uvw in meters for a given time and pointing direction.
    datetime is time (astropy.time.Time) to calculate uvw.
    radec is (ra,dec) as tuple in radians.
    Can optionally specify a telescope other than the VLA.
    """
    if telescope == 'JVLA' or 'VLA':
        telescope = 'VLA'

    phase_center = coordinates.SkyCoord(*radec, unit='rad', frame='icrs')

    if takeants is not None:
        antpos = coordinates.EarthLocation(x=xyz[takeants,0], y=xyz[takeants,1], z=xyz[takeants,2], unit='m') 
    else:
        antpos = coordinates.EarthLocation(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], unit='m') 

    if isinstance(datetime, str):
        datetime = time.Time(datetime.replace('/', '-', 2).replace('/', ' '), format='iso')

    tel_p, tel_v = coordinates.EarthLocation.of_site(telescope).get_gcrs_posvel(datetime)
    antpos_gcrs = coordinates.GCRS(antpos.get_gcrs_posvel(datetime)[0],
                                   obstime = datetime, obsgeoloc = tel_p,
                                   obsgeovel = tel_v)

    uvw_frame = phase_center.transform_to(antpos_gcrs).skyoffset_frame()
    antpos_uvw = antpos_gcrs.transform_to(uvw_frame).cartesian

    nant = len(antpos_uvw)
    antpairs = [(i,j) for j in range(nant) for i in range(j)]
    nbl = len(antpairs)
    u = np.empty(nbl, dtype='float32')
    v = np.empty(nbl, dtype='float32')
    w = np.empty(nbl, dtype='float32')
    for ibl, ant in enumerate(antpairs):
        bl = antpos_uvw[ant[1]] - antpos_uvw[ant[0]]
        u[ibl] = bl.y.value
        v[ibl] = bl.z.value
        w[ibl] = bl.x.value

    return u, v, w


def kalman_prep(data):
    """ Use prepared data to calculate noise and kalman inputs.
    Returns tuple (spec_std, sig_ts, kalman_coeffs) that are directly
    input to kalman_significance function.
    """

    from kalman_detector import kalman_prepare_coeffs

    data = np.ma.masked_equal(data, 0j)
    if data.shape[0] > 1:
        spec_std = data.real.mean(axis=3).mean(axis=1).std(axis=0)
    else:
        spec_std = data[0].real.mean(axis=2).std(axis=0)

    if not np.any(spec_std):
        logger.warning("spectrum std all zeros. Not estimating coeffs.")
        kalman_coeffs = []
    else:
        sig_ts, kalman_coeffs = kalman_prepare_coeffs(spec_std.filled(0))

    if not np.all(np.nan_to_num(sig_ts)):
        kalman_coeffs = []

    return spec_std.filled(0), sig_ts, kalman_coeffs


def calc_noise(st, segment, data, chunk=500):
    """ Calculate the noise properties of the data.
    Noise measurement defined as a tuple:
    (startmjd, delta_mjd, segment, imid, noiseperbl,
    zerofrac, imstd).
    """

    from rfpipe.search import grid_image

    startmjd, endmjd = st.segmenttimes[segment]
    deltamjd = st.inttime*len(st.get_search_ints(segment, 0, 0))/(24*3600)
    results = []
    if data.any():
        uvw = get_uvw_segment(st, segment)
        chunk = min(chunk, max(1, st.readints-1))  # ensure at least one measurement
        ranges = list(zip(list(range(0, st.readints-chunk, chunk)),
                          list(range(chunk, st.readints, chunk))))

        for (r0, r1) in ranges:
            imid = (r0+r1)//2
            noiseperbl = estimate_noiseperbl(data[r0:r1])
            imstd = grid_image(data, uvw, st.npixx, st.npixy, st.uvres,
                               'fftw', 1, integrations=imid).std()
            zerofrac = float(len(np.where(data[r0:r1] == 0j)[0]))/data[r0:r1].size
            results.append((startmjd, deltamjd, segment, imid, noiseperbl, zerofrac, imstd))

    return results


def estimate_noiseperbl(data):
    """ Takes large data array and sigma clips it to find noise per bl for
    input to detect_bispectra.
    Takes mean across pols and channels for now, as in detect_bispectra.
    """

    # define noise per baseline for data seen by detect_bispectra or image
    datamean = data.mean(axis=2).imag  # use imaginary part to estimate noise without calibrated, on-axis signal
    noiseperbl = datamean.std()  # measure single noise for input to detect_bispectra
    logger.debug('Measured noise per baseline of {0:.3f}'.format(noiseperbl))
    return noiseperbl


def calc_segment_times(state, scale_nsegment=1.):
    """ Helper function for set_pipeline to define segmenttimes list.
    Forces segment time windows to be fixed relative to integration boundaries.
    Can optionally push nsegment scaling up.
    """

#    stopdts = np.arange(int(math.ceil(state.t_overlap/state.inttime)), state.nints+1,
#                        min(max(1,
#                            int(math.ceil(state.fringetime/state.inttime/scale_nsegment))),
#                        state.nints-int(math.ceil(state.t_overlap/state.inttime))),
#                        dtype=int)[1:]
#    startdts = np.concatenate(([0],
#                              stopdts[:-1]-int(math.ceil(state.t_overlap/state.inttime))))

    stopdts = np.arange(int(round(state.t_overlap/state.inttime)), state.nints+1,
                        min(max(1,
                            int(round(state.fringetime/state.inttime/scale_nsegment))),
                        state.nints-int(round(state.t_overlap/state.inttime))),
                        dtype=int)[1:]
    startdts = np.concatenate(([0],
                              stopdts[:-1]-int(round(state.t_overlap/state.inttime))))

    assert all([len(stopdts), len(startdts)]), ('Could not set segment times. '
                                                'Confirm that: '
                                                't_overlap < scan length ({0} < {1}) '
                                                'and fringetime > inttime ({2} > {3})'
                                                .format(state.t_overlap,
                                                        state.nints*state.inttime,
                                                        state.fringetime,
                                                        state.inttime))

    segmenttimes = []
    for (startdt, stopdt) in zip(state.inttime*startdts, state.inttime*stopdts):
        starttime = time.Time(state.metadata.starttime_mjd, format='mjd').unix + startdt
        stoptime = time.Time(state.metadata.starttime_mjd, format='mjd').unix + stopdt

        # round to nearest ms for vys reading
        dt_1ms = np.round((stoptime-starttime), 3) - (stoptime-starttime)
        if dt_1ms > 1e-1:
            logger.warn("segmenttime calculation getting large deviation from inttime boundaries")
        segmenttimes.append((time.Time(starttime, format='unix').mjd,
                             time.Time(stoptime+dt_1ms, format='unix').mjd))

    return np.array(segmenttimes)


def find_segment_times(state):
    """ Iterates to optimal segment time list, given memory and fringe time limits.
    Segment sizes bounded by fringe time and memory limit,
    Solution found by iterating from fringe time to memory size that fits.
    """

    logger.debug('Total memory of {0} is over limit of {1} '
                 'with {2} segments. Searching to vis/im limits'
                 ' of {3}/{4} GB...'
                 .format(state.memory_total,
                         state.prefs.memory_limit,
                         state.nsegment, state.vismem_limit,
                         state.immem_limit))

    # calculate memory limit to stop iteration
    assert state.memory_total_limit < state.prefs.memory_limit, 'memory_limit of {0} is smaller than best solution of {1}. Try setting maxdm or image memory lower.'.format(state.prefs.memory_limit, state.memory_total_limit)

    if state.memory_total > state.prefs.memory_limit:
        scale_nsegment = 1.
        while state.memory_total > state.prefs.memory_limit:
            scale_nsegment *= state.memory_total/float(state.prefs.memory_limit)
            state._segmenttimes = calc_segment_times(state, scale_nsegment)


def madtostd(array):
    return 1.4826*np.median(np.abs(array-np.median(array)))


def make_transient_params(st, ntr=1, segment=None, dmind=None, dtind=None,
                          i=None, amp=None, lm=None, snr=None, data=None):
    """ Given a state, create ntr randomized detectable transients.
    Returns list of ntr tuples of parameters.
    If data provided, it is used to inject transient at fixed apparent SNR.
    selects random value from dmarr and dtarr.
    Mock transient will have either l or m equal to 0.
    Option exists to overload random selection with fixed segment, dmind, etc.
    """

    segment0 = segment
    dmind0 = dmind
    dtind0 = dtind
    i0 = i
    amp0 = amp
    lm0 = lm
    snr0 = snr

    mocks = []
    for tr in range(ntr):
        if segment is None:
            segment = random.choice(range(st.nsegment))

        if dmind is not None:
            dm = st.dmarr[dmind]
#            dmind = random.choice(range(len(st.dmarr)))
        else:
            dm = np.random.uniform(min(st.dmarr), max(st.dmarr)) # pc /cc

            dmarr = np.array(calc_dmarr(st))
            if dm > np.max(dmarr):
                logging.warning("Dm of injected transient is greater than the max DM searched.")
                dmind = len(dmarr) - 1
            else:
                dmind = np.argmax(dmarr>dm)
            

        if dtind is not None:
            dt = st.inttime*st.dtarr[dtind]
        else:
            #max_width = 0.04/st.inttime
            #dt = st.inttime*np.random.uniform(0, max_width) # s  #like an alias for "dt"
            dt = np.random.uniform(0.001, 0.04)
            if dt < st.inttime:
                dtind = 0
            else:
                boxcar_widths = np.array(st.dtarr)*st.inttime
                if dt > np.max(boxcar_widths):
                    logging.warning("Width of transient is greater than max dt searched.")
                dtind = np.argmin(np.abs(boxcar_widths - dt))

#            else:
#                dtind = int(np.round(np.log2(dt/st.inttime)))
#                if dtind >= len(st.dtarr):
#                    dtind = len(st.dtarr) - 1
#                    logging.warning("Width of transient is greater than max dt searched.")


# TODO: add support for arb dm/dt
#        dm = random.uniform(min(st.dmarr), max(st.dmarr))
#        dt = random.uniform(min(st.dtarr), max(st.dtarr))

        if i is None:
            ints = np.array(st.get_search_ints(segment, dmind, dtind))*st.dtarr[dtind]
            i = np.random.randint(min(ints), max(ints))
            
            #i = np.random.randint(0,st.readints)
            #i = random.choice(st.get_search_ints(segment, dmind, dtind))*st.dtarr[dtind]

        if amp is None:
            if data is None:
                amp = random.uniform(0.1, 0.5)
                logger.info("Setting mock amp to {0}".format(amp))
            else:
                if snr is None:
                    snr = random.uniform(10, 50)
                    # TODO: support flagged data in size calc and injection
                if data.shape != st.datashape:
                    logger.info("Looks like raw data passed in. Selecting and calibrating.")
                    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
                    data = calibration.apply_telcal(st, data.take(takepol, axis=3).take(st.chans, axis=2))
                # noise = madtostd(data[i].real)/np.sqrt(data[i].size*st.dtarr[dtind])
                noise = madtostd(data[i].real)/np.sqrt(data[i].size) #*st.dtarr[dtind])
                #width_factor = (dt//st.inttime)/np.sqrt(st.dtarr[dtind])
                width_factor = (dt/st.inttime)/np.sqrt(st.dtarr[dtind])
                amp = snr*noise*width_factor #*(st.inttime/dt)
                logger.info("Setting mock amp as {0}*{1}*{2}={3}".format(snr, noise, width_factor, amp))
        #else:
        #    if data is None:
        #        logger.info("Setting mock amp to {0}".format(amp))
        #    else:
        #        if data.shape != st.datashape:
        #            logger.info("Looks like raw data passed in. Selecting and calibrating.")
        #            takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
        #            data = calibration.apply_telcal(st, data.take(takepol, axis=3).take(st.chans, axis=2))
        #        bkgrnd = np.mean(data[i].real)
        #        amp = amp0 + bkgrnd 
        #        logger.info("Setting mock amp to {0}+{1} = {2}".format(amp0, bkgrnd, amp))
        #        noise = madtostd(data[i].real)/np.sqrt(data[i].size)
        #        snr_est = amp/(noise*np.sqrt(st.dtarr[dtind]))
        #        logger.info("Estimated SNR is {0}".format(snr_est))

        if lm is None:
            # flip a coin to set either l or m
            if random.choice([0, 1]):
                l = math.radians(random.uniform(-st.fieldsize_deg/2,
                                                st.fieldsize_deg/2))
                m = 0.
            else:
                l = 0.
                m = math.radians(random.uniform(-st.fieldsize_deg/2,
                                                st.fieldsize_deg/2))
        else:
            if lm == -1:
                l = math.radians(random.uniform(-st.fieldsize_deg/2,
                                                st.fieldsize_deg/2))
                m = math.radians(random.uniform(-st.fieldsize_deg/2,
                                                 st.fieldsize_deg/2))
            else:
                assert len(lm) == 2, "lm must be 2-tuple or -1"
                l, m = lm

        mocks.append((segment, i, dm, dt, amp, l, m))
        (segment, dmind, dtind, i, amp, lm, snr) = (segment0, dmind0, dtind0, i0,
                                                    amp0, lm0, snr0)

    return mocks

def make_transient_data(st, amp, i0, dm, dt, ampslope=0.):
    """ Create a dynamic spectrum for given parameters
    amp is apparent (after time gridding) in system units (post calibration)
    i0 is a float for integration relative to start of segment.
    dm/dt are in units of pc/cm3 and seconds, respectively
    ampslope adds to a linear slope up to amp+ampslope at last channel.
    """

    chans = np.arange(len(st.freq))
    model = np.zeros((len(st.freq), st.readints), dtype='complex64')
    ampspec = amp + ampslope*(np.linspace(0, 1, num=len(chans)))

    i = i0 + calc_delay2(st.freq, st.freq.max(), dm)/st.inttime
#     print(i)
    i_f = np.floor(i).astype(int)
    imax = np.ceil(i + dt/st.inttime).astype(int)
    imin = i_f
    i_r = imax - imin
    i_r_max = i_r.max()

    if np.any(i_r == 1):
        ir1 = np.where(i_r == 1)
#         print(ir1)
        model[chans[ir1], i_f[ir1]] += ampspec[chans[ir1]]

    if np.any(i_r == 2):
        ir2 = np.where(i_r == 2)
        i_c = np.ceil(i).astype(int)
        f1 = ((i + dt/st.inttime) - (imax - 1))/(dt/st.inttime)
        f0 = 1 - f1
#         print(ir2)
        model[chans[ir2], i_f[ir2]] += f0[ir2]*ampspec[chans[ir2]]
        model[chans[ir2], i_f[ir2]+1] += f1[ir2]*ampspec[chans[ir2]]

    if i_r_max > 2:
        for x in np.linspace(3, i_r_max, i_r_max-2, dtype=int):
            irx = np.where(i_r == x)
            f_high = ((i + dt/st.inttime) - (imax - 1))/(dt/st.inttime)
            f_low = ((i_f + 1) - i)/(dt/st.inttime)
            n_bet = x - 2
            fs = []
            fs.append(f_low)
            if n_bet > 0:
                f_bet = (1 - f_high - f_low)/n_bet
                for bet in range(n_bet):
                    fs.append(f_bet)
            fs.append(f_high)
            
            for y in range(x):
                model[chans[irx], i_f[irx]+y] += fs[y][irx]*ampspec[chans[irx]]
    return model
