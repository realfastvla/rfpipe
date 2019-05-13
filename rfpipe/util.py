from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
import math
import random
from numba import cuda, guvectorize
from numba import jit, complex64, int64
import pwkit.environments.casa.util as casautil
import sdmpy
from rfpipe import calibration
from astropy import time

import logging
logger = logging.getLogger(__name__)

#qa = casautil.tools.quanta()
me = casautil.tools.measures()


def getsdm(*args, **kwargs):
    """ Wrap sdmpy.SDM to get around schema change error """

    try:
        sdm = sdmpy.SDM(*args, **kwargs)
    except:
        kwargs['use_xsd'] = False
        sdm = sdmpy.SDM(*args, **kwargs)

    return sdm


def phase_shift(data, uvw, dl, dm, ints=None):
    """ Applies a phase shift to data for a given (dl, dm).
    """

    assert data.shape[1] == uvw[0].shape[0]
    assert data.shape[2] == uvw[0].shape[1]
    data = np.require(data, requirements='W')
    if ints is None:
        ints = list(range(len(data)))
    _phaseshift_jit(data, uvw, dl, dm, ints=ints)


@jit(nogil=True, nopython=True, cache=True)
def _phaseshift_jit(data, uvw, dl, dm, ints):

    sh = data.shape
    u, v, w = uvw

    if (dl != 0.) or (dm != 0.):
        for j in range(sh[1]):
            for k in range(sh[2]):
                frot = np.exp(-2j*np.pi*(dl*u[j, k] + dm*v[j, k]))
                for i in ints:
                    for l in range(sh[3]):    # iterate over pols
                        # phasor unwraps phase at (dl, dm) per (bl, chan)
                        data[i, j, k, l] = data[i, j, k, l] * frot


def meantsub(data):
    """ Subtract mean visibility in time.
    """

    # TODO: make outlier resistant to avoid oversubtraction

    _meantsub_jit(np.require(data, requirements='W'))
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


def calc_delay(freq, freqref, dm, inttime, scale=None):
    """ Calculates the delay in integration time bins due to dispersion delay.
    freq is array of frequencies. delay is relative to freqref.
    default scale is 4.1488e-3 as linear prefactor (reproducing for rtpipe<=1.54 requires 4.2e-3).
    Casts to int, so it uses a floor operation, not round.
    """

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


def get_uvw_segment(st, segment):
    """ Returns uvw in units of baselines for a given segment.
    Tuple of u, v, w given with each a numpy array of (nbl, nchan) shape.
    If available, uses a lock to control multithreaded casa measures call.
    """

    logger.debug("Getting uvw for segment {0}".format(segment))
    mjdstr = st.get_segmenttime_string(segment)

    if st.lock is not None:
        st.lock.acquire()
    (ur, vr, wr) = calc_uvw(datetime=mjdstr, radec=st.metadata.radec,
                            antpos=st.metadata.antpos,
                            telescope=st.metadata.telescope)
    if st.lock is not None:
        st.lock.release()

    u = np.outer(ur, st.freq * (1e9/3e8) * (-1))
    v = np.outer(vr, st.freq * (1e9/3e8) * (-1))
    w = np.outer(wr, st.freq * (1e9/3e8) * (-1))

    return u.astype('float32'), v.astype('float32'), w.astype('float32')


def calc_uvw(datetime, radec, antpos, telescope='JVLA'):
    """ Calculates and returns uvw in meters for a given time and pointing direction.
    datetime is time (as string) to calculate uvw (format: '2014/09/03/08:33:04.20')
    radec is (ra,dec) as tuple in units of degrees (format: (180., +45.))
    Can optionally specify a telescope other than the JVLA
    """

    assert '/' in datetime, 'datetime must be in yyyy/mm/dd/hh:mm:ss.sss format'
    assert len(radec) == 2, 'radec must be (ra,dec) tuple in units of degrees'

    direction = me.direction('J2000', str(np.degrees(radec[0]))+'deg',
                             str(np.degrees(radec[1]))+'deg')

    logger.debug('Calculating uvw at {0} for (RA, Dec) = {1}'
                 .format(datetime, radec))
    me.doframe(me.observatory(telescope))
    me.doframe(me.epoch('utc', datetime))
    me.doframe(direction)

    # calc bl
    bls = me.asbaseline(antpos)
    uvwlist = me.expand(me.touvw(bls)[0])[1]['value']

    # define new bl order to match sdm binary file bl order
    u = np.empty(int(len(uvwlist)/3), dtype='float32')
    v = np.empty(int(len(uvwlist)/3), dtype='float32')
    w = np.empty(int(len(uvwlist)/3), dtype='float32')
    nants = len(antpos['m0']['value'])
    ord1 = [i*nants+j for i in range(nants) for j in range(i+1, nants)]
    ord2 = [i*nants+j for j in range(nants) for i in range(j)]
    key = []
    for new in ord2:
        key.append(ord1.index(new))
    for i in range(len(key)):
        u[i] = uvwlist[3*key[i]]
        v[i] = uvwlist[3*key[i]+1]
        w[i] = uvwlist[3*key[i]+2]

    return u, v, w


def calc_noise(st, segment, data, chunk=500):
    """ Calculate the noise properties of the data.
    """

    from rfpipe.search import grid_image

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
            results.append((segment, imid, noiseperbl, zerofrac, imstd))

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
#        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(state.metadata.starttime_mjd+startdt/(24*3600), 'd'),
#                                                   form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
#        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(state.metadata.starttime_mjd+stopdt/(24*3600), 'd'),
#                                                  form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
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
            dt = st.inttime*min(st.dtarr[dtind], 2)  # dt>2 not yet supported
        else:
            #dtind = random.choice(range(len(st.dtarr)))
            dt = st.inttime*np.random.uniform(0, max(st.dtarr)) # s  #like an alias for "dt"
            if dt < st.inttime:
                dtind = 0
            else:    
                dtind = int(np.log2(dt/st.inttime))
                if dtind >= len(st.dtarr):
                    dtind = len(st.dtarr) - 1
                    logging.warning("Width of transient is greater than max dt searched.")


# TODO: add support for arb dm/dt
#        dm = random.uniform(min(st.dmarr), max(st.dmarr))
#        dt = random.uniform(min(st.dtarr), max(st.dtarr))

        if i is None:
            i = random.choice(st.get_search_ints(segment, dmind, dtind))

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
                noise = madtostd(data[i].real)/np.sqrt(data[i].size*st.dtarr[dtind])
                amp = snr*noise  #*(st.inttime/dt)
                logger.info("Setting mock amp as {0}*{1}={2}".format(snr, noise, amp))
                

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
            assert len(lm) == 2, "lm must be 2-tuple"
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
#    print(i)
    i_f = np.floor(i).astype(int)
    imax = np.ceil(i + dt/st.inttime).astype(int)
    imin = i_f
    i_r = imax - imin
#    print(i_r)
    if np.any(i_r == 1):
        ir1 = np.where(i_r == 1)
#        print(ir1)
        model[chans[ir1], i_f[ir1]] += ampspec[chans[ir1]]

    if np.any(i_r == 2):
        ir2 = np.where(i_r == 2)
        i_c = np.ceil(i).astype(int)
        f1 = (dt/st.inttime - (i_c - i))/(dt/st.inttime)
        f0 = 1 - f1
#        print(np.vstack((ir2, f0[ir2], f1[ir2])).transpose())
        model[chans[ir2], i_f[ir2]] += f0[ir2]*ampspec[chans[ir2]]
        model[chans[ir2], i_f[ir2]+1] += f1[ir2]*ampspec[chans[ir2]]

    if np.any(i_r == 3):
        ir3 = np.where(i_r == 3)
        f2 = (i + dt/st.inttime - (imax - 1))/(dt/st.inttime)
        f0 = ((i_f + 1) - i)/(dt/st.inttime)
        f1 = 1 - f2 - f0
#        print(np.vstack((ir3, f0[ir3], f1[ir3], f2[ir3])).transpose())
        model[chans[ir3], i_f[ir3]] += f0[ir3]*ampspec[chans[ir3]]
        model[chans[ir3], i_f[ir3]+1] += f1[ir3]*ampspec[chans[ir3]]
        model[chans[ir3], i_f[ir3]+2] += f2[ir3]*ampspec[chans[ir3]]
    if np.any(i_r >= 4):
        logger.warning("Some channels broadened more than 3 integrations, which is not yet supported.")

    return model
