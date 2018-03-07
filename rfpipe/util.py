from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
from numba import cuda, guvectorize
from numba import jit, complex64, int64
import pwkit.environments.casa.util as casautil

import logging
logger = logging.getLogger(__name__)

qa = casautil.tools.quanta()
me = casautil.tools.measures()


def phase_shift(data, uvw, dl, dm):
    """ Applies a phase shift to data for a given (dl, dm).
    """

    data = np.require(data, requirements='W')
    _phaseshift_jit(data, uvw, dl, dm)


@jit(nogil=True, nopython=True)
def _phaseshift_jit(data, uvw, dl, dm):

    sh = data.shape
    u, v, w = uvw

    if (dl != 0.) or (dm != 0.):
        for j in range(sh[1]):
            for k in range(sh[2]):
                frot = np.exp(-2j*np.pi*(dl*u[j, k] + dm*v[j, k]))
                for i in range(sh[0]):    # iterate over pols
                    for l in range(sh[3]):
                        # phasor unwraps phase at (dl, dm) per (bl, chan)
                        data[i, j, k, l] = data[i, j, k, l] * frot


def meantsub(data, parallel=False):
    """ Subtract mean visibility in time.
    Parallel controls use of multithreaded algorithm.
    """

    if parallel:
        _ = _meantsub_gu(np.require(np.swapaxes(data, 0, 3), requirements='W'))
    else:
        _meantsub_jit(np.require(data, requirements='W'))
    return data


@jit(nogil=True, nopython=True)
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


@guvectorize([str("void(complex64[:])")], str("(m)"),
             target='parallel', nopython=True)
def _meantsub_gu(data):
    b""" Subtract time mean while ignoring zeros.
    Vectorizes over time axis.
    Assumes time axis is last so use np.swapaxis(0,3) when passing visibility array in
    """

    ss = complex64(0)
    weight = int64(0)
    for i in range(data.shape[0]):
        ss += data[i]
        if data[i] != 0j:
            weight += 1
    mean = ss/weight
    for i in range(data.shape[0]):
        data[i] -= mean


@cuda.jit
def meantsub_cuda(data):
    """ Calculate mean in time (ignoring zeros) and subtract in place """

    x, y, z = cuda.grid(3)
    nint, nbl, nchan, npol = data.shape
    if x < nbl and y < nchan and z < npol:
        sum = complex64(0)
        weight = 0
        for i in range(nint):
            sum = sum + data[i, x, y, z]
            if data[i, x, y, z] == 0j:
                weight = weight + 1
        mean = sum/weight
        for i in range(nint):
            data[i, x, y, z] = data[i, x, y, z] - mean


def calc_delay(freq, freqref, dm, inttime, scale=None):
    """ Calculates the delay in integration time bins due to dispersion delay.
    freq is array of frequencies. delay is relative to freqref.
    default scale is 4.1488e-3 as linear prefactor (reproducing for rtpipe<=1.54 requires 4.2e-3).
    """

    scale = 4.1488e-3 if not scale else scale
    delay = np.zeros(len(freq), dtype=np.int32)

    for i in range(len(freq)):
        delay[i] = np.round(scale * dm * (1./freq[i]**2 - 1./freqref**2)/inttime, 0)

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

    # width functions and loss factor
    dt0 = lambda dm: np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2)
    dt1 = lambda dm, ddm: np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2 + ((k*ddm*bw)/(freq**3.))**2)
    loss = lambda dm, ddm: 1 - np.sqrt(dt0(dm)/dt1(dm, ddm))
    loss_cordes = lambda ddm, dfreq, dm_pulsewidth, freq: 1 - (np.sqrt(np.pi) / (2 * 6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))) * erf(6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))  # not quite right for underresolved pulses

    if maxdm == 0:
        return [0]
    else:
        # iterate over dmgrid to find optimal dm values. go higher than maxdm to be sure final list includes full range.
        dmgrid = np.arange(mindm, maxdm, 0.05)
        dmgrid_final = [dmgrid[0]]
        for i in range(len(dmgrid)):
            ddm = (dmgrid[i] - dmgrid_final[-1])/2.
            ll = loss(dmgrid[i],ddm)
            if ll > dm_maxloss:
                dmgrid_final.append(dmgrid[i])

    return dmgrid_final


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


def calc_segment_times(state, scale_nsegment=1.):
    """ Helper function for set_pipeline to define segmenttimes list.
    Forces segment time windows to be fixed relative to integration boundaries.
    Can optionally push nsegment scaling up.
    """

#    stopdts = np.linspace(int(round(state.t_overlap/state.inttime)), state.nints,
#                          nsegment+1)[1:]  # nseg+1 keeps at least one seg
#    startdts = np.concatenate(([0],
#                              stopdts[:-1]-int(round(state.t_overlap/state.inttime))))
    # or force on integer boundaries?

    stopdts = np.arange(int(round(state.t_overlap/state.inttime)), state.nints+1,
                        min(max(1,
                            int(round(state.fringetime/state.inttime/scale_nsegment))),
                        state.nints-int(round(state.t_overlap/state.inttime))),
                        dtype=int)[1:]
    startdts = np.concatenate(([0],
                              stopdts[:-1]-int(round(state.t_overlap/state.inttime))))

    assert all([len(stopdts), len(startdts)]), ('Could not set segment times.'
                                                't_overlap may be longer than '
                                                'nints or fringetime shorter '
                                                'than inttime.')

    segmenttimes = []
    for (startdt, stopdt) in zip(state.inttime*startdts, state.inttime*stopdts):
        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(state.metadata.starttime_mjd+startdt/(24*3600), 'd'),
                                                   form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(state.metadata.starttime_mjd+stopdt/(24*3600), 'd'),
                                                  form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        segmenttimes.append((starttime, stoptime))

    return np.array(segmenttimes)


def find_segment_times(state):
    """ Iterates to optimal segment time list, given memory and fringe time limits.
    Segment sizes bounded by fringe time and memory limit,
    Solution found by iterating from fringe time to memory size that fits.
    """

    # calculate memory limit to stop iteration
    assert state.memory_total_limit < state.prefs.memory_limit, 'memory_limit of {0} is smaller than best solution of {1}. Try setting maxdm/npix_max lower.'.format(state.prefs.memory_limit, state.immem_limit+state.vismem_limit)

    if state.memory_total > state.prefs.memory_limit:
        logger.debug("Total memory is larger than memory limit."
                     "Iterating to smaller segment size.")
        scale_nsegment = 1.
        while state.memory_total > state.prefs.memory_limit:
            scale_nsegment *= state.memory_total/float(state.prefs.memory_limit)
            state._segmenttimes = calc_segment_times(state, scale_nsegment)


def madtostd(array):
    return 1.4826*np.median(np.abs(array-np.median(array)))
