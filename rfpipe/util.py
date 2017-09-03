from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

import numpy as np
from numba import cuda, guvectorize
from numba import jit, complex64, int64

import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
me = casautil.tools.measures()

##
## data prep
##

def dataflag(st, data):
    """ Flagging data in single process 
    """

    import rtlib_cython as rtlib

    # **hack!**
    d = {'dataformat': 'sdm', 'ants': [int(ant.lstrip('ea')) for ant in st.ants], 'excludeants': st.prefs.excludeants, 'nants': len(st.ants)}

    for flag in st.prefs.flaglist:
        mode, sig, conv = flag
        for spw in st.spw:
            chans = np.arange(st.metadata.spw_nchan[spw]*spw, st.metadata.spw_nchan[spw]*(1+spw))
            for pol in range(st.npol):
                status = rtlib.dataflag(data, chans, pol, d, sig, mode, conv)
                logger.info(status)

    # hack to get rid of bad spw/pol combos whacked by rfi
    if st.prefs.badspwpol:
        logger.info('Comparing overall power between spw/pol. Removing those with {0} times typical value'.format(st.prefs.badspwpol))
        spwpol = {}
        for spw in st.spw:
            chans = np.arange(st.metadata.spw_nchan[spw]*spw, st.metadata.spw_nchan[spw]*(1+spw))
            for pol in range(st.npol):
                spwpol[(spw, pol)] = np.abs(data[:,:,chans,pol]).std()
        
        meanstd = np.mean(spwpol.values())
        for (spw,pol) in spwpol:
            if spwpol[(spw, pol)] > st.prefs.badspwpol*meanstd:
                logger.info('Flagging all of (spw %d, pol %d) for excess noise.' % (spw, pol))
                chans = np.arange(st.metadata.spw_nchan[spw]*spw, st.metadata.spw_nchan[spw]*(1+spw))
                data[:,:,chans,pol] = 0j


def meantsub(data, mode='multi'):
    """ Subtract mean visibility in time.
    Option to use single or multi threaded version of algorithm.
    """

    if mode == 'single':
        _meantsub_jit(np.require(data, requirements='W'))
        return data
    elif mode == 'multi':
        _ = _meantsub_gu(np.require(np.swapaxes(data, 0, 3), requirements='W'))
        return data
    else:
        logger.error('No such dedispersion mode.')


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
#    return data


@guvectorize(["void(complex64[:])"], '(m)', target='parallel', nopython=True)
def _meantsub_gu(data):
    """ Subtract time mean while ignoring zeros.
    Vectorizes over time axis.
    Assumes time axis is last so use np.swapaxis(0,3) when passing visibility array in """

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

    x,y,z = cuda.grid(3)
    nint, nbl, nchan, npol = data.shape
    if x < nbl and y < nchan and z < npol:
        sum = complex64(0)
        weight = 0
        for i in range(nint):
            sum = sum + data[i, x, y, z]
            if data[i,x,y,z] == 0j:
                weight = weight + 1
        mean = sum/weight
        for i in range(nint):
            data[i, x, y, z] = data[i, x, y, z] - mean


def calc_delay(freq, freqref, dm, inttime, scale=None):
    """ Calculates the delay due to dispersion relative to freqref in integer units of inttime.
    default scale is 4.1488e-3 as linear prefactor (reproducing for rtpipe<=1.54 requires 4.2e-3).
    """

    scale = 4.1488e-3 if not scale else scale
    delay = np.zeros(len(freq), dtype=np.int32)

    for i in range(len(freq)):
        delay[i] = np.round(scale * dm * (1./freq[i]**2 - 1./freqref**2)/inttime, 0)

    return delay


def calc_uvw(datetime, radec, antpos, telescope='JVLA'):
    """ Calculates and returns uvw in meters for a given time and pointing direction.
    datetime is time (as string) to calculate uvw (format: '2014/09/03/08:33:04.20')
    radec is (ra,dec) as tuple in units of degrees (format: (180., +45.))
    Can optionally specify a telescope other than the JVLA
    """

    assert '/' in datetime, 'datetime must be in yyyy/mm/dd/hh:mm:ss.sss format'
    assert len(radec) == 2, 'radec must be (ra,dec) tuple in units of degrees'

    direction = me.direction('J2000', str(np.degrees(radec[0]))+'deg', str(np.degrees(radec[1]))+'deg')

    logger.debug('Calculating uvw at %s for (RA, Dec) = %s' % (datetime, radec))
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
    ord1 = [i*nants+j for i in range(nants) for j in range(i+1,nants)]
    ord2 = [i*nants+j for j in range(nants) for i in range(j)]
    key=[]
    for new in ord2:
        key.append(ord1.index(new))
    for i in range(len(key)):
        u[i] = uvwlist[3*key[i]]
        v[i] = uvwlist[3*key[i]+1]
        w[i] = uvwlist[3*key[i]+2]

    return u, v, w


def madtostd(array):
    return 1.4826*np.median(np.abs(array-np.median(array)))
