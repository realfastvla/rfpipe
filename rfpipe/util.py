#from __future__ import print_function, division, absolute_import, unicode_literals
#from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open
import numpy as np
import logging
logger = logging.getLogger(__name__)

import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
me = casautil.tools.measures()


def calc_uvw(datetime, radec, antpos, telescope='JVLA'):
    """ Calculates and returns uvw in meters for a given time and pointing direction.
    datetime is time (as string) to calculate uvw (format: '2014/09/03/08:33:04.20')
    radec is (ra,dec) as tuple in units of degrees (format: (180., +45.))
    Can optionally specify a telescope other than the JVLA
    """

    assert '/' in datetime, 'datetime must be in yyyy/mm/dd/hh:mm:ss.sss format'
    assert len(radec) == 2, 'radec must be (ra,dec) tuple in units of degrees'

    direction = me.direction('J2000', str(radec[0])+'deg', str(radec[1])+'deg')
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


def calc_delay(freq, freqref, dm, inttime):
    """ Calculates the delay due to dispersion relative to freqref in integer units of inttime """

    delay = np.zeros(len(freq), dtype=np.int32)

    for i in range(len(freq)):
        delay[i] = np.round(4.1488e-3 * dm * (1./freq[i]**2 - 1./freqref**2)/inttime, 0)

    return delay
