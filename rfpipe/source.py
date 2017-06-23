#from __future__ import print_function, division, absolute_import, unicode_literals
#from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

import os.path
from lxml.etree import XMLSyntaxError
import numpy as np
import sdmpy
from astropy import time

import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
default_timeout = 10


def sdm_sources(sdmname):
    """ Use sdmpy to get all sources and ra,dec per scan as dict """

    sdm = getsdm(sdmname)
    sourcedict = {}

    for row in sdm['Field']:
        src = str(row.fieldName)
        sourcenum = int(row.sourceId)
        direction = str(row.referenceDir)
        (ra,dec) = [float(val) for val in direction.split(' ')[3:]]  # skip first two values in string

        sourcedict[sourcenum] = {}
        sourcedict[sourcenum]['source'] = src
        sourcedict[sourcenum]['ra'] = ra
        sourcedict[sourcenum]['dec'] = dec

    return sourcedict


def getsdm(*args, **kwargs):
    """ Wrap sdmpy.SDM to get around schema change error """

    try:
        sdm = sdmpy.SDM(*args, **kwargs)
    except XMLSyntaxError:
        kwargs['use_xsd'] = False
        sdm = sdmpy.SDM(*args, **kwargs)

    return sdm


def read_segment(st, segment, cfile=None, timeout=default_timeout):
    """ Read a segment of data, apply antenna flags, calibration and other data preparation.
    cfile and timeout are specific to vys data.
    """

    if ...:
        data_read = read_bdf_segment(st, segment)
    else ... :  
        data_read = read_vys_segment(st, segment, cfile=cfile, timeout=timeout)

    # read Flag.xml and apply flags for given ant/time range
    if st.prefs.applyonlineflags:

        sdm = getsdm(st.metadata.filename, bdfdir=st.metadata.bdfdir)
        scan = sdm.scan(st.metadata.scan)

        # segment flagged from logical OR from (start, stop) flags
        t0,t1 = st.segmenttimes[segment]
        flags = scan.flags([t0,t1]).all(axis=0)  # 0=bad, 1=good. axis=0 is time axis.

        if not flags.all():
            logger.info('Found antennas to flag in time range {0}-{1} '.format(t0, t1))
            data_read = np.where(flags[None,:,None, None] == 1, data_read, 0j)
        else:
            logger.info('No flagged antennas in time range {0}-{1} '.format(t0, t1))
    else:
        logger.info('Not applying online flags.')

    # If spw are rolled, roll them to increasing frequency order
    dfreq = np.array([st.metadata.spw_reffreq[i+1] - st.metadata.spw_reffreq[i]
                      for i in range(len(st.metadata.spw_reffreq)-1)])
    dfreqneg = [df for df in dfreq if df < 0]     # not a perfect test of permutability!

    if len(dfreqneg) <= 1:
        if len(dfreqneg) == 1:
            logger.info('Rolling spw frequencies to increasing order: %s'
                        % str(st.metadata.spw_reffreq))
            rollch = np.sum([st.metadata.spw_nchan[ss]
                             for ss in range(np.where(dfreq < 0)[0][0]+1)])
            data_read = np.roll(data_read, rollch, axis=2)
    else:
        raise StandardError('SPW out of order and can\'t be permuted '
                            'to increasing order: %s'
                            % str(st.metadata.spw_reffreq))

    # optionally integrate (downsample)
    if ((st.prefs.read_tdownsample > 1) or (st.prefs.read_fdownsample > 1)):
        raise NotImplementedError

        sh = data_read.shape
        tsize = sh[0]/st.prefs.read_tdownsample
        fsize = sh[2]/st.prefs.read_fdownsample
        data2 = np.zeros((tsize, sh[1], fsize, sh[3]), dtype='complex64')
        if st.prefs.read_tdownsample > 1:
            logger.info('Downsampling in time by {0}'.format(st.prefs.read_tdownsample))
            for i in range(tsize):
                data2[i] = data[
                    i*st.prefs.read_tdownsample:(i+1)*st.prefs.read_tdownsample].mean(axis=0)
        if st.prefs.read_fdownsample > 1:
            logger.info('Downsampling in frequency by {0}'.format(st.prefs.read_fdownsample))
            for i in range(fsize):
                data2[:, :, i, :] = data[
                    :, :, i * st.prefs.read_fdownsample:(i+1)*st.prefs.read_fdownsample].mean(axis=2)
        data = data2

#    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
#    logger.debug('Selecting pols {0}'.format(st.pols))
#    return data.take(st.chans, axis=2).take(takepol, axis=3)

    return data_read.take(st.chans, axis=2)


def read_vys_seg(st, seg, cfile=None, timeout=default_timeout):
    """ Read segment seg defined by state st from vys stream.
    Uses vysmaw application timefilter to receive multicast messages and pull spectra on the CBE.
    """

    try:
        import vysmaw_reader
    except ImportError:
        logger.error('ImportError for vysmaw app. Need vysmaw to consume vys data.')

    t0 = time.Time(st.segmenttimes[seg][0], format='mjd', precision=9).unix
    t1 = time.Time(st.segmenttimes[seg][1], format='mjd', precision=9).unix
    logger.info('Reading %d ints of size %f s from %d - %d unix seconds' % (st.readints, st.metadata.inttime, t0, t1))

    data = np.empty( (st.readints, st.metadata.nbl_orig, st.metadata.nchan_orig, st.metadata.npol_orig), dtype='complex64', order='C')
    with vysmaw_reader.Reader(t0, t1, cfile=cfile) as reader:
        data[:] = reader.readwindow(nant=st.nants, nspw=st.nspw, nchan=st.metadata.spw_nchan[0], npol=st.npol, 
                                    inttime_micros=st.metadata.inttime*1e6, timeout=timeout, excludeants=st.prefs.excludeants)

    return data


def read_bdf_segment(st, segment):
    """ Uses sdmpy to reads bdf (sdm) format data into numpy array in given segment.
    """

    assert segment < st.nsegment, 'segment {0} is too big for nsegment {1}' % (segment, st.nsegment)

    # define integration range
    nskip = (24*3600*(st.segmenttimes[segment, 0] - st.metadata.starttime_mjd) / st.metadata.inttime).astype(int)
    logger.info('Reading scan {0}, segment {1}/{2}, times {3} to {4}'.format(st.metadata.scan, segment, len(st.segmenttimes)-1,
                                                                             qa.time(qa.quantity(st.segmenttimes[segment, 0], 'd'),
                                                                                     form=['hms'], prec=9)[0],
                                                                             qa.time(qa.quantity(st.segmenttimes[segment, 1], 'd'),
                                                                                     form=['hms'], prec=9)[0]))
    data = read_bdf(st, nskip=nskip).astype('complex64')

    return data


def read_bdf(st, nskip=0):
    """ Uses sdmpy to read a given range of integrations from sdm of given scan.
    readints=0 will read all of bdf (skipping nskip).
    """

    assert os.path.exists(st.metadata.filename), 'sdmfile {0} does not exist'.format(st.metadata.filename)
    assert st.metadata.bdfstr, 'bdfstr not defined for scan {0}'.format(st.metadata.scan)

    logger.info('Reading %d ints starting at int %d' % (st.readints, nskip))
    sdm = getsdm(st.metadata.filename, bdfdir=st.metadata.bdfdir)
    scan = sdm.scan(st.metadata.scan)

    data = np.empty( (st.readints, st.metadata.nbl_orig, st.metadata.nchan_orig, st.metadata.npol_orig), dtype='complex64', order='C')
    data[:] = scan.bdf.get_data(trange=[nskip, nskip+st.readints]).reshape(data.shape)

    return data
