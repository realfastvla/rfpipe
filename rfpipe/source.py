from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import os.path
import numpy as np
import sdmpy
from astropy import time
import pwkit.environments.casa.util as casautil
from rfpipe import util, calibration, search

import logging
logger = logging.getLogger(__name__)

qa = casautil.tools.quanta()
default_timeout = 10


def data_prep(st, data):
    """ Applies calibration, flags, and subtracts time mean for data.
    """

    # ** need to make this portable or at least reimplement in rfpipe
    if np.any(data):
        if st.metadata.datasource != 'sim':
            if os.path.exists(st.gainfile):
                calibration.apply_telcal(st, data)
            else:
                logger.warn('Telcal file not found. No calibration being applied.')
        else:
            logger.info('Not applying telcal solutions for simulated data')

        # ** dataflag points to rtpipe for now
        util.dataflag(st, data)

        logger.info('Subtracting mean visibility in time.')
        if st.prefs.timesub == 'mean':
            util.meantsub(data)

        return data
    else:
        return []


def read_segment(st, segment, cfile=None, timeout=default_timeout):
    """ Read a segment of data and do low-level data preparation.
    Optionally applies antenna flags, rolls out of order data, and can
    downsample data in time or frequency.
    cfile and timeout are specific to vys data.
    """

    if st.metadata.datasource == 'sdm':
        data_read = read_bdf_segment(st, segment)
    elif st.metadata.datasource == 'vys':
        data_read = read_vys_segment(st, segment, cfile=cfile, timeout=timeout)
    elif st.metadata.datasource == 'sim':
        data_read = simulate_segment(st)
    else:
        logger.error('Datasource {0} not recognized.'
                     .format(st.metadata.datasource))

    # read Flag.xml and apply flags for given ant/time range
    if st.prefs.applyonlineflags and st.metadata.datasource == 'sdm':

        sdm = getsdm(st.metadata.filename, bdfdir=st.metadata.bdfdir)
        scan = sdm.scan(st.metadata.scan)

        # segment flagged from logical OR from (start, stop) flags
        t0, t1 = st.segmenttimes[segment]
        # 0=bad, 1=good. axis=0 is time axis.
        flags = scan.flags([t0, t1]).all(axis=0)

        if not flags.all():
            logger.info('Found antennas to flag in time range {0}-{1} '
                        .format(t0, t1))
            data_read = np.where(flags[None, :, None, None] == 1,
                                 data_read, 0j)
        else:
            logger.info('No flagged antennas in time range {0}-{1} '
                        .format(t0, t1))
    else:
        logger.info('Not applying online flags.')

    # optionally integrate (downsample)
    if ((st.prefs.read_tdownsample > 1) or (st.prefs.read_fdownsample > 1)):
        data_read2 = np.zeros(st.datashape, dtype='complex64')
        if st.prefs.read_tdownsample > 1:
            logger.info('Downsampling in time by {0}'
                        .format(st.prefs.read_tdownsample))
            for i in range(st.datashape[0]):
                data_read2[i] = data_read[
                    i*st.prefs.read_tdownsample:(i+1)*st.prefs.read_tdownsample].mean(axis=0)
        if st.prefs.read_fdownsample > 1:
            logger.info('Downsampling in frequency by {0}'
                        .format(st.prefs.read_fdownsample))
            for i in range(st.datashape[2]):
                data_read2[:, :, i, :] = data_read[
                    :, :, i * st.prefs.read_fdownsample:(i+1)*st.prefs.read_fdownsample].mean(axis=2)
        data_read = data_read2

    # optionally add transients
    if st.prefs.simulated_transient is not None:
        for params in st.prefs.simulated_transient:
            (amp, i0, dm, dt, l, m) = params
            logger.info("Adding transient with Amp {0} at int {1}, DM {2}, "
                        "dt {3} and l,m={4},{5}".format(amp, i0, dm, dt, l, m))
            model = generate_transient(st, amp, i0, dm, dt)
            search.phase_shift(data_read, st.get_uvw_segment(segment), l, m)
            data_read += model.transpose()[:, None, :, None]
            search.phase_shift(data_read, st.get_uvw_segment(segment), -l, -m)

#    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
#    logger.debug('Selecting pols {0}'.format(st.pols))
#    return data.take(st.chans, axis=2).take(takepol, axis=3)

    return data_read.take(st.chans, axis=2)


def read_vys_segment(st, seg, cfile=None, timeout=default_timeout):
    """ Read segment seg defined by state st from vys stream.
    Uses vysmaw application timefilter to receive multicast messages and pull
    spectra on the CBE.
    """

    # TODO: support for time downsampling

    try:
        import vysmaw_reader
    except ImportError:
        logger.error('ImportError for vysmaw app. Need vysmaw to '
                     'consume vys data.')

    t0 = time.Time(st.segmenttimes[seg][0], format='mjd', precision=9).unix
    t1 = time.Time(st.segmenttimes[seg][1], format='mjd', precision=9).unix
    logger.info('Reading {0} ints of size {1} s from {2} - {3} unix seconds'
                .format(st.readints, st.metadata.inttime, t0, t1))

    data = np.empty((st.readints, st.metadata.nbl_orig,
                     st.metadata.nchan_orig, st.metadata.npol_orig),
                    dtype='complex64', order='C')
    with vysmaw_reader.Reader(t0, t1, cfile=cfile) as reader:
        data[:] = reader.readwindow(nant=st.nants, nspw=st.nspw,
                                    nchan=st.metadata.spw_nchan[0],
                                    npol=st.npol,
                                    inttime_micros=st.metadata.inttime*1e6,
                                    timeout=timeout,
                                    excludeants=st.prefs.excludeants)

    return data


def read_bdf_segment(st, segment):
    """ Uses sdmpy to reads bdf (sdm) format data into numpy array in given
    segment.
    """

    assert segment < st.nsegment, ('segment {0} is too big for nsegment {1}'
                                   .format(segment, st.nsegment))

    # define integration range
    nskip = (24*3600*(st.segmenttimes[segment, 0]
                      - st.metadata.starttime_mjd)/st.metadata.inttime).astype(int)
    logger.info('Reading scan {0}, segment {1}/{2}, times {3} to {4}'
                .format(st.metadata.scan, segment, len(st.segmenttimes)-1,
                        qa.time(qa.quantity(st.segmenttimes[segment, 0], 'd'),
                                form=['hms'], prec=9)[0],
                        qa.time(qa.quantity(st.segmenttimes[segment, 1], 'd'),
                                form=['hms'], prec=9)[0]))
    data = read_bdf(st, nskip=nskip).astype('complex64')

    return data


def simulate_segment(st, loc=0., scale=1.):
    """ Simulates visibilities for a segment.
    """
    logger.info('Simulating data with shape {0}'.format(st.datashape))

    data_read = np.zeros(shape=st.datashape, dtype='complex64')
    data_read.real = np.random.normal(loc=loc, scale=scale, size=st.datashape)
    data_read.imag = np.random.normal(loc=loc, scale=scale, size=st.datashape)

    return data_read


def read_bdf(st, nskip=0):
    """ Uses sdmpy to read a given range of integrations from sdm of given scan.
    readints=0 will read all of bdf (skipping nskip).
    Returns data in increasing frequency order.
    """

    assert os.path.exists(st.metadata.filename), ('sdmfile {0} does not exist'
                                                  .format(st.metadata.filename))
    assert st.metadata.bdfstr, ('bdfstr not defined for scan {0}'
                                .format(st.metadata.scan))

    logger.info('Reading %d ints starting at int %d' % (st.readints, nskip))
    sdm = getsdm(st.metadata.filename, bdfdir=st.metadata.bdfdir)
    scan = sdm.scan(st.metadata.scan)
    data = np.empty((st.readints, st.metadata.nbl_orig, st.metadata.nchan_orig,
                     st.metadata.npol_orig), dtype='complex64', order='C')

    sortind = np.argsort(st.metadata.spw_reffreq)
    for i in range(nskip, nskip+st.readints):
        read = scan.bdf.get_integration(i).get_data(spwidx='all', type='cross')
        data[i] = read.take(sortind, axis=1).reshape(st.metadata.nbl_orig,
                                                     st.metadata.nchan_orig,
                                                     st.metadata.npol_orig)

#    data[:] = scan.bdf.get_data(trange=[nskip, nskip+st.readints]).reshape(data.shape)

    return data


def sdm_sources(sdmname):
    """ Use sdmpy to get all sources and ra,dec per scan as dict """

    sdm = getsdm(sdmname)
    sourcedict = {}

    for row in sdm['Field']:
        src = str(row.fieldName)
        sourcenum = int(row.sourceId)
        direction = str(row.referenceDir)
        # skip first two values in string
        (ra, dec) = [float(val) for val in direction.split(' ')[3:]]

        sourcedict[sourcenum] = {}
        sourcedict[sourcenum]['source'] = src
        sourcedict[sourcenum]['ra'] = ra
        sourcedict[sourcenum]['dec'] = dec

    return sourcedict


def getsdm(*args, **kwargs):
    """ Wrap sdmpy.SDM to get around schema change error """

    try:
        sdm = sdmpy.SDM(*args, **kwargs)
    except:
        kwargs['use_xsd'] = False
        sdm = sdmpy.SDM(*args, **kwargs)

    return sdm


def generate_transient(st, amp, i0, dm, dt):
    """ Create a dynamic spectrum for given parameters
    amp is in system units (post calibration)
    i0 is a float for integration relative to start of segment.
    dm/dt are in units of pc/cm3 and seconds, respectively
    """

    model = np.zeros((st.nchan, st.readints), dtype='float')
    chans = np.arange(st.nchan)

    i = i0 + (4.1488e-3 * dm * (1/st.freq**2 - 1/st.freq.max()**2))/st.inttime
    i_f = np.floor(i).astype(int)
    imax = np.ceil(i + dt/st.inttime).astype(int)
    imin = i_f
    i_r = imax - imin
#    print(i_r)
    if np.any(i_r == 1):
        ir1 = np.where(i_r == 1)
#        print(ir1)
        model[chans[ir1], i_f[ir1]] += amp

    if np.any(i_r == 2):
        ir2 = np.where(i_r == 2)
        i_c = np.ceil(i).astype(int)
        f1 = (dt/st.inttime - (i_c - i))/(dt/st.inttime)
        f0 = 1 - f1
#        print(np.vstack((ir2, f0[ir2], f1[ir2])).transpose())
        model[chans[ir2], i_f[ir2]] += f0[ir2]*amp
        model[chans[ir2], i_f[ir2]+1] += f1[ir2]*amp

    if np.any(i_r == 3):
        ir3 = np.where(i_r == 3)
        f2 = (i + dt/st.inttime - (imax - 1))/(dt/st.inttime)
        f0 = ((i_f + 1) - i)/(dt/st.inttime)
        f1 = 1 - f2 - f0
#        print(np.vstack((ir3, f0[ir3], f1[ir3], f2[ir3])).transpose())
        model[chans[ir3], i_f[ir3]] += f0[ir3]*amp
        model[chans[ir3], i_f[ir3]+1] += f1[ir3]*amp
        model[chans[ir3], i_f[ir3]+2] += f2[ir3]*amp

    return model
