from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import os.path
import numpy as np
import sdmpy
from astropy import time
import pwkit.environments.casa.util as casautil
from rfpipe import util, calibration

import logging
logger = logging.getLogger(__name__)

qa = casautil.tools.quanta()
default_timeout = 10  # multiple of read time in seconds to wait


def data_prep(st, segment, data):
    """ Applies calibration, flags, and subtracts time mean for data.
    """

    mode = 'single' if st.prefs.nthread == 1 else 'multi'

    if np.any(data):
        data = prep_standard(st, segment, data)

        # TODO: allow parallel execution with apply_telcal2
        if st.metadata.datasource != 'sim':
            if os.path.exists(st.gainfile):
                data = calibration.apply_telcal(st, np.require(data, requirements='W'))
            else:
                logger.warn('Telcal file not found. No calibration to apply.')
        else:
            logger.info('Not applying telcal solutions for simulated data')

        # TODO: update flagging from rtpipe dataflag
        data = util.dataflag(st, np.require(data, requirements='W'))

        if st.prefs.timesub == 'mean':
            logger.info('Subtracting mean visibility in time.')
            data = util.meantsub(data, mode=mode)
        else:
            logger.info('No visibility subtraction done.')

        if st.prefs.savenoise:
            logger.warn("Saving of noise properties not implemented yet.")

    return data


def read_segment(st, segment, cfile=None, timeout=default_timeout):
    """ Read a segment of data and do low-level data preparation.
    Optionally applies antenna flags, rolls out of order data, and can
    downsample data in time or frequency.
    cfile and timeout are specific to vys data.
    """

    # assumed read shape (st.readints, st.nbl, st.metadata.nchan_orig, st.npol)
    if st.metadata.datasource == 'sdm':
        data_read = read_bdf_segment(st, segment)
    elif st.metadata.datasource == 'vys':
        data_read = read_vys_segment(st, segment, cfile=cfile, timeout=timeout)
    elif st.metadata.datasource == 'sim':
        data_read = simulate_segment(st)
    else:
        logger.error('Datasource {0} not recognized.'
                     .format(st.metadata.datasource))

    if not np.any(data_read):
        logger.info('No data read.')
        return np.array([])
    else:
        return data_read


def prep_standard(st, segment, data_read):
    """ Common first data prep stages, incl
    online flags, resampling, and mock transients.
    """

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
                data_read2[:, :, i, :] = data_read[:, :, i*st.prefs.read_fdownsample:(i+1)*st.prefs.read_fdownsample].mean(axis=2)
        data_read = data_read2

    # optionally add transients
    if st.prefs.simulated_transient is not None:
        assert isinstance(st.prefs.simulated_transient, list)

        uvw = util.get_uvw_segment(st, segment)
        for params in st.prefs.simulated_transient:
            assert len(params) == 7
            (mock_segment, i0, dm, dt, amp, l, m) = params
            if segment == mock_segment:
                logger.info("Adding transient to segment {0} at int {1}, DM {2}, "
                            "dt {3} with amp {4} and l,m={5},{6}"
                            .format(mock_segment, i0, dm, dt, amp, l, m))
                try:
                    model = np.require(np.broadcast_to(generate_transient(st, amp, i0, dm, dt)
                                                       .transpose()[:, None, :, None],
                                                       data_read.shape),
                                       requirements='W')
                except IndexError:
                    logger.warn("IndexError while adding transient. Skipping...")
                    continue

                if os.path.exists(st.gainfile):  # corrupt by -gain
                    model = calibration.apply_telcal(st, model, sign=-1)
                else:
                    logger.warn("No gainfile {0} found. Not applying inverse "
                                "gain.".format(st.gainfile))

                util.phase_shift(data_read, uvw, l, m)
                data_read += model
                util.phase_shift(data_read, uvw, -l, -m)

#    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
#    logger.debug('Selecting pols {0}'.format(st.pols))
#    return data.take(st.chans, axis=2).take(takepol, axis=3)

    return data_read.take(st.chans, axis=2)


def read_vys_segment(st, seg, cfile=None, timeout=default_timeout, offset=4):
    """ Read segment seg defined by state st from vys stream.
    Uses vysmaw application timefilter to receive multicast messages and pull
    spectra on the CBE.
    timeout is a multiple of read time in seconds to wait.
    offset is extra time in seconds to keep vys reader open.
    """

    # TODO: support for time downsampling

    try:
        import vysmaw_reader
    except ImportError:
        logger.error('ImportError for vysmaw_reader. Cannot '
                     'consume vys data.')

    t0 = time.Time(st.segmenttimes[seg][0], format='mjd', precision=9).unix
    t1 = time.Time(st.segmenttimes[seg][1], format='mjd', precision=9).unix

    data = np.empty((st.readints, st.nbl,
                     st.metadata.nchan_orig, st.metadata.npol_orig),
                    dtype='complex64', order='C')

    logger.info('Reading {0} s ints into shape {1} from {2} - {3} unix seconds'
                .format(st.metadata.inttime, st.datashape, t0, t1))

    polmap_standard = ['A*A', 'A*B', 'B*A', 'B*B']
    bbmap_standard = ['AC1', 'AC2', 'AC', 'BD1', 'BD2', 'BD']

    # TODO: vysmaw currently pulls all data, but allocates buffer based on these.
    # buffer will be too small if taking subset of all data.
    pollist = np.array([polmap_standard.index(pol) for pol in st.metadata.pols_orig], dtype=np.int32)  # TODO: use st.pols when vysmaw filter can too
    antlist = np.array([int(ant.lstrip('ea')) for ant in st.ants], dtype=np.int32)
    spwlist = list(zip(*st.metadata.spworder)[0])  # list of strings ["bb-spw"] in increasing freq order
    bbsplist = np.array([(int(bbmap_standard.index(spw.split('-')[0])), int(spw.split('-')[1])) for spw in spwlist], dtype=np.int32)

    with vysmaw_reader.Reader(t0, t1, antlist, pollist, bbsplist,
                              inttime_micros=long(st.metadata.inttime*1000000),
                              nchan=st.metadata.spw_nchan[0],
                              cfile=cfile, timeout=timeout, offset=offset) as reader:
        data[:] = reader.readwindow()

    # TODO: move pol selection up and into vysmaw filter function
    return data.take(st.chans, axis=2).take([polmap_standard.index(pol) for pol in st.pols], axis=3)


def read_bdf_segment(st, segment):
    """ Uses sdmpy to reads bdf (sdm) format data into numpy array in given
    segment. Each segment has st.readints integrations.
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
        data[i-nskip] = read.take(sortind,
                                  axis=1).reshape(st.metadata.nbl_orig,
                                                  st.metadata.nchan_orig,
                                                  st.metadata.npol_orig)

#    data[:] = scan.bdf.get_data(trange=[nskip, nskip+st.readints]).reshape(data.shape)

    return data


def simulate_segment(st, loc=0., scale=1.):
    """ Simulates visibilities for a segment.
    """

    shape = (st.readints, st.nbl, st.metadata.nchan_orig, st.npol)
    logger.info('Simulating data with shape {0}'.format(shape))
    data_read = np.empty(shape, dtype='complex64', order='C')
    data_read.real = np.random.normal(loc=loc, scale=scale, size=shape)
    data_read.imag = np.random.normal(loc=loc, scale=scale, size=shape)

    return data_read


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

    model = np.zeros((st.metadata.nchan_orig, st.readints), dtype='complex64')
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
