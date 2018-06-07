from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import os.path
import numpy as np
from numba import jit
from astropy import time
import pwkit.environments.casa.util as casautil
from rfpipe import util, calibration, fileLock, flagging
import pickle

import logging
logger = logging.getLogger(__name__)

try:
    import vysmaw_reader
except ImportError:
    pass

qa = casautil.tools.quanta()
default_timeout = 10  # multiple of read time in seconds to wait


def data_prep(st, segment, data, flagversion="latest"):
    """ Applies calibration, flags, and subtracts time mean for data.
    flagversion can be "latest" or "rtpipe".
    Optionally prepares data with antenna flags, fixing out of order data,
    calibration, downsampling, etc..
    """

    if not np.any(data):
        return data

    # take pols of interest
    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
    logger.debug('Selecting pols {0}'.format(st.pols))

    # TODO: check on reusing 'data' to save memory
    datap = np.nan_to_num(data.take(takepol, axis=3), copy=True)
    datap = prep_standard(st, segment, datap)
    if not np.any(datap):
        logger.info("All data zeros after prep_standard")
        return datap

    if st.gainfile is not None:
        datap = calibration.apply_telcal(st, datap)
        if not np.any(datap):
            logger.info("All data zeros after apply_telcal")
            return datap
    else:
        logger.info("No gainfile found, so not applying calibration.")

    # support backwards compatibility for reproducible flagging
    if flagversion == "latest":
        datap = flagging.flag_data(st, datap)
    elif flagversion == "rtpipe":
        datap = flagging.flag_data_rtpipe(st, datap)

    if st.prefs.timesub == 'mean':
        logger.info('Subtracting mean visibility in time.')
        datap = util.meantsub(datap, parallel=st.prefs.nthread > 1)
    else:
        logger.info('No visibility subtraction done.')

    if st.prefs.savenoise:
        save_noise(st, segment, datap.take(st.chans, axis=2))

    logger.debug('Selecting chans {0}'.format(st.chans))

    return datap.take(st.chans, axis=2)


def read_segment(st, segment, cfile=None, timeout=default_timeout):
    """ Read a segment of data.
    cfile and timeout are specific to vys data.
    Returns data as defined in metadata (no downselection yet)
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


def prep_standard(st, segment, data):
    """ Common first data prep stages, incl
    online flags, resampling, and mock transients.
    """

    if not np.any(data):
        return data

    # read and apply flags for given ant/time range. 0=bad, 1=good
    if st.prefs.applyonlineflags and st.metadata.datasource in ['vys', 'sdm']:
        flags = flagging.getonlineflags(st, segment)
        data = np.where(flags[None, :, None, None],
                        np.require(data, requirements='W'), 0j)
    else:
        logger.info('Not applying online flags.')

    if not np.any(data):
        return data

    # optionally integrate (downsample)
    if ((st.prefs.read_tdownsample > 1) or (st.prefs.read_fdownsample > 1)):
        data2 = np.zeros(st.datashape, dtype='complex64')
        if st.prefs.read_tdownsample > 1:
            logger.info('Downsampling in time by {0}'
                        .format(st.prefs.read_tdownsample))
            for i in range(st.datashape[0]):
                data2[i] = data[
                    i*st.prefs.read_tdownsample:(i+1)*st.prefs.read_tdownsample].mean(axis=0)
        if st.prefs.read_fdownsample > 1:
            logger.info('Downsampling in frequency by {0}'
                        .format(st.prefs.read_fdownsample))
            for i in range(st.datashape[2]):
                data2[:, :, i, :] = data[:, :, i*st.prefs.read_fdownsample:(i+1)*st.prefs.read_fdownsample].mean(axis=2)
        data = data2

    # optionally add transients
    if st.prefs.simulated_transient is not None:
        # for an int type, overload prefs.simulated_transient random mocks
        if isinstance(st.prefs.simulated_transient, int):
            logger.info("Filling simulated_transient with {0} random transients"
                        .format(st.prefs.simulated_transient))
            st.prefs.simulated_transient = util.make_transient_params(st,
                                                                      ntr=st.prefs.simulated_transient,
                                                                      data=data)

        assert isinstance(st.prefs.simulated_transient, list), "Simulated transient must be list of tuples."

        uvw = util.get_uvw_segment(st, segment)
        for params in st.prefs.simulated_transient:
            assert len(params) == 7 or len(params) == 8, ("Transient requires 7 or 8 parameters: "
                                                          "(segment, i0/int, dm/pc/cm3, dt/s, "
                                                          "amp/sys, dl/rad, dm/rad) and optionally "
                                                          "ampslope/sys")
            if segment == params[0]:
                if len(params) == 7:
                    (mock_segment, i0, dm, dt, amp, l, m) = params
                    ampslope = 0

                    logger.info("Adding transient to segment {0} at int {1}, DM {2}, "
                                "dt {3} with amp {4} and l,m={5},{6}"
                                .format(mock_segment, i0, dm, dt, amp, l, m))
                elif len(params) == 8:
                    (mock_segment, i0, dm, dt, amp, l, m, ampslope) = params
                    logger.info("Adding transient to segment {0} at int {1}, DM {2}, "
                                "dt {3} with amp {4}-{5} and l,m={6},{7}"
                                .format(mock_segment, i0, dm, dt, amp,
                                        amp+ampslope, l, m))
                try:
                    model = np.require(np.broadcast_to(util.make_transient_data(st, amp, i0, dm, dt, ampslope=ampslope)
                                                       .transpose()[:, None, :, None],
                                                       data.shape),
                                       requirements='W')
                except IndexError:
                    logger.warn("IndexError while adding transient. Skipping...")
                    continue

                if st.gainfile is not None:
                    model = calibration.apply_telcal(st, model, sign=-1)
                util.phase_shift(model, uvw, -l, -m)
                data += model

    return data


def read_vys_segment(st, seg, cfile=None, timeout=default_timeout, offset=4):
    """ Read segment seg defined by state st from vys stream.
    Uses vysmaw application timefilter to receive multicast messages and pull
    spectra on the CBE.
    timeout is a multiple of read time in seconds to wait.
    offset is extra time in seconds to keep vys reader open.
    """

    # TODO: support for time downsampling

    t0 = time.Time(st.segmenttimes[seg][0], format='mjd', precision=9).unix
    t1 = time.Time(st.segmenttimes[seg][1], format='mjd', precision=9).unix

    logger.info('Reading {0} s ints into shape {1} from {2} - {3} unix seconds'
                .format(st.metadata.inttime, st.datashape_orig, t0, t1))

    # TODO: vysmaw currently pulls all data, but allocates buffer based on these.
    # buffer will be too small if taking subset of all data.
    antlist = np.array([int(ant.lstrip('ea'))
                        for ant in st.ants], dtype=np.int32)
    bbmap_standard = ['AC1', 'AC2', 'AC', 'BD1', 'BD2', 'BD']
    spwlist = list(zip(*st.metadata.spworder))[0]  # list of strings ["bb-spw"] in increasing freq order
    bbsplist = np.array([(int(bbmap_standard.index(spw.split('-')[0])),
                          int(spw.split('-')[1])) for spw in spwlist],
                        dtype=np.int32)

    assert st.prefs.selectpol in ['auto', 'all'], 'auto and all pol selection supported in vys'
    polauto = st.prefs.selectpol == 'auto'
    with vysmaw_reader.Reader(t0, t1, antlist, bbsplist, polauto,
                              inttime_micros=st.metadata.inttime*1000000.,
                              nchan=st.metadata.spw_nchan[0],
                              cfile=cfile,
                              timeout=timeout,
                              offset=offset) as reader:
        if reader is not None:
            data = reader.readwindow()
        else:
            data = None

    # TODO: move pol selection up and into vysmaw filter function
    if data is not None:
        return data
    else:
        return np.array([])


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
    sdm = util.getsdm(st.metadata.filename, bdfdir=st.metadata.bdfdir)
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


def save_noise(st, segment, data, chunk=200):
    """ Calculates noise properties and save values to pickle.
    chunk defines window for measurement. at least one measurement always made.
    """

    from rfpipe.search import grid_image

    uvw = util.get_uvw_segment(st, segment)
    chunk = min(chunk, max(1, st.readints-1))  # ensure at least one measurement
    ranges = list(zip(list(range(0, st.readints-chunk, chunk)),
                      list(range(chunk, st.readints, chunk))))

    results = []
    for (r0, r1) in ranges:
        imid = (r0+r1)//2
        noiseperbl = estimate_noiseperbl(data[r0:r1])
        imstd = grid_image(data, uvw, st.npixx, st.npixy, st.uvres,
                           'fftw', 1, integrations=imid).std()
        zerofrac = float(len(np.where(data[r0:r1] == 0j)[0]))/data[r0:r1].size
        results.append((segment, imid, noiseperbl, zerofrac, imstd))

    try:
        noisefile = st.noisefile
        with fileLock.FileLock(noisefile+'.lock', timeout=10):
            with open(noisefile, 'ab+') as pkl:
                pickle.dump(results, pkl)
    except fileLock.FileLock.FileLockException:
        noisefile = ('{0}_seg{1}.pkl'
                     .format(st.noisefile.rstrip('.pkl'), segment))
        logger.warn('Noise file writing timeout. '
                    'Spilling to new file {0}.'.format(noisefile))
        with open(noisefile, 'ab+') as pkl:
            pickle.dump(results, pkl)

    if len(results):
        logger.info('Wrote {0} noise measurement{1} from segment {2} to {3}'
                    .format(len(results), 's'[:len(results)-1], segment, noisefile))


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


def simulate_segment(st, loc=0., scale=1.):
    """ Simulates visibilities for a segment.
    """

    logger.info('Simulating data with shape {0}'.format(st.datashape_orig))
    data = np.empty(st.datashape_orig, dtype='complex64', order='C')
    data.real = np.random.normal(loc=loc, scale=scale,
                                 size=st.datashape_orig).astype(np.float32)
    data.imag = np.random.normal(loc=loc, scale=scale,
                                 size=st.datashape_orig).astype(np.float32)

    return data


def sdm_sources(sdmname):
    """ Use sdmpy to get all sources and ra,dec per scan as dict """

    sdm = util.getsdm(sdmname)
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
