from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import os.path
import numpy as np
from numba import jit
from astropy import time
import sdmpy
import pwkit.environments.casa.util as casautil
from rfpipe import util, calibration, fileLock
import pickle

import logging
logger = logging.getLogger(__name__)

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

    data_prep = prep_standard(st, segment, np.require(data.take(takepol,
                                                                axis=3),
                                                      requirements='W'))
    data_prep = calibration.apply_telcal(st, data_prep)

    # support backwards compatibility for reproducible flagging
    if flagversion == "latest":
        data_prep = flag_data(st, data_prep)
    elif flagversion == "rtpipe":
        data_prep = flag_data_rtpipe(st, data_prep)

    if st.prefs.timesub == 'mean':
        logger.info('Subtracting mean visibility in time.')
        data_prep = util.meantsub(data_prep, parallel=st.prefs.nthread > 1)
    else:
        logger.info('No visibility subtraction done.')

    if st.prefs.savenoise:
        save_noise(st, segment, data_prep.take(st.chans, axis=2))

    logger.debug('Selecting chans {0}'.format(st.chans))

    return data_prep.take(st.chans, axis=2)


def read_segment(st, segment, cfile=None, timeout=default_timeout):
    """ Read a segment of data.
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


def prep_standard(st, segment, data):
    """ Common first data prep stages, incl
    online flags, resampling, and mock transients.
    """

    if not np.any(data):
        return data

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
            data = np.where(flags[None, :, None, None] == 1,
                            np.require(data, requirements='W'), 0j)
        else:
            logger.info('No flagged antennas in time range {0}-{1} '
                        .format(t0, t1))
    else:
        logger.info('Not applying online flags.')

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
        assert isinstance(st.prefs.simulated_transient, list), "Simulated transient must be list of tuples."

        uvw = util.get_uvw_segment(st, segment)
        for params in st.prefs.simulated_transient:
            assert len(params) == 7, ("Transient requires 7 parameters: "
                                      "(segment, i0/int, dm/pc/cm3, dt/s, "
                                      "amp/sys, dl/rad, dm/rad)")
            (mock_segment, i0, dm, dt, amp, l, m) = params
            if segment == mock_segment:
                logger.info("Adding transient to segment {0} at int {1}, DM {2}, "
                            "dt {3} with amp {4} and l,m={5},{6}"
                            .format(mock_segment, i0, dm, dt, amp, l, m))
                try:
                    model = np.require(np.broadcast_to(generate_transient(st, amp, i0, dm, dt)
                                                       .transpose()[:, None, :, None],
                                                       data.shape),
                                       requirements='W')
                except IndexError:
                    logger.warn("IndexError while adding transient. Skipping...")
                    continue

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

    try:
        import vysmaw_reader
    except ImportError:
        logger.error('ImportError for vysmaw_reader. Cannot '
                     'consume vys data.')

    t0 = time.Time(st.segmenttimes[seg][0], format='mjd', precision=9).unix
    t1 = time.Time(st.segmenttimes[seg][1], format='mjd', precision=9).unix

#    data = np.empty((st.readints, st.nbl,
#                     st.metadata.nchan_orig, st.metadata.npol_orig),
#                    dtype='complex64', order='C')

    logger.info('Reading {0} s ints into shape {1} from {2} - {3} unix seconds'
                .format(st.metadata.inttime, st.datashape_orig, t0, t1))

    polmap_standard = ['A*A', 'A*B', 'B*A', 'B*B']
    bbmap_standard = ['AC1', 'AC2', 'AC', 'BD1', 'BD2', 'BD']

    # TODO: vysmaw currently pulls all data, but allocates buffer based on these.
    # buffer will be too small if taking subset of all data.
    pollist = np.array([polmap_standard.index(pol)
                        for pol in st.metadata.pols_orig], dtype=np.int32)  # TODO: use st.pols when vysmaw filter can too
    antlist = np.array([int(ant.lstrip('ea'))
                        for ant in st.ants], dtype=np.int32)
    spwlist = list(zip(*st.metadata.spworder))[0]  # list of strings ["bb-spw"] in increasing freq order
    bbsplist = np.array([(int(bbmap_standard.index(spw.split('-')[0])),
                          int(spw.split('-')[1])) for spw in spwlist],
                        dtype=np.int32)

    with vysmaw_reader.Reader(t0, t1, antlist, pollist, bbsplist,
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


def flag_data(st, data):
    """ Identifies bad data and flags it to 0.
    """

#    data = np.ma.masked_equal(data, 0j)  # TODO remove this and ignore zeros manually
    flags = np.ones_like(data, dtype=bool)

    for flagparams in st.prefs.flaglist:
        mode, arg0, arg1 = flagparams
        if mode == 'blstd':
            flags *= flag_blstd(data, arg0, arg1)[:, None, :, :]
        elif mode == 'badchtslide':
            flags *= flag_badchtslide(data, arg0, arg1)[:, None, :, :]
        else:
            logger.warn("Flaging mode {0} not available.".format(mode))

    return data*flags


def flag_blstd(data, sigma, convergence):
    """ Use data (4d) to calculate (int, chan, pol) to be flagged.
    """

    sh = data.shape
    flags = np.ones((sh[0], sh[2], sh[3]), dtype=bool)

    blstd = data.std(axis=1)

    # iterate to good median and std values
    blstdmednew = np.ma.median(blstd)
    blstdstdnew = blstd.std()
    blstdstd = blstdstdnew*2  # TODO: is this initialization used?
    while (blstdstd-blstdstdnew)/blstdstd > convergence:
        blstdstd = blstdstdnew
        blstdmed = blstdmednew
        blstd = np.ma.masked_where(blstd > blstdmed + sigma*blstdstd, blstd, copy=False)
        blstdmednew = np.ma.median(blstd)
        blstdstdnew = blstd.std()

    # flag blstd too high
    badt, badch, badpol = np.where(blstd > blstdmednew + sigma*blstdstdnew)
    logger.info("flag by blstd: {0} of {1} total channel/time/pol cells flagged."
                .format(len(badt), sh[0]*sh[2]*sh[3]))

    for i in range(len(badt)):
        flags[badt[i], badch[i], badpol[i]] = False

    return flags


def flag_badchtslide(data, sigma, win):
    """ Use data (4d) to calculate (int, chan, pol) to be flagged
    """

    sh = data.shape
    flags = np.ones((sh[0], sh[2], sh[3]), dtype=bool)

    meanamp = np.abs(data).mean(axis=1)
    spec = meanamp.mean(axis=0)
    lc = meanamp.mean(axis=1)

    # calc badch as deviation from median of window
    specmed = slidedev(spec, win)
    badch = np.where(specmed > sigma*specmed.std(axis=0))

    # calc badt as deviation from median of window
    lcmed = slidedev(lc, win)
    badt = np.where(lcmed > sigma*lcmed.std(axis=0))

    badtcnt = len(np.unique(badt))
    badchcnt = len(np.unique(badch))
    logger.info("flag by badchtslide: {0}/{1} pol-times and {2}/{3} pol-chans flagged."
                .format(badtcnt, sh[0]*sh[3], badchcnt, sh[2]*sh[3]))

    for i in range(len(badch[0])):
        flags[:, badch[0][i], badch[1][i]] = False

    for i in range(len(badt[0])):
        flags[badt[0][i], :, badt[1][i]] = False

    return flags


@jit
def slidedev(arr, win):
    """ Given a (len x 2) array, calculate the deviation from the median per pol.
    Calculates median over a window, win.
    """

    med = np.zeros((len(arr), 2))
    for i in range(len(arr)):
        inds = list(range(max(0, i-win//2), i)) + list(range(i+1, min(i+win//2, len(arr))))
        for j in inds:
            med[j] = np.ma.median(arr.take(inds, axis=0), axis=0)

    return arr-med


def flag_data_rtpipe(st, data):
    """ Flagging data in single process
    Deprecated.
    """
    try:
        import rtlib_cython as rtlib
    except ImportError:
        logger.error("rtpipe not installed. Cannot import rtlib for flagging.")

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
                spwpol[(spw, pol)] = np.abs(data[:, :, chans, pol]).std()
        
        meanstd = np.mean(list(spwpol.values()))
        for (spw,pol) in spwpol:
            if spwpol[(spw, pol)] > st.prefs.badspwpol*meanstd:
                logger.info('Flagging all of (spw %d, pol %d) for excess noise.' % (spw, pol))
                chans = np.arange(st.metadata.spw_nchan[spw]*spw, st.metadata.spw_nchan[spw]*(1+spw))
                data[:, :, chans, pol] = 0j

    return data


def simulate_segment(st, loc=0., scale=1.):
    """ Simulates visibilities for a segment.
    """

    shape = (st.readints, st.nbl, st.metadata.nchan_orig, st.metadata.npol_orig)
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

    i = i0 + util.calc_delay2(st.freq, st.freq.max(), dm)/st.inttime
#    print(i)
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
