from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import os.path
import numpy as np
from astropy import time
from rfpipe import fileLock
import pickle

import logging
logger = logging.getLogger(__name__)

try:
    import vysmaw_reader
except ImportError:
    pass


def data_prep(st, segment, data, flagversion="latest", returnsoltime=False):
    """ Applies calibration, flags, and subtracts time mean for data.
    flagversion can be "latest" or "rtpipe".
    Optionally prepares data with antenna flags, fixing out of order data,
    calibration, downsampling, OTF rephasing...
    """

    from rfpipe import calibration, flagging, util

    if not np.any(data):
        return data

    # select pols
    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
    logger.debug('Selecting pols {0} and chans {1}'.format(st.pols, st.chans))

    # TODO: check on reusing 'data' to save memory
#    datap = np.nan_to_num(np.require(data, requirements='W').take(takepol, axis=3).take(st.chans, axis=2))
    datap = np.require(data, requirements='W').take(takepol, axis=3).take(st.chans, axis=2)
    datap = prep_standard(st, segment, datap)

    if not np.any(datap):
        logger.info("All data zeros after prep_standard")
        return datap

    if st.gainfile is not None:
        logger.info("Applying calibration with {0}".format(st.gainfile))
        ret = calibration.apply_telcal(st, datap, savesols=st.prefs.savesols,
                                       returnsoltime=returnsoltime)
        if returnsoltime:
            datap, soltime = ret
        else:
            datap = ret

        if not np.any(datap):
            logger.info("All data zeros after apply_telcal")
            return datap
    else:
        logger.info("No gainfile found, so not applying calibration.")

    # support backwards compatibility for reproducible flagging
    logger.info("Flagging with version: {0}".format(flagversion))
    if flagversion == "latest":
        datap = flagging.flag_data(st, datap)
    elif flagversion == "rtpipe":
        datap = flagging.flag_data_rtpipe(st, datap)

    zerofrac = 1-np.count_nonzero(datap)/datap.size
    if zerofrac > st.prefs.max_zerofrac:
        logger.warning('Flagged {0:.1f}% of data. Zeroing all if greater than {1:.1f}%.'
                       .format(zerofrac*100, st.prefs.max_zerofrac*100))
        return np.array([])

    if st.prefs.timesub == 'mean':
        logger.info('Subtracting mean visibility in time.')
        datap = util.meantsub(datap)
    else:
        logger.info('No visibility subtraction done.')

    if (st.prefs.apply_chweights or st.prefs.apply_blweights) and st.readints > 3:
        if st.prefs.apply_chweights:
            # TODO: find better estimator. Currently loses sensitivity to FRB 121102 bursts.
            chvar = np.std(np.abs(datap).mean(axis=1), axis=0)
            chvar_norm = np.mean(1/chvar**2, axis=0)

        if st.prefs.apply_blweights:
            blvar = np.std(np.abs(datap).mean(axis=2), axis=0)
            blvar_norm = np.mean(1/blvar**2, axis=0)

        if st.prefs.apply_chweights:
            logger.info('Reweighting data by channel variances')
            datap = (datap/chvar[None, None, :, :])/chvar_norm[None, None, None, :]

        if st.prefs.apply_blweights:
            logger.info('Reweighting data by baseline variances')
            datap = (datap/blvar[None, :, None, :])/blvar_norm[None, None, None, :]

    if st.prefs.savenoise:
        save_noise(st, segment, datap)

    if returnsoltime:
        return datap, soltime
    else:
        return datap


def read_segment(st, segment, cfile=None, timeout=2):
    """ Read a segment of data.
    cfile and timeout are specific to vys data.
    cfile used as proxy for real-time environment when simulating data.
    Returns data as defined in metadata (no downselection yet)
    default timeout is multiple of read time in seconds to wait.
    """

    # assumed read shape (st.readints, st.nbl, st.metadata.nchan_orig, st.npol)
    logger.info("Reading segment {0} of scanId {1}"
                .format(segment, st.metadata.scanId))
    if st.metadata.datasource == 'sdm':
        data_read = read_bdf_segment(st, segment)
    elif st.metadata.datasource == 'vys':
        data_read = read_vys_segment(st, segment, cfile=cfile, timeout=timeout)
    elif st.metadata.datasource == 'sim':
        simseg = segment if cfile else None
        data_read = simulate_segment(st, segment=simseg)
    elif st.metadata.datasource == 'vyssim':
        data_read = read_vys_segment(st, segment, cfile=cfile, timeout=timeout,
                                     returnsim=True)
    else:
        logger.error('Datasource {0} not recognized.'
                     .format(st.metadata.datasource))

    # report bad values
    if np.any(np.isnan(data_read)):
        logger.warning("Read data has some NaNs")
    if np.any(np.isinf(data_read)):
        logger.warning("Read data has some Infs")
    if np.any(np.abs(data_read) > 1e20):
        logger.warning("Read data has values larger than 1e20")

    if not np.any(data_read):
        logger.info('Read data are all zeros for segment {0}.'.format(segment))
        return np.array([])
    else:
        logger.info('Read data with zero fraction of {0:.3f} for segment {1}'
                    .format(1-np.count_nonzero(data_read)/data_read.size,
                            segment))
        return data_read


def prep_standard(st, segment, data):
    """ Common first data prep stages, incl
    online flags, resampling, and mock transients.
    """

    from rfpipe import calibration, flagging, util

    if not np.any(data):
        return data

    # read and apply flags for given ant/time range. 0=bad, 1=good
    if st.prefs.applyonlineflags and st.metadata.datasource in ['vys', 'sdm']:
        flags = flagging.getonlineflags(st, segment)
        data = np.where(flags[None, :, None, None], data, 0j)
    else:
        logger.info('Not applying online flags.')

    if not np.any(data):
        return data

    if st.prefs.simulated_transient is not None or st.otfcorrections is not None:
        uvw = util.get_uvw_segment(st, segment)

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
            st.prefs.simulated_transient = util.make_transient_params(st, segment=segment,
                                                                      ntr=st.prefs.simulated_transient,
                                                                      data=data)

        assert isinstance(st.prefs.simulated_transient, list), "Simulated transient must be list of tuples."

        for params in st.prefs.simulated_transient:
            assert len(params) == 7 or len(params) == 8, ("Transient requires 7 or 8 parameters: "
                                                          "(segment, i0/int, dm/pc/cm3, dt/s, "
                                                          "amp/sys, dl/rad, dm/rad) and optionally "
                                                          "ampslope/sys")
            if segment == params[0]:
                if len(params) == 7:
                    (mock_segment, i0, dm, dt, amp, l, m) = params
                    ampslope = 0

                    logger.info("Adding transient to segment {0} at int {1}, "
                                "DM {2}, dt {3} with amp {4} and l,m={5},{6}"
                                .format(mock_segment, i0, dm, dt, amp, l, m))
                elif len(params) == 8:
                    (mock_segment, i0, dm, dt, amp, l, m, ampslope) = params
                    logger.info("Adding transient to segment {0} at int {1}, "
                                " DM {2}, dt {3} with amp {4}-{5} and "
                                "l,m={6},{7}"
                                .format(mock_segment, i0, dm, dt, amp,
                                        amp+ampslope, l, m))
                try:
                    model = np.require(np.broadcast_to(util.make_transient_data(st, amp, i0, dm, dt, ampslope=ampslope)
                                                       .transpose()[:, None, :, None],
                                                       st.datashape),
                                       requirements='W')
                except IndexError:
                    logger.warning("IndexError while adding transient. Skipping...")
                    continue

                if st.gainfile is not None:
                    model = calibration.apply_telcal(st, model, sign=-1)
                util.phase_shift(model, uvw, -l, -m)
                data += model

    if st.otfcorrections is not None:
        # shift phasecenters to first phasecenter in segment
        if len(st.otfcorrections[segment]) > 1:
            ints, ra0, dec0 = st.otfcorrections[segment][0]  # new phase center for segment
            logger.info("Correcting {0} phasecenters to first at RA,Dec = {1},{2}"
                        .format(len(st.otfcorrections[segment])-1, ra0, dec0))
            for ints, ra_deg, dec_deg in st.otfcorrections[segment][1:]:
                l0 = np.radians((ra_deg-ra0)*np.cos(np.radians(dec0)))
                m0 = np.radians(dec_deg-dec0)
                util.phase_shift(data, uvw, l0, m0, ints=ints)

    return data


def read_vys_segment(st, seg, cfile=None, timeout=2, offset=4, returnsim=False):
    """ Read segment seg defined by state st from vys stream.
    Uses vysmaw application timefilter to receive multicast messages and pull
    spectra on the CBE.
    timeout is a multiple of read time in seconds to wait.
    offset is extra time in seconds to keep vys reader open.
    returnsim can optionally swap in sim data if vys client returns anything.
    """

    # TODO: support for time downsampling

    t0 = time.Time(st.segmenttimes[seg][0], format='mjd', precision=9).unix
    t1 = time.Time(st.segmenttimes[seg][1], format='mjd', precision=9).unix

    logger.info('Reading scanId {0}: {1} s ints with shape {2} from {3} - {4} unix seconds'
                .format(st.metadata.scanId, st.metadata.inttime, st.datashape_orig, t0, t1))

    # TODO: vysmaw currently pulls all data, but allocates buffer based on these.
    # buffer will be too small if taking subset of all data.
    antlist = np.array([int(ant.lstrip('ea'))
                        for ant in st.ants], dtype=np.int32)
    bbmap_standard = ['AC1', 'AC2', 'AC', 'BD1', 'BD2', 'BD']
    spwlist = list(zip(*st.metadata.spworder))[0]  # list of strings ["bb-spw"] in increasing freq order
    bbsplist = np.array([(int(bbmap_standard.index(spw.split('-')[0])),
                          int(spw.split('-')[1])) for spw in spwlist],
                        dtype=np.int32)

    # TODO: move pol selection up and into vysmaw filter function
    assert st.prefs.selectpol in ['auto', 'all'], 'auto and all pol selection supported in vys'
    polauto = st.prefs.selectpol == 'auto'
    logger.info("Passing to vysmaw: antlist {0} \n\t bbsplist {1} \n"
                .format(antlist, bbsplist))
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

    if data is not None:
        return data
    else:
        if (reader is not None) and returnsim:
            return simulate_segment(st)
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
                        time.Time(st.segmenttimes[segment][0], format='mjd', precision=9).unix,
                        time.Time(st.segmenttimes[segment][1], format='mjd', precision=9).unix))
    data = read_bdf(st, nskip=nskip).astype('complex64')

    return data


def read_bdf(st, nskip=0):
    """ Uses sdmpy to read a given range of integrations from sdm of given scan.
    readints=0 will read all of bdf (skipping nskip).
    Returns data with spw in increasing frequency order.
    """

    from rfpipe import util

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

def calc_gridfrac(st, segment):
    """ Calculate the fraction of baseline*chan values that are gridded.
    **TODO: figure out if this can collide with rfgpu calls in the search module**
    """

    try:
        import rfgpu
    except ImportError:
        logger.warn("Cannot import rfgpu, so cannot calculate gridfrac")
        return 1

    uvw = util.get_uvw_segment(st, segment)
    u, v, w = uvw
    u_us = 1e6*u[:, 0]/(1e9*st.freq[0])
    v_us = 1e6*v[:, 0]/(1e9*st.freq[0])

    grid = rfgpu.Grid(st.nbl, st.nchan, st.readints, upix, vpix, 0)  # choose device 0
    grid.set_uv(u_us, v_us)  # u, v in us
    grid.set_freq(st.freq*1e3)  # freq in MHz
    grid.set_cell(st.uvres)  # uv cell size in wavelengths (== 1/FoV(radians))
    grid.compute()
    grid.conjugate(vis_raw)

    # calc fraction of data gridded (any grid will do)
    gridfrac = grid.get_nnz()/(st.nbl*st.nchan)

    return gridfrac


def save_noise(st, segment, data, chunk=500):
    """ Calculates noise properties and save values to pickle.
    chunk defines window for measurement. at least one measurement always made.
    """

    from rfpipe import util

    results = util.calc_noise(st, segment, data, chunk=chunk)

    try:
        noisefile = st.noisefile
        with fileLock.FileLock(noisefile+'.lock', timeout=60):
            with open(noisefile, 'ab+') as pkl:
                pickle.dump(results, pkl)
    except fileLock.FileLock.FileLockException:
        noisefile = ('{0}_seg{1}.pkl'
                     .format(st.noisefile.rstrip('.pkl'), segment))
        logger.warning('Noise file writing timeout. '
                       'Spilling to new file {0}.'.format(noisefile))
        with open(noisefile, 'ab+') as pkl:
            pickle.dump(results, pkl)

    if len(results):
        logger.info('Wrote {0} noise measurement{1} from segment {2} to {3}'
                    .format(len(results), 's'[:len(results)-1], segment, noisefile))


def simulate_segment(st, loc=0., scale=1., segment=None):
    """ Simulates visibilities for a segment.
    If segment (int) given, then read will behave like vysmaw client and skip if too late.
    """

    # mimic real-time environment by skipping simulation when late
    if segment is not None:
        currenttime = time.Time.now().mjd
        t1 = st.segmenttimes[segment][1]
        if currenttime > t1:
            logger.info('Current time {0} is later than window end {1}. '
                        'Skipping segment {2}.'
                        .format(currenttime, t1, segment))
            return np.array([])

    logger.info('Simulating data with shape {0}'.format(st.datashape_orig))

    data = np.empty(st.datashape_orig, dtype='complex64', order='C')
    for i in range(len(data)):
        data[i].real = np.random.normal(loc=loc, scale=scale,
                                        size=st.datashape_orig[1:]).astype(np.float32)
        data[i].imag = np.random.normal(loc=loc, scale=scale,
                                        size=st.datashape_orig[1:]).astype(np.float32)

    return data


def sdm_sources(sdmname):
    """ Use sdmpy to get all sources and ra,dec per scan as dict """

    from rfpipe import util

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
