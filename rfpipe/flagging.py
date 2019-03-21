from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import numpy as np
from numba import jit
from rfpipe import util

import logging
logger = logging.getLogger(__name__)


def getonlineflags(st, segment):
    """ Gets antenna flags for a given segment from either sdm or mcaf server.
    Returns an array of flags (1: good, 0: bad) for each baseline.
    """

    t0, t1 = st.segmenttimes[segment]
    if st.metadata.datasource == 'sdm':
        sdm = util.getsdm(st.metadata.filename, bdfdir=st.metadata.bdfdir)
        scan = sdm.scan(st.metadata.scan)
        flags = scan.flags([t0, t1]).all(axis=0)
    elif st.metadata.datasource == 'vys':
        try:
            from realfast.mcaf_servers import getblflags
            flags = getblflags(st.metadata.datasetId, st.blarr,
                               startTime=t0, endTime=t1)
        except (ImportError, Exception):
            logger.warning("No mcaf antenna flag server flags available")
            flags = np.ones(st.nbl)

    if not flags.all():
        logger.info('Found antennas to flag in time range {0}-{1} '
                    .format(t0, t1))
    else:
        logger.info('No flagged antennas in time range {0}-{1} '
                    .format(t0, t1))

    return flags


def flag_data(st, data):
    """ Identifies bad data and flags it to 0.
    Should pass in masked array.
    """

    data = np.ma.masked_equal(data, 0j)  # TODO remove this and ignore zeros manually
    flags = np.ones_like(data, dtype=bool)

    spwchans = st.spw_chan_select
    for flagparams in st.prefs.flaglist:
        mode, arg0, arg1 = flagparams
        if mode == 'blstd':
            data *= flag_blstd(data, arg0, arg1)[:, None, :, :]
        elif mode == 'badchtslide':
            data *= flag_badchtslide(data, spwchans, arg0, arg1)[:, None, :, :]
        elif mode == 'badspw':
            data *= flag_badspw(data, spwchans, arg0, arg1)[None, None, :, None]
        else:
            logger.warning("Flaging mode {0} not available.".format(mode))

    return data


def flag_blstd(data, sigma, convergence):
    """ Use data (4d) to calculate (int, chan, pol) to be flagged.
    """

    sh = data.shape
    flags = np.ones((sh[0], sh[2], sh[3]), dtype=bool)

    blstd = np.nanstd(data, axis=1)

    # iterate to good median and std values
    blstdmednew = np.ma.median(blstd)
    blstdstdnew = np.nanstd(blstd)
    blstdstd = blstdstdnew*2  # TODO: is this initialization used?
    while (blstdstd-blstdstdnew)/blstdstd > convergence:
        blstdstd = blstdstdnew
        blstdmed = blstdmednew
        blstd = np.ma.masked_where(blstd > blstdmed + sigma*blstdstd, blstd, copy=False)
        blstdmednew = np.ma.median(blstd)
        blstdstdnew = np.nanstd(blstd)

    # flag blstd too high
    badt, badch, badpol = np.where(blstd > blstdmednew + sigma*blstdstdnew)
    logger.info("flagged by blstd: {0} of {1} total channel/time/pol cells."
                .format(len(badt), sh[0]*sh[2]*sh[3]))

    for i in range(len(badt)):
        flags[badt[i], badch[i], badpol[i]] = False

    return flags


def flag_badchtslide(data, spwchans, sigma, win):
    """ Use data (4d) to calculate (int, chan, pol) to be flagged
    """

    sh = data.shape
    flags = np.ones((sh[0], sh[2], sh[3]), dtype=bool)

    meanamp = np.abs(data).mean(axis=1)

    # calc badch as deviation from median of window
    spec = meanamp.mean(axis=0)
#    specmed = slidedev(spec, win)
    specmed = np.concatenate([spec[chans] - np.median(spec[chans]) for chans in spwchans])
    badch = np.where(specmed > sigma*np.nanstd(specmed, axis=0))

    # calc badt as deviation from median of window
    lc = meanamp.mean(axis=1)
    lcmed = slidedev(lc, win)
    badt = np.where(lcmed > sigma*np.nanstd(lcmed, axis=0))

    badtcnt = len(np.unique(badt))
    badchcnt = len(np.unique(badch))
    logger.info("flagged by badchtslide: {0}/{1} pol-times and {2}/{3} pol-chans."
                .format(badtcnt, sh[0]*sh[3], badchcnt, sh[2]*sh[3]))

    for i in range(len(badch[0])):
        flags[:, badch[0][i], badch[1][i]] = False

    for i in range(len(badt[0])):
        flags[badt[0][i], :, badt[1][i]] = False

    return flags


def flag_badspw(data, spwchans, sigma, win):
    """ Use data median variance between spw to flag spw
    """

    sh = data.shape
    nspw = len(spwchans)
    flags = np.ones((sh[2],), dtype=bool)
    if nspw >= 4:
        # calc badspw
        spec = np.abs(data).mean(axis=3).mean(axis=1).mean(axis=0)
#        specmed = slidedev(spec, win)
        variances = []
        for chans in spwchans:
            if len(chans) > 3:
                variances.append(np.var(spec[chans]-np.median(spec[chans])))
            else:
                variances.append(np.nan)
        variances = np.nan_to_num(variances)
        logger.debug("Variance per spw: {0}".format(variances))

        if np.median(variances):
            badspw = np.where(variances > sigma*np.median(variances))[0]
            logger.info("flagged {0}/{1} spw ({2})"
                        .format(len(badspw), nspw, badspw))
            if len(badspw) > nspw//2:
                logger.warning("more than half of spw flagged. flagging all spw.")
                badspw = list(range(nspw))

            for i in badspw:
                flags[spwchans[i]] = False
        else:
            logger.warning("flagged no badspw (no variance found)")

    else:
        logger.warning("Fewer than 4 spw. Not performing badspw detetion.")

    return flags


@jit(cache=True)
def slidedev(arr, win):
    """ Given a (len x 2) array, calculate the deviation from the median per pol.
    Calculates median over a window, win.
    """

    med = np.zeros_like(arr)
    for i in range(len(arr)):
        inds = list(range(max(0, i-win//2), i)) + list(range(i+1, min(i+win//2, len(arr))))
        med[i] = np.ma.median(arr.take(inds, axis=0), axis=0)

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
        for (spw, pol) in spwpol:
            if spwpol[(spw, pol)] > st.prefs.badspwpol*meanstd:
                logger.info('Flagging all of (spw %d, pol %d) for excess noise.' % (spw, pol))
                chans = np.arange(st.metadata.spw_nchan[spw]*spw, st.metadata.spw_nchan[spw]*(1+spw))
                data[:, :, chans, pol] = 0j

    return data
