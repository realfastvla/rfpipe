from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import numpy as np
from numba import jit
from rfpipe import util

import logging
logger = logging.getLogger(__name__)


def flag_data(st, data):
    """ Identifies bad data and flags it to 0.
    Converts to masked array for flagging, but returns zeroed numpy array.
    """

    datam = np.ma.masked_values(data, 0j, copy=False, shrink=False)

    spwchans = st.spw_chan_select
    for flagparams in st.prefs.flaglist:
        if len(flagparams) == 3:
            mode, arg0, arg1 = flagparams
        else:
            mode, arg0 = flagparams

        if mode == 'blstd':
            flag_blstd(datam, arg0, arg1)
        elif mode == 'badchtslide':
            flag_badchtslide(datam, spwchans, arg0, arg1)
        elif mode == 'badspw':
            flag_badspw(datam, spwchans, arg0)
        else:
            logger.warning("Flaging mode {0} not available.".format(mode))

    return datam.filled(0)


def flag_blstd(data, sigma, convergence):
    """ Use data (4d) to calculate (int, chan, pol) to be flagged.
    Masked arrays assumed as input.
    """

    sh = data.shape

    blstd = util.blstd(data.data, data.mask)  # uses mask
    blstd = np.ma.masked_equal(blstd, 0)
#    blstd = np.ma.std(data, axis=1)

    # iterate to good median and std values
    blstdmednew = np.ma.median(blstd)
    blstdstdnew = np.ma.std(blstd)
    blstdstd = blstdstdnew*2  # TODO: is this initialization used?
    while (blstdstd-blstdstdnew)/blstdstd > convergence:
        blstdstd = blstdstdnew
        blstdmed = blstdmednew
        blstd = np.ma.masked_where(blstd > blstdmed + sigma*blstdstd, blstd, copy=False)
        blstdmednew = np.ma.median(blstd)
        blstdstdnew = np.ma.std(blstd)

    # flag blstd too high
    badt, badch, badpol = np.where(blstd > blstdmednew + sigma*blstdstdnew)
    logger.info("flagged by blstd: {0} of {1} total channel/time/pol cells."
                .format(len(badt), sh[0]*sh[2]*sh[3]))

    for i in range(len(badt)):
        data.mask[badt[i], :, badch[i], badpol[i]] = True


def flag_badchtslide(data, spwchans, sigma, win):
    """ Use data (4d) to calculate (int, chan, pol) to be flagged
    """

    sh = data.shape

    meanamp = np.abs(data).mean(axis=1)

    # calc badch as deviation from median of window
    spec = meanamp.mean(axis=0)
#    specmed = slidedev(spec, win)
    specmed = np.concatenate([spec[chans] - np.ma.median(spec[chans]) for chans in spwchans])
    badch = np.where(specmed > sigma*np.ma.std(specmed, axis=0))

    # calc badt as deviation from median of window
    lc = meanamp.mean(axis=1)
    lcmed = slidedev(lc, win)
    badt = np.where(lcmed > sigma*np.ma.std(lcmed, axis=0))

    badtcnt = len(np.ma.unique(badt))
    badchcnt = len(np.ma.unique(badch))
    logger.info("flagged by badchtslide: {0}/{1} pol-times and {2}/{3} pol-chans."
                .format(badtcnt, sh[0]*sh[3], badchcnt, sh[2]*sh[3]))

    for i in range(len(badch[0])):
        data.mask[:, :, badch[0][i], badch[1][i]] = True

    for i in range(len(badt[0])):
        data.mask[badt[0][i], :, :, badt[1][i]] = True


def flag_badspw(data, spwchans, sigma):
    """ Use data median variance between spw to flag spw
    Best to use this after flagging bad channels.
    """

    nspw = len(spwchans)

    if nspw >= 4:
        # calc badspw
        spec = np.abs(data).mean(axis=3).mean(axis=1).mean(axis=0)
        deviations = []
        for chans in spwchans:
            if spec[chans].count() > 3:
                deviations.append(np.ma.std(spec[chans]))
            else:
                deviations.append(0)
        deviations = np.ma.masked_equal(np.nan_to_num(deviations), 0)
        logger.info("badspw flagging finds deviations per spw: {0}"
                    .format(deviations))

        badspw = []
        badspwnew = np.where(deviations > sigma*np.ma.median(deviations))[0]
        while len(badspwnew) > len(badspw):
            badspw = badspwnew
            goodspw = [spw for spw in range(nspw) if spw not in badspw]
            badspwnew = np.where(deviations > sigma*np.ma.median(deviations.take(goodspw)))[0]

        badspw = np.concatenate((badspw, np.where(deviations.mask)[0])).astype(int)

        logger.info("flagged {0}/{1} spw ({2})"
                    .format(len(badspw), nspw, badspw))

        for i in badspw:
            data.mask[:, :, spwchans[i], :] = True

    else:
        logger.warning("Fewer than 4 spw. Not performing badspw detetion.")


def slidedev(arr, win):
    """ Given a (len x 2) array, calculate the deviation from the median per pol.
    Calculates median over a window, win.
    """

    med = np.zeros_like(arr)
    for i in range(len(arr)):
        inds = list(range(max(0, i-win//2), i)) + list(range(i+1, min(i+win//2, len(arr))))
        med[i] = np.ma.median(arr.take(inds, axis=0), axis=0)

    return arr-med


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
