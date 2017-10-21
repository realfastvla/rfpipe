from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import pickle
import os.path
from collections import OrderedDict
import numpy as np
import pandas as pd
from rfpipe import preferences, state, util, search, source, metadata

import logging
logger = logging.getLogger(__name__)


def oldcands_read(candsfile, sdmscan=None):
    """ Read old-style candfile and create new-style DataFrame
    Returns a list of tuples (state, dataframe) per scan.
    """

    with open(candsfile, 'rb') as pkl:
        d = pickle.load(pkl)
        loc, prop = pickle.load(pkl)

    if not sdmscan:
        scanind = d['featureind'].index('scan')
        scans = np.unique(loc[:, scanind])
    else:
        scans = [sdmscan]

    ll = []
    for scan in scans:
        try:
            st, df = oldcands_readone(candsfile, scan)
            ll.append((st, df))
        except AttributeError:
            pass

    return ll


def oldcands_readone(candsfile, scan):
    """ For old-style merged candidate file, create new state and candidate
    dataframe for a given scan.
    Requires sdm locally with bdf for given scan.
    """

    with open(candsfile, 'rb') as pkl:
        d = pickle.load(pkl)
        loc, prop = pickle.load(pkl)

    inprefs = preferences.oldstate_preferences(d, scan=scan)
    inprefs.pop('gainfile')
    sdmfile = os.path.basename(d['filename'])
    if os.path.exists(sdmfile):
        logger.info('Parsing metadata from sdmfile {0}'.format(sdmfile))
        st = state.State(sdmfile=sdmfile, sdmscan=scan, inprefs=inprefs)
    else:
        logger.info('Parsing metadata from cands file')
        meta = metadata.oldstate_metadata(d, scan=scan)
        st = state.State(inmeta=meta, inprefs=inprefs, showsummary=False)

    st.rtpipe_version = float(d['rtpipe_version'])
    if st.rtpipe_version <= 1.54:
        logger.info('Candidates detected with rtpipe version {0}. All '
                    'versions <=1.54 used an incorrect DM scaling prefactor.'
                    .format(st.rtpipe_version))

    colnames = d['featureind']
    logger.info('Calculating candidate properties for scan {0}'.format(scan))
    df = pd.DataFrame(OrderedDict(zip(colnames, loc.transpose())))
    df2 = pd.DataFrame(OrderedDict(zip(st.features, prop.transpose())))
    df3 = pd.concat([df, df2], axis=1)[df.scan == scan]

    df3.metadata = st.metadata
    df3.prefs = st.prefs

    return st, df3


def pipeline_dataprep(st, candloc):
    """ Prepare (read, cal, flag) data for a given state and candloc.
    """

    segment, candint, dmind, dtind, beamnum = candloc.astype(int)

    # prep data
    data = source.read_segment(st, segment)
    data_prep = source.data_prep(st, data)

    return data_prep


def pipeline_datacorrect(st, candloc, data_prep=None):
    """ Prepare and correct for dm and dt sampling of a given candloc
    Can optionally pass in prepared (flagged, calibrated) data, if available.
    """

    if data_prep is None:
        data_prep = pipeline_dataprep(st, candloc)

    segment, candint, dmind, dtind, beamnum = candloc.astype(int)
    dt = st.dtarr[dtind]
    dm = st.dmarr[dmind]

    scale = None
    if hasattr(st, "rtpipe_version"):
        scale = 4.2e-3 if st.rtpipe_version <= 1.54 else None
    delay = util.calc_delay(st.freq, st.freq.max(), dm, st.inttime,
                            scale=scale)

    data_dm = search.dedisperse(data_prep, delay)
    data_dmdt = search.resample(data_dm, dt)

    return data_dmdt


def pipeline_imdata(st, candloc, data_dmdt=None):
    """ Generate image and phased visibility data for candloc.
    Phases to peak pixel in image of candidate.
    Can optionally pass in corrected data, if available.
    """

    segment, candint, dmind, dtind, beamnum = candloc.astype(int)
    dt = st.dtarr[dtind]
    dm = st.dmarr[dmind]

    uvw = st.get_uvw_segment(segment)
    wisdom = search.set_wisdom(st.npixx, st.npixy)

    if data_dmdt is None:
        data_dmdt = pipeline_datacorrect(st, candloc)

    i = candint//dt
    image = search.image(data_dmdt, uvw, st.npixx, st.npixy, st.uvres,
                         wisdom, integrations=[i])[0]
    dl, dm = st.pixtolm(np.where(image == image.max()))
    search.phase_shift(data_dmdt, uvw, dl, dm)
    dataph = data_dmdt[i-st.prefs.timewindow//2:i+st.prefs.timewindow//2].mean(axis=1)
    search.phase_shift(data_dmdt, uvw, -dl, -dm)

    canddata = search.CandData(state=st, loc=tuple(candloc), image=image,
                               data=dataph)

    # output is as from search.image_thresh
    return [canddata]


def pipeline_candidate(st, candloc, canddata=None):
    """ End-to-end pipeline to reproduce candidate plot and calculate features.
    Can optionally pass in image and corrected data, if available.
    """

    segment, candint, dmind, dtind, beamnum = candloc.astype(int)

    if canddata is None:
        canddatalist = pipeline_imdata(st, candloc)

    candidate = search.calc_features(canddatalist)

#    search.candplot(canddatalist)

    return candidate
