from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import pickle
import os.path
from collections import OrderedDict
import numpy as np
import pandas as pd
from rfpipe import preferences, state, util, search, source

import logging
logger = logging.getLogger(__name__)


def oldcands_read(candsfile, sdmscan=None, sdmfile=None):
    """ Read old-style candfile and create new-style DataFrame
    Metadata best defined by sdmfile/sdmscan, but can get most from old
    candsfile.
    If no file or scan argument specified, it will return a list of (st, df)
    tuples.
    """

    with open(candsfile, 'rb') as pkl:
        d = pickle.load(pkl)
        loc, prop = pickle.load(pkl)

    scanind = d['featureind'].index('scan')
    scans = np.unique(loc[:, scanind])

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
    st = state.State(sdmfile=sdmfile, sdmscan=scan, inprefs=inprefs)

    st.rtpipe_version = float(d['rtpipe_version'])
    if st.rtpipe_version <= 1.54:
        logger.info('Candidates detected with rtpipe version {0}. All versions \
                    <=1.54 used an incorrect DM scaling prefactor.'
                    .format(st.rtpipe_version))

    colnames = d['featureind']
    logger.info('Calculating candidate properties for scan {0}'.format(scan))
    df = pd.DataFrame(OrderedDict(zip(colnames, loc.transpose())))
    df2 = pd.DataFrame(OrderedDict(zip(st.features, prop.transpose())))
    df3 = pd.concat([df, df2], axis=1)[df.scan == scan]

    df3.metadata = st.metadata
    df3.prefs = st.prefs

    return st, df3


def pipeline(st, candloc, get='candidate'):
    """ End-to-end processing to reproduce a given candloc (segment,
    integration, dmind, dtind, beamnum).
    Assumes sdm data and positive SNR candidates for now.
    """

    segment, candint, dmind, dtind, beamnum = candloc.astype(int)
    dt = st.dtarr[dtind]
    dm = st.dmarr[dmind]

    # prep data
    data = source.read_segment(st, segment)
    data_prep = source.data_prep(st, data)

    # prepare to transform data
    uvw = st.get_uvw_segment(segment)

    wisdom = search.set_wisdom(st.npixx, st.npixy)
    scale = 4.2e-3 if st.rtpipe_version <= 1.54 else None
    delay = util.calc_delay(st.freq, st.freq.max(), dm, st.metadata.inttime,
                            scale=scale)

    # dedisperse, resample, image, threshold
    data_dm = search.dedisperse(data_prep, delay)
    data_dmdt = search.resample(data_dm, dt)
#    candplot = delayed(search.candplot)(st, ims_thresh, data_dm)

    if get == 'dmdt':
        return data_dmdt
    elif get == 'image':
        image = search.image(data_dmdt, uvw, st.npixx, st.npixy, st.uvres,
                             wisdom, integrations=[candint/dt])
        return image
    elif get == 'phased':
        dl, dm = st.pixtolm(np.where(image == image.max()))
        search.phase_shift(data_dmdt, uvw, dl, dm)
        return data_dmdt
    elif get == 'candidate':
        image = search.image(data_dmdt, uvw, st.npixx, st.npixy, st.uvres,
                             wisdom, integrations=[candint/dt])
        snr = image.max()/util.madtostd(image)
        imgall = ([image], [snr], [candint/dt])
        search_coords = OrderedDict(zip(['segment', 'dmind', 'dtind',
                                         'beamnum'],
                                        [segment, dmind, dtind, 0]))
        candidate = search.calc_features(st, imgall, search_coords)
        return candidate
    elif get == 'plot':
        image = search.image(data_dmdt, uvw, st.npixx, st.npixy, st.uvres,
                             wisdom, integrations=[candint/dt])
        snr = image.max()/util.madtostd(image)
        imgall = ([image], [snr], [candint/dt])
        loclabel = [st.metadata.scan, segment, candint, dmind, dtind, beamnum]
        search.candplot(st, imgall, data_dmdt, loclabel, snrs=[snr])
