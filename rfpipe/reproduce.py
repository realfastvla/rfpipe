from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

import pickle
from collections import OrderedDict
import pandas as pd

from rfpipe import preferences, state, metadata, util, search, source


def oldcands_read(candsfile, sdmscan=None, sdmfile=None, returnstate=False, returndf=True):
    """ Read old-style candfile and create new-style DataFrame
    Metadata best defined by sdmfile/sdmscan, but can get most from old candsfile.
    """

    with open(candsfile, 'rb') as pkl:                                                                                                       
        d = pickle.load(pkl)
        loc, prop = pickle.load(pkl)

    inprefs = preferences.oldstate_preferences(d, scan=sdmscan)
    if sdmfile and sdmscan:
        st = state.State(sdmfile=sdmfile, sdmscan=sdmscan, inprefs=inprefs)
    else:
        inmeta = metadata.oldstate_metadata(d, scan=sdmscan)
        st = state.State(inprefs=inprefs, inmeta=inmeta)

    # need to track this to recalculate delay properly for old candidates
    st.rtpipe_version = float(d['rtpipe_version']) 
    if st.rtpipe_version <= 1.54:
        logger.info('Candidates detected with rtpipe version {0}. All versions <=1.54 used an incorrect DM scaling prefactor.'.format(st.rtpipe_version))

    # ** Probably also need to iterate state definition for each scan

    colnames = d['featureind']

    df = pd.DataFrame(OrderedDict(zip(colnames, loc.transpose())))
    df2 = pd.DataFrame(OrderedDict(zip(st.features, prop.transpose())))
    df3 = pd.concat([df, df2], axis=1)

    df3.metadata = st.metadata
    df3.prefs = st.prefs

    if returndf and not returnstate:
        return df3
    elif returnstate and not returndf:
        return st
    elif returnstate and returndf:
        return st, df3
    else:
        return


def pipeline(st, candloc):
    """ End-to-end processing to reproduce a given candloc (segment, integration, dmind, dtind, beamnum).
    Assumes sdm data for now.
    """

    segment, candint, dmind, dtind, beamnum = candloc
    dt = st.dtarr[dtind]
    dm = st.dmarr[dmind]

    # prep data
    data = source.read_segment(st, segment)
    data_prep = source.data_prep(st, data)

    # prepare to transform data
    uvw = st.get_uvw_segment(segment)
    wisdom = search.set_wisdom(st.npixx, st.npixy)
    scale = 4.2e-3 if st.rtpipe_version <= 1.54 else None
    delay = util.calc_delay(st.freq, st.freq.max(), dm, st.metadata.inttime, scale=scale)

    # dedisperse, resample, image, threshold
    data_dm = search.dedisperse(data_prep, delay)
    data_dmdt = search.resample(data_dm, dt)
    ims_thresh = search.image_thresh(data_dmdt, uvw, st.npixx, st.npixy, st.uvres, st.prefs.sigma_image1, wisdom, integrations=[candint/dt])
#    candplot = delayed(search.candplot)(st, ims_thresh, data_dm)

    search_coords = OrderedDict(zip(['segment', 'dmind', 'dtind', 'beamnum'], [segment, dmind, dtind, 0]))
    candidates = search.calc_features(st, ims_thresh, search_coords)

    return candidates
