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

    # ** not supporting dt>1

    segment, candint, dmind, dtind, beamnum = candloc

    # prep data
    data = source.read_segment(st, segment)#, timeout=vys_timeout, cfile=cfile)
    data_prep = source.data_prep(st, data)

    # prepare to transform data
    uvw = st.get_uvw_segment(segment)
    wisdom = search.set_wisdom(st.npixx, st.npixy)
    delay = util.calc_delay(st.freq, st.freq.max(), st.dmarr[dmind], st.metadata.inttime)

    # dedisperse, resample, image, threshold
    data_dm = search.dedisperse(data_prep, delay)
    ims_thresh = search.resample_image(data_dm, st.dtarr[dtind], uvw, st.npixx, st.npixy, st.uvres, st.prefs.sigma_image1, wisdom, integrations=[candint])
#    candplot = delayed(search.candplot)(st, ims_thresh, data_dm)

    search_coords = OrderedDict(zip(['segment', 'dmind', 'dtind', 'beamnum'], [segment, dmind, dtind, 0]))
    candidates = search.calc_features(st, ims_thresh, search_coords)

    return candidates, data_prep, data_dm
