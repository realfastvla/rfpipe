from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

import pickle
from collections import OrderedDict
import pandas as pd

from rfpipe import preferences, state, metadata


def oldcands_read(candsfile, sdmscan=None, sdmfile=None):
    """ Read old-style candfile and create new-style DataFrame
    Metadata best defined by sdmfile/sdmscan, but can get most from old candsfile.
    """

    with open(candsfile, 'rb') as pkl:                                                                                                       
        d = pickle.load(pkl)
        loc, prop = pickle.load(pkl)

    colnames = d['featureind']

    inprefs = preferences.oldstate_preferences(d)
    if sdmfile and sdmscan:
        st = state.State(sdmfile=sdmfile, sdmscan=sdmscan, inprefs=inprefs)
    else:
        inmeta = metadata.oldstate_metadata(d, scan=sdmscan)
        st = state.State(inprefs=inprefs, inmeta=inmeta)

    # ** Probably also need to iterate state definition for each scan

    df = pd.DataFrame(OrderedDict(zip(colnames, loc.transpose())))
    df2 = pd.DataFrame(OrderedDict(zip(st.features, prop.transpose())))
    df3 = pd.concat([df, df2], axis=1)

    df3.metadata = st.metadata
    df3.prefs = st.prefs

    return df3
