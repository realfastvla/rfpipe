from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)


from rfpipe import preferences, state, metadata

def oldcands_read(candsfile, sdmscan=None, sdmfile=None):
    """ Read old-style candfile and create new-style DataFrame
    """


    with open(candsfile) as pkl:                                                                                                       
        d = pickle.load(pkl)
        loc, prop = pickle.load(pkl)

    colnames = d['featureind']

    prefs = preferences.Preferences(**preferences.oldstate_preferences(d))
    if sdmfile and sdmscan:
        st = state.State(sdmfile=sdmfile, sdmscan=sdmscan, inprefs=prefs)
    else:
        meta = metadata.Metadata(**metadata.oldstate_metadata(d, scan=sdmscan))
        st = state.State(inprefs=prefs, inmeta=meta)

    # ** Probably also need to iterate state definition for each scan

    df = pandas.DataFrame(OrderedDict(zip(colnames, loc.transpose())))
    df2 = pandas.DataFrame(OrderedDict(zip(st.features, prop.transpose())))
    df3 = pandas.concat([df, df2], axis=1)

    df3.metadata = meta
    df3.prefs = prefs

    return df3
