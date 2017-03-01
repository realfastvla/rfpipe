from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

from rtpipe.parsecal import telcal_sol

def apply_telcal(st, segment, data):
    """ Applies gain calibration from telcal file to input data """

    data2 = data.copy()
    
    radec = ()
    spwind = []
    calname = ''  # set defaults

    sols = telcal_sol(st.gainfile)
    sols.set_selection(st.segmenttimes[segment].mean(), st.freq*1e9, st.blarr, calname=calname, pols=st.pols, radec=radec, spwind=spwind)
    sols.apply(data2)

    return data2
