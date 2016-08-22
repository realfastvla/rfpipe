# defines data sources for pipeline
# first rate support for sdm files
# can generalize to include streaming data from CBE?

import rtpipe
from . import search
import numpy as np

def dataprep(st, segment):
    data_read = rtpipe.parsesdm.read_bdf_segment(st, segment)
#    sols = rtpipe.parsecal.telcal_sol(st['gainfile'])   # parse gainfile
#    sols.set_selection(st['segmenttimes'][segment].mean(), st['freq']*1e9, rtlib.calc_blarr(st), calname='', pols=st['pols'], radec=(), spwind=[])
#    sols.apply(data)
#    rtpipe.RT.dataflag(st, data)
#    data_sub = search.meantsub(data_read)
    return data_read


def calc_uvw(st, segment):
    return rtpipe.parsesdm.get_uvw_segment(st, segment)


def randomdata(st):
    data = np.zeros(shape=(st['readints'], st['nbl'], st['nchan'], st['npol']), dtype='complex64')
    data.real = np.random.normal(size=data.shape)
    data.imag = np.random.normal(size=data.shape)
    return data


def randomuvw(st):
    return np.random.randint(-100, 100, size=st['nbl'])
