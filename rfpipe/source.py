# defines data sources for pipeline
# first rate support for sdm files
# can generalize to include streaming data from CBE?

import rtpipe
import numpy as np

def dataprep(st, segment):
    data_read = rtpipe.parsesdm.read_bdf_segment(st, segment)
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
