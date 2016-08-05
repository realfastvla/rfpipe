# home of much of rtpipe.RT
# state transformation stages should be in state
from __future__ import division  # for Python 2

# playing with numba gridding
import numba, math
from numba import cuda
import numpy as np
#import cProfile


@cuda.jit()
def grid_visibilities(visdata, grid, us, vs, freqs):
    freq0 = freqs[-1]
    nint, nbl, nchan, npol = visdata.shape

    for i in xrange(nbl):
        for freq in freqs:
            u = int(us[i]*freq/freq0)
            v = int(vs[i]*freq/freq0)

        for m in xrange(nint):
            for n in xrange(npol):
                grid[u, v] += visdata[m, i, j, n]

def meantsub0(data):
    """ Reference pure python function for meantsub """

    nint, nbl, nchan, npol = data.shape
    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                nonzero = np.where(data[:,i,j,k] != 0j)
                data[:,i,j,k] -= data[nonzero,i,j,k].mean()
    

@cuda.jit
def meantsub(data):
    """ Calculate mean in time (ignoring zeros) and subtract in place """

    x,y,z = cuda.grid(3)
    nint, nbl, nchan, npol = data.shape
    if x < nbl and y < nchan and z < npol:
        sum = numba.complex64(0)
        weight = 0
        for i in range(nint):
            sum = sum + data[i, x, y, z]
            if data[i,x,y,z] == 0j:
                weight = weight + 1
        mean = sum/weight
        for i in range(nint):
            data[i, x, y, z] = data[i, x, y, z] - mean


@numba.vectorize([numba.bool_(numba.complex64)])
def get_mask(x):
    """ Returns equal sized array of 0/1 """
    
    return x != 0j


def correct_dm():
    """ Dispersion shift in place  """

    pass


def correct_dt():
    """ Resample (integrate) in time """

    pass


def image():
    """ Takes uv grid and returns image grid """

    pass


def imagearm():
    """ Takes visibilities and images arms of VLA """

    pass


def runcuda(func, data, threadsperblock, *args, **kwargs):
    """ Function to run cuda kernels while defining threads/blocks """

    blockspergrid = []
    for tpb, sh in threadsperblock, data.shape:
        blockspergrid.append = int(math.ceil(sh / tpb))
    func[tuple(blockspergrid), threadsperblock](data, *args, **kwargs)


if __name__ == '__main__':
    data = np.zeros(shape=(160,32,32,4), dtype='complex64')
    data.real = np.random.normal(size=(160,32,32,4))
    data.imag = np.random.normal(size=(160,32,32,4))
#    cProfile.run('meantsub0(data); print')
#    cProfile.run('runmeantsub(data); print')

    runcuda(meantsub, data, (16,16,1))



