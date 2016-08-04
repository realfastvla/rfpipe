# home of much of rtpipe.RT
# state transformation stages should be in state
from __future__ import division  # for Python 2



# playing with numba gridding
import numba
from numba import cuda
import operator


@cuda.jit()
def grid_visibilities(visdata, grid, us, vs, freqs):
    freq0 = freqs[-1]
    nint, nbl, nchan, npol = visdata.shape

    for i in xrange(nbl):
        for freq in freqs:
        u = int(us[i]*freq/freq0)
        v = int(vs[i]*freq/freq0)

        for m in xrange(nint):
            for n xrange(npol):
                grid[u, v] += visdata[m, i, j, n]


@numba.vectorize([numba.bool_(numba.complex64)])
def get_mask(x):
    """ Returns equal sized array of 0/1 """
    
    return x != 0j


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


def runmeantsub(data):
    threadsperblock = (16, 16, 1)
    blockspergrid_x = int(math.ceil(data.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(data.shape[1] / threadsperblock[1]))
    blockspergrid_z = int(math.ceil(data.shape[2] / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    meantsub[blockspergrid, threadsperblock](data)

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



def run(data):
    threadsperblock = 32
    blockspergrid = (data.size + (threadsperblock - 1)) // threadperblock
    grid_visibilities[blockspergrid, threadsperblock](data)
