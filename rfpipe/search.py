# home of much of rtpipe.RT
# state transformation stages should be in state
from __future__ import division  # for Python 2

# playing with numba gridding
import numba, math
from numba import cuda
import numpy as np
import threading
import pyfftw
#import cProfile

##
# profiling with mean time subtraction algorithm
##

def meantsub_python(data):
    """ Reference pure python function for meantsub """

    nint, nbl, nchan, npol = data.shape
    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                nonzero = np.where(data[:,i,j,k] != 0j)
                data[:,i,j,k] -= data[nonzero,i,j,k].mean()
    

@cuda.jit
def meantsub_cuda(data):
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


@numba.jit
def meantsub_numba(data, nopython=True):
    """ Calculate mean in time (ignoring zeros) and subtract in place """

    nint, nbl, nchan, npol = data.shape
    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                ss = numba.complex64(0)
                weight = 0
                for l in range(nint):
                    ss += data[l, i, j, k]
                    if data[l, i, j, k] != 0j:
                        weight = weight + 1
                mean = ss/weight
                for l in range(nint):
                    data[l, i, j, k] -= mean


@numba.guvectorize([(numba.complex64[:,:,:], numba.complex64[:,:,:])], '(m,n,o)->(m,n,o)', nopython=True, target='parallel')
def meantsub_gu(data, res):
    """ Vectorizes over time axis *at end*. Use np.moveaxis(0, 3) for input visbility array """ 

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ss = numba.complex64(0)
            weight = numba.int16(0)
            for k in range(data.shape[2]):
                ss += data[i,j,k]
                if data[i,j,k] != 0j:
                    weight = weight + 1
                mean = ss/weight
            for k in range(data.shape[0]):
                res[i,j,k] = data[i,j,k] - mean
    

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


@numba.jit
def fft1d_numba(data, res):
    sh = data.shape

    for i in range(sh[1]):
        for j in range(sh[2]):
            for k in range(sh[3]):
                res[:, i, j, k] = np.fft.fft(data[:, i, j, k])


def fft1d_python(data, res):
    sh = data.shape

    for i in range(sh[1]):
        for j in range(sh[2]):
            for k in range(sh[3]):
                res[:, i, j, k] = np.fft.fft(data[:, i, j, k])

@numba.jit(nogil=True)
def npifft2(data, result):
    """ Input for multithread wrapper. nogil seems to improve nothing """

    result = np.fft.ifft2(data)


@numba.jit(nogil=True)
def fftw(data, result):
    ifft2 = pyfftw.builders.ifft2(data, auto_align_input=True, auto_contiguous=True, planner_effort='FFTW_PATIENT')
    result = ifft2(data)


def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_mt(data):
        result = np.empty_like(data)

        # Create argument tuples for each input chunk
        length = len(data)
        chunklen = length // numthreads
        i0s = [chunklen*i for i in range(numthreads)]
        i1s = [chunklen*(i+1) for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=(data[i0:i1], result[i0:i1])) for (i0,i1) in zip(i0s, i1s)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt
    

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



