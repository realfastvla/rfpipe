# home of much of rtpipe.RT
# state transformation stages should be in state
from __future__ import division  # for Python 2

# playing with numba gridding
import numba, math
from numba import cuda
import numpy as np
import pyfftw
from numba import float32, int32, int64, float_, complex64, bool_

##
## utilities
##

@numba.vectorize([int32(float32, float32, float_, float_)], nopython=True)
def calc_delay(freq, freqref, dm, inttime):
    """ Calculates the delay due to dispersion relative to freqref in integer units of inttime """

    return np.round(4.2e-3 * dm * (1/(freq*freq) - 1/(freqref*freqref))/inttime, 0)


#@numba.vectorize([int32(float32, float32, float32, float_)], nopython=True)
@numba.vectorize(nopython=True)
def uvcell(uv, freq, freqref, uvres):
    """ Scales u or v coord by freq and rounds to units of uvres """

    return np.round(uv*freq/freqref/uvres, 0)


@numba.vectorize([bool_(complex64)])
def get_mask(x):
    """ Returns equal sized array of 0/1 """
    
    return x != 0j


def runcuda(func, data, threadsperblock, *args, **kwargs):
    """ Function to run cuda kernels while defining threads/blocks """

    blockspergrid = []
    for tpb, sh in threadsperblock, data.shape:
        blockspergrid.append = int32(math.ceil(sh / tpb))
    func[tuple(blockspergrid), threadsperblock](data, *args, **kwargs)


##
## data prep
##

@numba.jit(nopython=True)
def meantsub(data):
    """ Calculate mean in time (ignoring zeros) and subtract in place

    Could ultimately parallelize by computing only on subset of data.
    """

    nint, nbl, nchan, npol = data.shape

    for i in range(nbl):
        for j in range(nchan):
            for k in range(npol):
                ss = complex64(0)
                weight = 0
                for l in range(nint):
                    ss += data[l, i, j, k]
                    if data[l, i, j, k] != 0j:
                        weight = weight + 1
                if weight:
                    mean = ss/weight
                else:
                    mean = 0j
                for l in range(nint):
                    data[l, i, j, k] -= mean
    return data


@numba.guvectorize([(complex64[:,:,:], complex64[:,:,:])], '(m,n,o)->(m,n,o)', nopython=True, target='parallel')
def meantsub_gu(data, res):
    """ Vectorizes over time axis *at end*. Use np.moveaxis(0, 3) for input visbility array """ 

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ss = complex64(0)
            weight = int32(0)
            for k in range(data.shape[2]):
                ss += data[i,j,k]
                if data[i,j,k] != 0j:
                    weight = weight + 1
                mean = ss/weight
            for k in range(data.shape[0]):
                res[i,j,k] = data[i,j,k] - mean
    

@numba.jit(nopython=True)
def dedisperse(data, freqs, inttime, dm):
    """ Dispersion shift in place """

    sh = data.shape
    delay = calc_delay(freqs, freqs[-1], float_(dm), float_(inttime))

    for k in range(sh[2]):
        if delay[k] > 0:
            for i in range(sh[0] - delay[k]):
                iprime = i + delay[k]
                for l in range(sh[3]):
                    for j in range(sh[1]):
                        data[i,j,k,l] = data[iprime,j,k,l]

    return data


#@numba.jit([complex64[:,:,:,:](complex64[:,:,:,:], int32)], nopython=True)
@numba.jit(nopython=True)
def resample(data, resample):
    """ Resample (integrate) in place """

    sh = data.shape
    newlen0 = int64(sh[0]/resample)
    if resample > 1:
        newdata = np.zeros_like(data[:newlen0])

        for j in range(sh[1]):
            for k in range(sh[2]):
                for l in range(sh[3]):
                    for i in range(newlen0):
                        iprime = int64(i*resample)
                        for r in range(resample):
                            newdata[i,j,k,l] = newdata[i,j,k,l] + data[iprime+r,j,k,l]
                        newdata[i,j,k,l] = newdata[i,j,k,l]/resample

        return newdata
    else:
        return data



##
## CUDA 
##

@cuda.jit
def meantsub_cuda(data):
    """ Calculate mean in time (ignoring zeros) and subtract in place """

    x,y,z = cuda.grid(3)
    nint, nbl, nchan, npol = data.shape
    if x < nbl and y < nchan and z < npol:
        sum = complex64(0)
        weight = 0
        for i in range(nint):
            sum = sum + data[i, x, y, z]
            if data[i,x,y,z] == 0j:
                weight = weight + 1
        mean = sum/weight
        for i in range(nint):
            data[i, x, y, z] = data[i, x, y, z] - mean

## 
## fft and imaging
##

#@numba.jit([complex64[:,:](complex64[:,:,:,:], float32[:], float32[:], float32[:], int32, int32, float_)], nopython=True)
@numba.jit(nopython=True)
def grid_visibilities(visdata, us, vs, freqs, npixx, npixy, uvres):
    """ Grid visibilities into rounded uv coordinates """

    nint, nbl, nchan, npol = visdata.shape

    grid = np.zeros(shape=(npixx, npixy), dtype=np.complex64)

    for j in range(nbl):
        ubl = uvcell(us[j], freqs, freqs[-1], uvres)
        vbl = uvcell(vs[j], freqs, freqs[-1], uvres)

#        if np.logical_and((np.abs(ubl) < npixx/2), (np.abs(vbl) < npixy/2)):
        for k in range(nchan):
            if (ubl[k] < npixx/2) and (np.abs(vbl[k]) < npixy/2):
                u = int64(np.mod(ubl[k], npixx))
                v = int64(np.mod(vbl[k], npixy))
                for i in range(nint):
                    for l in xrange(npol):
                        grid[u, v] = grid[u, v] + visdata[i, j, k, l]

    return grid


def imagearm():
    """ Takes visibilities and images arms of VLA """

    pass



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

@numba.jit
def npifft2(data, result):
    """ Input for multithread wrapper. nogil seems to improve nothing """

    result = np.fft.ifft2(data)


@numba.jit
def image_fftw(data, result):
    ifft2 = pyfftw.builders.ifft2(data, auto_align_input=True, auto_contiguous=True, planner_effort='FFTW_PATIENT')
    result = ifft2(data)
    return result


