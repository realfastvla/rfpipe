# home of much of rtpipe.RT
# state transformation stages should be in state



# playing with numba gridding
import numba
import numpy as np

npixx, npixy = 1024, 1024
grid = np.zeros((npixx, npixy), dtype='complex64')

@jit
def grid_visibilities(data, grid, us, vs, freqs):
    freq0 = freqs[-1]
    nint, nbl, nchan, npol = data.shape

    for i in xrange(nbl):
        for freq in freqs:
        u = int(us[i]*freq/freq0)
        v = int(vs[i]*freq/freq0)

        for m in xrange(nint):
            for n xrange(npol):
                grid[u, v] += data[m, i, j, n]
