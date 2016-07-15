

# these should be modified to change state based on input
# - nsegments or dmarr + memory_limit + metadata => segmenttimes
# - dmarr or dm_parameters + metadata => dmarr
# - uvoversample + npix_max + metadata => npixx, npixy

# maybe this all goes in state.py outside of class?

def apply_metadata(state, sdmfile):
    """ Run all functions to apply metadata to transform initial state """

    parsesdm(sdmfile, state)
    set_dmgrid(state)
    set_imagegrid(state)
    set_segments(state)


def set_segments(state):
    """ Helper function for set_pipeline to define segmenttimes list, given nsegments definition
    """

    # this casts to int (flooring) to avoid 0.5 int rounding issue. 
    stopdts = n.linspace(d['t_overlap']/d['inttime'], d['nints'], d['nsegments']+1)[1:]   # nseg+1 assures that at least one seg made
    startdts = n.concatenate( ([0], stopdts[:-1]-d['t_overlap']/d['inttime']) )
            
    segmenttimes = []
    for (startdt, stopdt) in zip(d['inttime']*startdts, d['inttime']*stopdts):
        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+startdt/(24*3600),'d'),form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+stopdt/(24*3600), 'd'), form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        segmenttimes.append((starttime, stoptime))
    d['segmenttimes'] = n.array(segmenttimes)
    totaltimeread = 24*3600*(d['segmenttimes'][:, 1] - d['segmenttimes'][:, 0]).sum()            # not guaranteed to be the same for each segment
    d['readints'] = n.round(totaltimeread / (d['inttime']*d['nsegments']*d['read_tdownsample'])).astype(int)
    d['t_segment'] = totaltimeread/d['nsegments']


def calc_memory_footprint(d, headroom=4., visonly=False, limit=False):
    """ Given pipeline state dict, this function calculates the memory required
    to store visibilities and make images.
    headroom scales visibility memory size from single data object to all copies (and potential file read needs)
    limit=True returns a the minimum memory configuration
    Returns tuple of (vismem, immem) in units of GB.
    """

    toGB = 8/1024.**3   # number of complex64s to GB
    d0 = d.copy()

    # limit defined for dm sweep time and max nchunk/nthread ratio
    if limit:
        d0['readints'] = d['t_overlap']/d['inttime']
        d0['nchunk'] = max(d['dtarr'])/min(d['dtarr']) * d['nthread']

    vismem = headroom * datasize(d0) * toGB
    if visonly:
        return vismem
    else:
        immem = d0['nthread'] * (d0['readints']/d0['nchunk'] * d0['npixx'] * d0['npixy']) * toGB
        return (vismem, immem)


def calc_fringetime(d):
    """ Estimate largest time span of a "segment".
    A segment is the maximal time span that can be have a single bg fringe subtracted and uv grid definition.
    Max fringe window estimated for 5% amp loss at first null averaged over all baselines. Assumes dec=+90, which is conservative.
    Returns time in seconds that defines good window.
    """

    maxbl = d['uvres']*d['npix']/2    # fringe time for imaged data only
    fringetime = 0.5*(24*3600)/(2*n.pi*maxbl/25.)   # max fringe window in seconds
    return fringetime


def calc_dmgrid(d, maxloss=0.05, dt=3000., mindm=0., maxdm=0.):
    """ Function to calculate the DM values for a given maximum sensitivity loss.
    maxloss is sensitivity loss tolerated by dm bin width. dt is assumed pulse width in microsec.
    """

    # parameters
    tsamp = d['inttime']*1e6  # in microsec
    k = 8.3
    freq = d['freq'].mean()  # central (mean) frequency in GHz
    bw = 1e3*(d['freq'][-1] - d['freq'][0])
    ch = 1e3*(d['freq'][1] - d['freq'][0])  # channel width in MHz

    # width functions and loss factor
    dt0 = lambda dm: n.sqrt(dt**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2)
    dt1 = lambda dm, ddm: n.sqrt(dt**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2 + ((k*ddm*bw)/(freq**3.))**2)
    loss = lambda dm, ddm: 1 - n.sqrt(dt0(dm)/dt1(dm,ddm))
    loss_cordes = lambda ddm, dfreq, dt, freq: 1 - (n.sqrt(n.pi) / (2 * 6.91e-3 * ddm * dfreq / (dt*freq**3))) * erf(6.91e-3 * ddm * dfreq / (dt*freq**3))  # not quite right for underresolved pulses

    if maxdm == 0:
        return [0]
    else:
        # iterate over dmgrid to find optimal dm values. go higher than maxdm to be sure final list includes full range.
        dmgrid = n.arange(mindm, maxdm, 0.05)
        dmgrid_final = [dmgrid[0]]
        for i in range(len(dmgrid)):
            ddm = (dmgrid[i] - dmgrid_final[-1])/2.
            ll = loss(dmgrid[i],ddm)
            if ll > maxloss:
                dmgrid_final.append(dmgrid[i])

    return dmgrid_final


def set_imagegrid(state):
    """ """

    if d['uvres'] == 0:
        d['uvres'] = d['uvres_full']
    else:
        urange = d['urange'][scan]*(d['freq'].max()/d['freq_orig'][0])   # uvw from get_uvw already in lambda at ch0
        vrange = d['vrange'][scan]*(d['freq'].max()/d['freq_orig'][0])
        powers = n.fromfunction(lambda i,j: 2**i*3**j, (14,10), dtype='int')   # power array for 2**i * 3**j
        rangex = n.round(d['uvoversample']*urange).astype('int')
        rangey = n.round(d['uvoversample']*vrange).astype('int')
        largerx = n.where(powers-rangex/d['uvres'] > 0, powers, powers[-1,-1])
        p2x, p3x = n.where(largerx == largerx.min())
        largery = n.where(powers-rangey/d['uvres'] > 0, powers, powers[-1,-1])
        p2y, p3y = n.where(largery == largery.min())
        d['npixx_full'] = (2**p2x * 3**p3x)[0]
        d['npixy_full'] = (2**p2y * 3**p3y)[0]

    # set number of pixels to image
    d['npixx'] = d['npixx_full']
    d['npixy'] = d['npixy_full']
    if 'npix_max' in d:
        if d['npix_max']:
            d['npixx'] = min(d['npix_max'], d['npixx_full'])
            d['npixy'] = min(d['npix_max'], d['npixy_full'])
    if d['npix']:
        d['npixx'] = d['npix']
        d['npixy'] = d['npix']
    else:
        d['npix'] = max(d['npixx'], d['npixy'])   # this used to define fringe time

