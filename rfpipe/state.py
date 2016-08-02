import json
# from collections import OrderedDict #?

class State(object):

    def  __init__(self, paramfile='', version=2, **kwargs):
        """ Define rfpipe state 
    
        For version==1, this parses in the rtpipe-style. Input uses python exec to run each line.
        For version>=2, this will make more general rfpipe-style pipeline state.
        """

        self.version = version

        if version == 1:
            # default values
            self.chans = []; self.spw = []    
            self.nskip = 0; self.excludeants = []; self.read_tdownsample = 1; self.read_fdownsample = 1
            self.selectpol = ['RR', 'LL', 'XX', 'YY']   # default processing assumes dual-pol
            self.nthread = 1; self.nchunk = 0; self.nsegments = 0; self.scale_nsegments = 1
            self.timesub = ''
            self.dmarr = []; self.dtarr = [1]    # dmarr = [] will autodetect, given other parameters
            self.dm_maxloss = 0.05; self.maxdm = 0; self.dm_pulsewidth = 3000   # dmloss is fractional sensitivity loss, maxdm in pc/cm3, width in microsec
            self.searchtype = 'image1'; self.sigma_image1 = 7.; self.sigma_image2 = 7.
            self.l0 = 0.; self.m0 = 0.
            self.uvres = 0; self.npix = 0; self.npix_max = 0; self.uvoversample = 1.
            self.flaglist = [('badchtslide', 4., 0.) , ('badap', 3., 0.2), ('blstd', 3.0, 0.05)]
            self.flagantsol = True; self.gainfile = ''; self.bpfile = ''; self.fileroot = ''; self.applyonlineflags = True
            self.savenoise = False; self.savecands = False; self.logfile = True; self.loglevel = 'INFO'
            self.writebdfpkl = False; self.mock = 0
                           
            # overload with the parameter file values, if provided
            if len(paramfile):
                self.parseparamfile(paramfile)

        elif version == 2:
            # parse paramfile

            self.__dict__['featureind'] = ['scan', 'segment', 'int', 'dmind', 'dtind', 'beamnum']  # feature index. should be stable.

            # parse kwargs and overload defaults
            for key in kwargs:
                setattr(self, key, value)

            raise NotImplementedError


    def parseparamfile(self, paramfile):
        """ Read parameter file and set parameter values.
        File should have python-like syntax. Full file name needed.
        """

        with open(paramfile, 'r') as f:
            for line in f.readlines():
                line_clean = line.rstrip('\n').split('#')[0]   # trim out comments and trailing cr
                if line_clean and '=' in line:   # use valid lines only
                    attribute, value = line_clean.split('=')
                    try:
                        value_eval = eval(value.strip())
                    except NameError:
                        value_eval = value.strip()
                    finally:
                        setattr(self, attribute.strip(), value_eval)


    def parseyaml(self, paramfile, name='default'):
        # maybe use pyyaml to parse parameters more reliably
        # could save multiple per yml paramfile
        pass


    @property
    def reproducekeys(self):
        """ Minimal set of state keys required to assure that state can reproduce a given candidate. 

        Should be independent of data? Or assumes post-reading of metadata and applying parameter generation functions?
        """

        # this set is input, version defines functions that transform this to pipeline state
        return sorted(['sdmfile', 'excludeants', 'read_tdownsample', 'read_fdownsample',
                       'selectpol', 'timesub', 'dmarr', 'dtarr', 'searchtype',
#                       'features', 'sigma_image1', 'sigma_image2', 'sigma_bisp',   # not sure about this set
                       'uvres', 'npixx', 'npixy', 'version',
                       'flaglist', 'gainfile', 'bpfile', 'onlineflags'])

    # should each of the above actually be considered transformation functions? 
    #  so, input is observation name, next step is transform by ignore ants, next step...
    #
    #   nsegments or dmarr + memory_limit + metadata => segmenttimes
    #   dmarr or dm_parameters + metadata => dmarr
    #   uvoversample + npix_max + metadata => npixx, npixy

    # flagantsol should be merged with flaglist


    @property
    def hash(self):
        """ Hash that identifies pipeline state that produces unique set of output products """

        extant_keys = self.keys()
        if all([kk in self.reproducekeys for kk in extant_keys]):
            values = [self[key] for key in self.reproducekeys]
            return hash(json.dumps(repr(values)))  # is repr safe?
        else:
            print('Cannot make hash without minimal set defined in reproducekeys property.')
            return None


    @property
    def defined(self):
        return self.__dict__.keys()


    def __getitem__(self, key):
        return self.__dict__[key]


    def __setitem__(self, key, value):
        self.__dict__[key] = value


    def __delitem__(self, key):
        del self.__dict__[key]


    def __str__(self):
        return str(self.__dict__)


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

