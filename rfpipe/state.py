import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

import json, attr
from . import source
# from collections import OrderedDict #?






@attr.s(frozen=True)
class Parameters(object):
    """ Parameters are immutable and express half of info needed to define state.
    Using parameters with metadata produces a unique state and pipeline outcome.
    """

    # data selection
    chans = attr.ib(default=None)
    spw = attr.ib(default=None)
    excludeants = attr.ib(default=None)
    selectpol = attr.ib(default=None) # ['RR', 'LL', 'XX', 'YY']   # default processing assumes dual-pol

    # preprocessing
    read_tdownsample = attr.ib(default=1)
    read_fdownsample = attr.ib(default=1)
    l0 = attr.ib(default=0.)  # in radians
    m0 = attr.ib(default=0.)  # in radians
    timesub = attr.ib(default=None)
    flaglist = attr.ib(default=[('badchtslide', 4., 0.) , ('badap', 3., 0.2), ('blstd', 3.0, 0.05)])
    flagantsol = attr.ib(default=True)
    badspwpol = attr.ib(default=2.)
    applyonlineflags = attr.ib(default=True)
    gainfile = attr.ib(default=None)
    mock = attr.ib(default=0)

    # processing
    nthread = attr.ib(default=1)
    nchunk = attr.ib(default=0)
    nsegments = attr.ib(default=0)
    scale_nsegments = attr.ib(default=1)  # remove?
    memory_limit = attr.ib(default=20)

    # search
    dmarr = attr.ib(default=None)
    dtarr = attr.ib(default=1)
    dm_maxloss = attr.ib(default=0.05) # fractional sensitivity loss
    mindm = attr.ib(default=0)
    maxdm = attr.ib(default=0) # in pc/cm3
    dm_pulsewidth = attr.ib(default=3000)   # in microsec
    searchtype = attr.ib(default='image1')
    sigma_image1 = attr.ib(default=7.)
    sigma_image2 = attr.ib(default=7.)
    sigma_plot = attr.ib(default=7.)
    uvres = attr.ib(default=0)
    npixx = attr.ib(default=0)
    npixy = attr.ib(default=0)
    npix_max = attr.ib(default=0)
    uvoversample = attr.ib(default=1.)

    savenoise = attr.ib(default=False)
    savecands = attr.ib(default=False)
#    logfile = attr.ib(default=True)
    loglevel = attr.ib(default='INFO')


class State(object):
    """ Defines initial pipeline preferences and methods for calculating state.
    Uses attributes for immutable inputs and properties for derived quantities that depend on metadata.

    Scheme:
    1) initial, minimal state defines either parameters for later use or fixes final state
    2) read metadata from observation for given scan (if final value not yet set)
    3) run functions to set state (sets hashable state)
    4) may also set convenience attibutes
    5) run data processing for given segment

    these should be modified to change state based on input
    - nsegments or dmarr + memory_limit + metadata => segmenttimes
    - dmarr or dm_parameters + metadata => dmarr
    - uvoversample + npix_max + metadata => npixx, npixy
    """

    def __init__(self, paramfile=None, sdmfile=None, scan=None, version=1):
        """ Initialize parameter attributes with text file.
        Versions define functions that derive state from parameters and metadata
        """

        self.version = version

        logger.parent.setLevel(getattr(logging, 'INFO'))

        # get pipeline parameters
        inpars = parseparamfile(paramfile)  # returns empty dict for paramfile=None
        self.parameters = Parameters(**inpars)

        # get metadata
        if sdmfile and scan:
            metadata = source.sdm_metadata(sdmfile, scan)
            self.metadata = metadata

        logger.parent.setLevel(getattr(logging, self.parameters.loglevel))


    @property
    def dmarr(self):
        if self.parameters.dmarr:
            return self.parameters.dmarr
        else:
            return self.calc_dmarr(self.parameters.dm_maxloss, self.parameters.dm_pulsewidth, self.parameters.mindm, self.parameters.maxdm)


    @property
    def freq(self):
        # TODO: need to support spw selection and downsampling, e.g.:
        #    if spectralwindow[ii] in d['spw']:
        #    np.array([np.mean(spwch[i:i+d['read_fdownsample']]) for i in range(0, len(spwch), d['read_fdownsample'])], dtype='float32') / 1e9

        return self.metadata.freq_orig[self.parameters.chans]


    @property
    def nchan(self):
        return len(self.freq)


    @property
    def nspw(self):
        return len(self.metadata.spw_orig[self.parameters.spw])


    @property
    def spw_nchan_select(self):
        return [len([ch for ch in range(self.metadata.spw_chanr[i][0], self.metadata.spw_chanr[i][1]) if ch in self.parameters.chans])
                for i in range(len(self.metadata.spw_chanr))]


    @property
    def spw_chanr_select(self):
        chanr_select = []
        i0 = 0
        for nch in self.spw_nchan_select:
            chanr_select.append((i0, i0+nch))
            i0 += nch

        return chanr_select


    @property
    def uvres(self):
        if self.parameters.uvres:
            return self.parameters.uvres
        else:
            return self.uvres_full


    @property
    def uvres_full(self):
        return np.round(self.metadata.dishdiameter / (3e-1 / self.freq.min()) / 2).astype('int')


    @property
    def npixx_full(self):
        """ Finds optimal uv/image pixel extent in powers of 2 and 3"""

        urange_orig, vrange_orig = self.metadata.uvrange_orig
        urange = urange_orig * (self.freq.max() / self.metadata.freq_orig[0])
        powers = np.fromfunction(lambda i, j: 2**i*3**j, (14, 10), dtype='int')
        rangex = np.round(self.parameters.uvoversample*urange).astype('int')
        largerx = np.where(powers - rangex / self.uvres_full > 0,
                           powers, powers[-1, -1])
        p2x, p3x = np.where(largerx == largerx.min())
        return (2**p2x * 3**p3x)[0]


    @property
    def npixy_full(self):
        """ Finds optimal uv/image pixel extent in powers of 2 and 3"""

        urange_orig, vrange_orig = self.metadata.uvrange_orig
        vrange = vrange_orig * (self.freq.max() / self.metadata.freq_orig[0])
        rangey = np.round(self.parameters.uvoversample*vrange).astype('int')
        largery = np.where(powers - rangey / self.uvres_full > 0,
                           powers, powers[-1, -1])
        p2y, p3y = np.where(largery == largery.min())
        return (2**p2y * 3**p3y)[0]


    @property
    def npixx(self):
        """ Number of x pixels in uv/image grid.
        First defined by input parameter set with default to npixx_full
        """

        if self.parameters.npixx:
            return self.parameters.npixx
        else:
            if self.parameters.npix_max:
                npix = min(self.parameters.npix_max, self.npixx_full)
            return npix


    @property
    def npixy(self):
        """ Number of y pixels in uv/image grid.
        First defined by input parameter set with default to npixy_full
        """
        
        if self.parameters.npixy:
            return self.parameters.npixy
        else:
            if self.parameters.npix_max:
                npix = min(self.parameters.npix_max, self.npixy_full)
            return npix


    @property
    def fringetime(self):
        """ Estimate largest time span of a "segment".
        A segment is the maximal time span that can be have a single bg fringe subtracted and uv grid definition.
        Max fringe window estimated for 5% amp loss at first null averaged over all baselines. Assumes dec=+90, which is conservative.
        Returns time in seconds that defines good window.
        """

        maxbl = self.uvres*max(self.npixx, self.npixy)/2    # fringe time for image grid extent
        fringetime = 0.5*(24*3600)/(2*n.pi*maxbl/25.)   # max fringe window in seconds
        return fringetime


    @property
    def ants(self):
        return sorted([ant for ant in self.metadata.ants_orig if ant not in self.parameters.excludeants])


    @property
    def nants(self):
        return len(self.ants)


    @property
    def nbl(self):
        return self.nants*(self.nants-1)/2



    def calc_dmarr(self, dm_maxloss, dm_pulsewidth, mindm, maxdm):
        """ Function to calculate the DM values for a given maximum sensitivity loss.
        dm_maxloss is sensitivity loss tolerated by dm bin width. dm_pulsewidth is assumed pulse width in microsec.
        """

        # parameters
        tsamp = self.inttime*1e6  # in microsec
        k = 8.3
        freq = self.freq.mean()  # central (mean) frequency in GHz
        bw = 1e3*(self.freq[-1] - self.freq[0])
        ch = 1e3*(self.freq[1] - self.freq[0])  # channel width in MHz

        # width functions and loss factor
        dt0 = lambda dm: n.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2)
        dt1 = lambda dm, ddm: n.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2 + ((k*ddm*bw)/(freq**3.))**2)
        loss = lambda dm, ddm: 1 - n.sqrt(dt0(dm)/dt1(dm,ddm))
        loss_cordes = lambda ddm, dfreq, dm_pulsewidth, freq: 1 - (n.sqrt(n.pi) / (2 * 6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))) * erf(6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))  # not quite right for underresolved pulses

        if maxdm == 0:
            return [0]
        else:
            # iterate over dmgrid to find optimal dm values. go higher than maxdm to be sure final list includes full range.
            dmgrid = n.arange(mindm, maxdm, 0.05)
            dmgrid_final = [dmgrid[0]]
            for i in range(len(dmgrid)):
                ddm = (dmgrid[i] - dmgrid_final[-1])/2.
                ll = loss(dmgrid[i],ddm)
                if ll > dm_maxloss:
                    dmgrid_final.append(dmgrid[i])

        return dmgrid_final


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



def parseparamfile(paramfile=None):
    """ Read parameter file and set parameter values.
    File should have python-like syntax. Full file name needed.
    """
    

    pars = {}

    if paramfile:
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
                        pars[attribute.strip()] =  value_eval

    return pars


def parseyaml(self, paramfile, name='default'):
    # maybe use pyyaml to parse parameters more reliably
    # could save multiple per yml paramfile
    pass



