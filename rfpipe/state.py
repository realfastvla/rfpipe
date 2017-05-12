#from __future__ import print_function, division, absolute_import, unicode_literals
#from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger('rfpipe')

import json, attr, os
from . import source, util
import numpy as np
from scipy.special import erf
# from collections import OrderedDict #?

import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
logger.info('Using pwkit casa')


@attr.s
class Preferences(object): 
    """ Preferences *should* be immutable and express half of info needed to define state.
    Using preferences with metadata produces a unique state and pipeline outcome.

    TODO: can we freeze attributes while still having cached values?
    """

    # data selection
    chans = attr.ib(default=None)
    spw = attr.ib(default=None)
    excludeants = attr.ib(default=())
    selectpol = attr.ib(default='auto')  # 'auto', 'all'
    fileroot = attr.ib(default=None)

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
    memory_limit = attr.ib(default=20)

    # search
    dmarr = attr.ib(default=None)
    dtarr = attr.ib(default=None)
    dm_maxloss = attr.ib(default=0.05) # fractional sensitivity loss
    mindm = attr.ib(default=0)
    maxdm = attr.ib(default=0) # in pc/cm3
    dm_pulsewidth = attr.ib(default=3000)   # in microsec
    searchtype = attr.ib(default='image1')  # supported: image1, image1stat
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

    def __init__(self, paramfile=None, config=None, sdmfile=None, scan=None, inpars={}, inmeta={}, version=1):
        """ Initialize parameter attributes with text file.
        params is a dict with key-value pairs to overload paramfile values.
        Versions define functions that derive state from parameters and metadata

        Metadata source can be:
        1) Config object is (expected to be) like EVLA_config object prototyped for pulsar work by Paul.
        2) sdmfile and scan are as in rtpipe.

        inmeta is a dict with key-value pairs to overload metadata
        """

        self.version = version

        # get pipeline parameters
        params = parseparamfile(paramfile)  # returns empty dict for paramfile=None

        # optionally overload parameters
        for key in inpars:
            params[key] = inpars[key]

        self.preferences = Preferences(**inpars)

        # get metadata
        if sdmfile and scan:
            meta = source.sdm_metadata(sdmfile, scan)
        elif config and not (sdmfile or scan):
            meta = source.config_metadata(config)
        else:
            meta = {}

        # optionally overload metadata
        for key in inmeta:
            meta[key] = inmeta[key]

        self.metadata = source.Metadata(**meta)

        logger.parent.setLevel(getattr(logging, self.preferences.loglevel))
        self.summarize()


    def summarize(self):
        """ Print overall state, if metadata set """

        if self.metadata.atdefaults():
            logger.info('Metadata not set. Cannot calculate properties')
        else:
            logger.info('')
            logger.info('Pipeline summary:')

            logger.info('\t Products saved with {0}. telcal calibration with {1}.'.format(self.fileroot, os.path.basename(self.gainfile)))
            logger.info('\t Using {0} segment{1} of {2} ints ({3} s) with overlap of {4} s'.format(self.nsegments, "s"[not self.nsegments-1:], self.readints, self.t_segment, self.t_overlap))
            if self.t_overlap > self.t_segment/3.:
                logger.info('\t\t Lots of segments needed, since Max DM sweep ({0} s) close to segment size ({1} s)'.format(self.t_overlap, self.t_segment))

            logger.info('\t Downsampling in time/freq by {0}/{1}.'.format(self.preferences.read_tdownsample, self.preferences.read_fdownsample))
            logger.info('\t Excluding ants {0}'.format(self.preferences.excludeants))
            logger.info('\t Using pols {0}'.format(self.pols))
            logger.info('')
            
            logger.info('\t Search with {0} and threshold {1}.'.format(self.preferences.searchtype, self.preferences.sigma_image1))
            logger.info('\t Using {0} DMs from {1} to {2} and dts {3}.'.format(len(self.dmarr), min(self.dmarr), max(self.dmarr), self.dtarr))
            logger.info('\t Using uvgrid npix=({0}, {1}) and res={2}.'.format(self.npixx, self.npixy, self.uvres))
            logger.info('\t Expect {0} thermal false positives per segment.'.format(self.nfalse))
            
            logger.info('')
            logger.info('\t Visibility memory usage is {0} GB/segment'.format(self.vismem))
#            logger.info('\t Imaging in {0} chunk{1} using max of {2} GB/segment'.format(self.nchunk, "s"[not self.nsegments-1:], immem))
#            logger.info('\t Grand total memory usage: {0} GB/segment'.format(vismem + immem))


    @property
    def fileroot(self):
        if self.preferences.fileroot:
            return self.preferences.fileroot
        else:
            return os.path.basename(self.metadata.filename)


    @property
    def dmarr(self):
        if not hasattr(self, '_dmarr'):
            if self.preferences.dmarr:
                self._dmarr = self.preferences.dmarr
            else:
                self._dmarr = calc_dmarr(self)

        return self._dmarr


    @property
    def dtarr(self):
        if self.preferences.dtarr:
            return self.preferences.dtarr
        else:
            return [1]


    @property
    def freq(self):
        # TODO: need to support spw selection and downsampling, e.g.:
        #    if spectralwindow[ii] in d['spw']:
        #    np.array([np.mean(spwch[i:i+d['read_fdownsample']]) for i in range(0, len(spwch), d['read_fdownsample'])], dtype='float32') / 1e9

        return self.metadata.freq_orig[self.chans]


    @property
    def chans(self):
        """ List of channel indices to use. Drawn from parameters, with backup to take all defined in metadata. """
        if self.preferences.chans:
            return self.preferences.chans
        else:
            return range(sum(self.metadata.spw_nchan))


    @property
    def nchan(self):
        return len(self.chans)


    @property
    def dmshifts(self):
        """ Calculate max DM delay in units of integrations for each dm trial.

        TODO: probably should put dm calculation into a library module for calling from all parts of code base
        """


        return [util.calc_delay(self.freq, self.freq[-1], dm, self.metadata.inttime).max() for dm in self.dmarr]
        

    @property
    def t_overlap(self):
        """ Max DM delay in seconds. Gets cached. """

        if not hasattr(self, '_t_overlap'):
            self._t_overlap = max(self.dmshifts)*self.metadata.inttime
        return self._t_overlap


    @property
    def nspw(self):
        return len(self.metadata.spw_orig[self.preferences.spw] if self.preferences.spw else self.metadata.spw_orig)


    @property
    def spw_nchan_select(self):
        return [len([ch for ch in range(self.metadata.spw_chanr[i][0], self.metadata.spw_chanr[i][1]) if ch in self.chans])
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
        if self.preferences.uvres:
            return self.preferences.uvres
        else:
            return self.uvres_full


    @property
    def npol(self):
        """ Number of polarization products selected. Cached. """

        if not hasattr(self, '_npol'):
            self._npol = len(self.pols)

        return self._npol


    @property
    def pols(self):
        """ Polarizations to use based on preference in parameters.selectpol """

        if self.preferences.selectpol == 'auto':
            return [pp for pp in self.metadata.pols_orig if pp[0] == pp[1]]
        elif self.preferences.selectpol == 'all':
            return self.metadata.pols_orig
        else:
            logger.warn('selectpol of {0} not supported'.format(self.preferences.selectpol))


    @property
    def uvres_full(self):
        return np.round(self.metadata.dishdiameter / (3e-1 / self.freq.min()) / 2).astype('int')


    @property
    def npixx_full(self):
        """ Finds optimal uv/image pixel extent in powers of 2 and 3"""

        urange_orig, vrange_orig = self.metadata.uvrange_orig
        urange = urange_orig * (self.freq.max() / self.metadata.freq_orig[0])
        powers = np.fromfunction(lambda i, j: 2**i*3**j, (14, 10), dtype='int')
        rangex = np.round(self.preferences.uvoversample*urange).astype('int')
        largerx = np.where(powers - rangex / self.uvres_full > 0,
                           powers, powers[-1, -1])
        p2x, p3x = np.where(largerx == largerx.min())
        return (2**p2x * 3**p3x)[0]


    @property
    def npixy_full(self):
        """ Finds optimal uv/image pixel extent in powers of 2 and 3"""

        urange_orig, vrange_orig = self.metadata.uvrange_orig
        vrange = vrange_orig * (self.freq.max() / self.metadata.freq_orig[0])
        powers = np.fromfunction(lambda i, j: 2**i*3**j, (14, 10), dtype='int')
        rangey = np.round(self.preferences.uvoversample*vrange).astype('int')
        largery = np.where(powers - rangey / self.uvres_full > 0,
                           powers, powers[-1, -1])
        p2y, p3y = np.where(largery == largery.min())
        return (2**p2y * 3**p3y)[0]


    @property
    def npixx(self):
        """ Number of x pixels in uv/image grid.
        First defined by input parameter set with default to npixx_full
        """

        if self.preferences.npixx:
            return self.preferences.npixx
        else:
            if self.preferences.npix_max:
                npix = min(self.preferences.npix_max, self.npixx_full)
            else:
                npix = self.npixx_full
            return npix


    @property
    def npixy(self):
        """ Number of y pixels in uv/image grid.
        First defined by input parameter set with default to npixy_full
        """
        
        if self.preferences.npixy:
            return self.preferences.npixy
        else:
            if self.preferences.npix_max:
                npix = min(self.preferences.npix_max, self.npixy_full)
            else:
                npix = self.npixy_full
            return npix


    @property
    def fringetime(self):
        """ Estimate largest time span of a "segment".
        A segment is the maximal time span that can be have a single bg fringe subtracted and uv grid definition.
        Max fringe window estimated for 5% amp loss at first null averaged over all baselines. Assumes dec=+90, which is conservative.
        Returns time in seconds that defines good window.
        """

        maxbl = self.uvres*max(self.npixx, self.npixy)/2    # fringe time for image grid extent
        fringetime = 0.5*(24*3600)/(2*np.pi*maxbl/25.)   # max fringe window in seconds
        return fringetime


    @property
    def ants(self):
        return sorted([ant for ant in self.metadata.antids if ant not in self.preferences.excludeants])


    @property
    def nants(self):
        return len(self.ants)


    @property
    def nbl(self):
        return int(self.nants*(self.nants-1)/2)


    @property
    def gainfile(self):
        """ Calibration file (telcal) from preferences or found from ".GN" suffix """
        
        if not self.preferences.gainfile:
            # look for gainfile in workdir
            gainfile = os.path.join(self.metadata.workdir, self.metadata.filename + '.GN')

            if os.path.exists(gainfile):
                logger.info('Autodetected telcal file {0}'.format(gainfile))
        else:
            gainfile = self.preferences.gainfile
                
        return gainfile

    
    @property
    def blarr(self):
        if not hasattr(self, '_blarr'):
            self._blarr = np.array([ [self.ants[i], self.ants[j]] for j in range(self.nants) for i in range(0,j)])

        return self._blarr


    @property
    def nsegments(self):
        if self.preferences.nsegments:
            return self.preferences.nsegments
        else:
            return len(self.segmenttimes)


    @property
    def segmenttimes(self):
        """ List of tuples containing MJD times defining segment start and stop.
        Calculated from parameters.nsegments first.
        Alternately, best times found based on fringe time and memory limit
        """

        if not hasattr(self, '_segmenttimes'):
            if self.preferences.nsegments:
                self._segmenttimes = calc_segment_times(self)
            else:
                find_segment_times(self)

        return self._segmenttimes


    def get_segmenttime_string(self, segment):
        mid_mjd = self.segmenttimes[segment].mean()
        return qa.time(qa.quantity(mid_mjd,'d'), form="ymd", prec=8)[0]


    def get_uvw_segment(self, segment):
        mjdstr = self.get_segmenttime_string(segment)
        (u, v, w) = util.calc_uvw(datetime=mjdstr, radec=self.metadata.radec, antpos=self.metadata.antpos, telescope=self.metadata.telescope)
        u = u * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
        v = v * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
        w = w * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)

        return u.astype('float32'), v.astype('float32'), w.astype('float32')


    @property
    def readints(self):
        """ Number of integrations read per segment. 
        Defines shape of numpy array for visibilities.

        TODO: Need to support self.preferences.read_tdownsample
        """

        totaltimeread = 24*3600*(self.segmenttimes[:, 1] - self.segmenttimes[:, 0]).sum()            # not guaranteed to be the same for each segment
        return np.round(totaltimeread / (self.metadata.inttime*self.nsegments)).astype(int)


    @property
    def nints(self):
        if self.metadata.nints:
            return self.metadata.nints
        else:
            return self.fringetime/self.metadata.inttime


    @property
    def t_segment(self):
        totaltimeread = 24*3600*(self.segmenttimes[:, 1] - self.segmenttimes[:, 0]).sum()            # not guaranteed to be the same for each segment
        return totaltimeread/self.nsegments


    @property
    def datashape(self):
        return (self.readints/self.preferences.read_tdownsample, self.nbl, self.nchan/self.preferences.read_fdownsample, self.npol)


    @property
    def datasize(self):
        return long(self.readints*self.nbl*self.nchan*self.npol/(self.preferences.read_tdownsample*self.preferences.read_fdownsample))


    @property
    def nfalse(self):
        """ Calculate the number of thermal-noise false positives per segment.
        """

        dtfactor = np.sum([1./i for i in self.dtarr])    # assumes dedisperse-all algorithm
        ntrials = self.readints * dtfactor * len(self.dmarr) * self.npixx * self.npixy
        qfrac = 1 - (erf(self.preferences.sigma_image1/np.sqrt(2)) + 1)/2.
        nfalse = int(qfrac*ntrials)
        return nfalse

    @property
    def features(self):
        """ Given searchtype, return features to be extracted in initial analysis """

        if self.preferences.searchtype == 'image1':
            return ('snr1', 'immax1', 'l1', 'm1')
        elif self.preferences.searchtype == 'image1stats':
            return ('snr1', 'immax1', 'l1', 'm1', 'specstd', 'specskew', 'speckurtosis', 'imskew', 'imkurtosis')  # note: spec statistics are all or nothing.


    @property
    def vismem(self):
        return self.memory_footprint(visonly=True)


    def memory_footprint(self, visonly=False, limit=False):
        """ Calculates the memory required to store visibilities and make images.
        limit=True returns a the minimum memory configuration
        Returns tuple of (vismem, immem) in units of GB.
        """

        toGB = 8/1024.**3   # number of complex64s to GB

        # limit defined for dm sweep time and max nchunk/nthread ratio
        if limit:
            readints_scale = (self.t_overlap/self.metadata.inttime)/self.readints
            vismem = self.datasize * readints_scale * toGB

            nchunk_scale = max(self.dtarr)/min(self.dtarr)
            immem = self.preferences.nthread * (self.readints/(self.preferences.nthread*nchunk_scale) * self.npixx * self.npixy) * toGB
        else:
            vismem = self.datasize * toGB
            immem = self.preferences.nthread * (self.readints/self.preferences.nthread * self.npixx * self.npixy) * toGB

        if visonly:
            return vismem
        else:
            return (vismem, immem)


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
        return [key for key in self.__dict__.keys() if key[0] != '_']


def calc_dmarr(state):
    """ Function to calculate the DM values for a given maximum sensitivity loss.
    dm_maxloss is sensitivity loss tolerated by dm bin width. dm_pulsewidth is assumed pulse width in microsec.
    """

    dm_maxloss = state.preferences.dm_maxloss
    dm_pulsewidth = state.preferences.dm_pulsewidth
    mindm = state.preferences.mindm
    maxdm = state.preferences.maxdm

    # parameters
    tsamp = state.metadata.inttime*1e6  # in microsec
    k = 8.3
    freq = state.freq.mean()  # central (mean) frequency in GHz
    bw = 1e3*(state.freq[-1] - state.freq[0])
    ch = 1e3*(state.freq[1] - state.freq[0])  # channel width in MHz

    # width functions and loss factor
    dt0 = lambda dm: np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2)
    dt1 = lambda dm, ddm: np.sqrt(dm_pulsewidth**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2 + ((k*ddm*bw)/(freq**3.))**2)
    loss = lambda dm, ddm: 1 - np.sqrt(dt0(dm)/dt1(dm,ddm))
    loss_cordes = lambda ddm, dfreq, dm_pulsewidth, freq: 1 - (np.sqrt(np.pi) / (2 * 6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))) * erf(6.91e-3 * ddm * dfreq / (dm_pulsewidth*freq**3))  # not quite right for underresolved pulses

    if maxdm == 0:
        return [0]
    else:
        # iterate over dmgrid to find optimal dm values. go higher than maxdm to be sure final list includes full range.
        dmgrid = np.arange(mindm, maxdm, 0.05)
        dmgrid_final = [dmgrid[0]]
        for i in range(len(dmgrid)):
            ddm = (dmgrid[i] - dmgrid_final[-1])/2.
            ll = loss(dmgrid[i],ddm)
            if ll > dm_maxloss:
                dmgrid_final.append(dmgrid[i])

    return dmgrid_final


def calc_segment_times(state, nsegments=0):
    """ Helper function for set_pipeline to define segmenttimes list, given nsegments definition
    Can optionally overload state.nsegments to calculate new times
    """

    if not nsegments:
        nsegments = state.nsegments
    # this casts to int (flooring) to avoid 0.5 int rounding issue. 
    stopdts = np.linspace(state.t_overlap/state.metadata.inttime, state.nints, state.nsegments+1)[1:]   # nseg+1 assures that at least one seg made
    startdts = np.concatenate( ([0], stopdts[:-1]-state.t_overlap/state.metadata.inttime) )
            
    segmenttimes = []
    for (startdt, stopdt) in zip(state.metadata.inttime*startdts, state.metadata.inttime*stopdts):
        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(state.metadata.starttime_mjd+startdt/(24*3600), 'd'), 
                                                   form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(state.metadata.starttime_mjd+stopdt/(24*3600), 'd'),
                                                  form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        segmenttimes.append((starttime, stoptime))

    return np.array(segmenttimes)


def find_segment_times(state):
    """ Iterates to optimal segment time list, given memory and fringe time limits.
    Segment sizes bounded by fringe time and memory limit,
    Solution found by iterating from fringe time to memory size that fits.
    """

    # initialize at fringe time limit. nsegments must be between 1 and state.nints
    scale_nsegments = 1.
    nsegments = max(1, min(state.nints, int(scale_nsegments*state.metadata.inttime*state.nints/(state.fringetime-state.t_overlap))))

    # calculate memory limit to stop iteration
    (vismem0, immem0) = state.memory_footprint(limit=True)
    assert vismem0+immem0 < state.parameter.memory_limit, 'memory_limit of {0} is smaller than best solution of {1}. Try forcing nsegments/nchunk larger than {2}/{3} or reducing maxdm/npix'.format(state.parameter.memory_limit, vismem0+immem0, state.nsegments, max(state.dtarr)/min(state.dtarr))

    (vismem, immem) = state.memory_footprint()
    if vismem+immem > state.parameter.memory_limit:
        logger.info('Over memory limit of {4} when reading {0} segments with {1} chunks ({2}/{3} GB for visibilities/imaging). Searching for solution down to {5}/{6} GB...'.format(state.nsegments, state.parameter.nchunk, vismem, immem, state.parameter.memory_limit, vismem0, immem0))

        while vismem+immem > state.parameter.memory_limit:
            (vismem, immem) = state.memory_footprint()
            logger.debug('Using {0} segments with {1} chunks ({2}/{3} GB for visibilities/imaging). Searching for better solution...'.format(state.parameter.nchunk, vismem, immem, state.parameter.memory_limit))

            scale_nsegments *= (vismem+immem)/float(state.parameter.memory_limit)
            nsegments = max(1, min(state.nints, int(scale_nsegments*state.metadata.inttime*state.nints/(fringetime-state.t_overlap))))  # at least 1, at most nints
            state._segment_times = calc_segment_times(state, nsegments=nsegments)

            (vismem, immem) = state.memory_footprint()
            while vismem+immem > state.parameter.memory_limit:
                logger.debug('Doubling nchunk from %d to fit in %d GB memory limit.' % (state.parameter.nchunk, state.parameter.memory_limit))
                self.parameter.nchunk = 2*self.parameter.nchunk
                (vismem, immem) = state.memory_footprint()
                if self.parameter.nchunk >= max(self.dtarr)/min(self.dtarr)*self.nthread: # limit nchunk/nthread to at most the range in dt
                    self.nchunk = self.nthread
                    break

                (vismem, immem) = state.memory_footprint()

    # final set up of memory
    state._segment_times = calc_segment_times(state)
    (vismem, immem) = state.memory_footprint()


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



