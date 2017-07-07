from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

import json, attr, os, yaml
from rfpipe import source, util, preferences, metadata
import numpy as np
from scipy.special import erf
from collections import OrderedDict
from astropy import time

import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
logger.info('Using pwkit casa')


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
    - nsegment or dmarr + memory_limit + metadata => segmenttimes
    - dmarr or dm_parameters + metadata => dmarr
    - uvoversample + npix_max + metadata => npixx, npixy
    """

    def __init__(self, config=None, sdmfile=None, sdmscan=None, inprefs={}, inmeta={}, preffile=None, name=None, version=1, showsummary=True):
        """ Initialize preference attributes with text file, preffile.
        name can select preference set from within yaml file.
        preferences are overloaded with inprefs.
        Versions define functions that derive state from preferences and metadata

        Metadata source can be:
        1) Config object is (expected to be) like EVLA_config object prototyped for pulsar work by Paul.
        2) sdmfile and sdmscan are as in rtpipe.

        inmeta is a dict with key-value pairs to overload metadata (e.g., to mock metadata from a simulation)
        """

        self.version = version
        self.config = config
        self.sdmfile = sdmfile
        self.sdmscan = sdmscan

        if isinstance(inprefs, dict):
            # get pipeline preferences
            prefs = preferences.parsepreffile(preffile)  # returns empty dict for paramfile=None

            # optionally overload preferences
            for key in inprefs:
                prefs[key] = inprefs[key]
                
            self.prefs = preferences.Preferences(**prefs)
        elif isinstance(inprefs, preferences.Preferences):
            self.prefs = inprefs
        else:
            logger.warn('inprefs should be either a dictionary or preferences.Preferences object')

        logger.parent.setLevel(getattr(logging, self.prefs.loglevel))

        if isinstance(inmeta, dict):
            # get metadata
            if (self.sdmfile and self.sdmscan) and not self.config:
                meta = metadata.sdm_metadata(sdmfile, sdmscan)
            elif self.config and not (self.sdmfile or self.sdmscan):
                meta = metadata.config_metadata(config)
            else:
                meta = {}

            # optionally overload metadata
            for key in inmeta:
                meta[key] = inmeta[key]

            self.metadata = metadata.Metadata(**meta)
        elif isinstance(inmeta, metadata.Metadata):
            self.metadata = inmeta
        else:
            logger.warn('inmeta should be either a dictionary or metadata.Metadata object')

        if showsummary:
            self.summarize()


    def summarize(self):
        """ Print overall state, if metadata set """

        if self.metadata.atdefaults():
            logger.info('Metadata not set. Cannot calculate properties')
        else:
            logger.info('Metadata summary:')
            logger.info('\t Working directory and fileroot: {0}, {1}'.format(self.metadata.workdir, self.fileroot))
            logger.info('\t Using scan {0}, source {1}'.format(int(self.metadata.scan), self.metadata.source))
            logger.info('\t nants, nbl: {0}, {1}'.format(self.nants, self.nbl))
            logger.info('\t nchan, nspw: {0}, {1}'.format(self.nchan, self.nspw))
            logger.info('\t Freq range: {0:.3f} -- {1:.3f}'.format(self.freq.min(), self.freq.max()))
            logger.info('\t Scan has {0} ints ({0:.1f} s) and inttime {1:.3f} s'.format(self.nints, self.nints*self.metadata.inttime, self.metadata.inttime))
            logger.info('\t {0} polarizations: {1}'.format(self.metadata.npol_orig, self.metadata.pols_orig))
            logger.info('\t Ideal uvgrid npix=({0}, {1}) and res={2} (oversample {3:.1f})'.format(self.npixx_full, self.npixy_full, self.uvres_full, self.prefs.uvoversample))

            logger.info('Pipeline summary:')
            logger.info('\t Products saved with {0}. telcal calibration with {1}.'.format(self.fileroot, os.path.basename(self.gainfile)))
            logger.info('\t Using {0} segment{1} of {2} ints ({3:.1f} s) with overlap of {4:.1f} s'.format(self.nsegment, "s"[not self.nsegment-1:], self.readints, self.t_segment, self.t_overlap))
            if self.t_overlap > self.t_segment/3. and self.t_overlap < self.t_segment:
                logger.info('\t\t Lots of segments needed, since Max DM sweep ({0:.1f} s) close to segment size ({1:.1f} s)'.format(self.t_overlap, self.t_segment))
            elif self.t_overlap >= self.t_segment:
                logger.warn('\t\t Max DM sweep ({0:.1f} s) is larger than segment size ({1:.1f} s). Pipeline will fail!'.format(self.t_overlap, self.t_segment))

            if self.metadata.inttime > self.fringetime:
                logger.warn('\t\t Integration time larger than fringe timescale ({0} > {1}). Mean visibility subtraction will not work well.'.format(self.metadata.inttime, self.fringetime))

            logger.info('\t Downsampling in time/freq by {0}/{1}.'.format(self.prefs.read_tdownsample, self.prefs.read_fdownsample))
            logger.info('\t Excluding ants {0}'.format(self.prefs.excludeants))
            logger.info('\t Using pols {0}'.format(self.pols))
            logger.info('')
            
            logger.info('\t Search with {0} and threshold {1}.'.format(self.prefs.searchtype, self.prefs.sigma_image1))
            logger.info('\t Using {0} DMs from {1} to {2} and dts {3}.'.format(len(self.dmarr), min(self.dmarr), max(self.dmarr), self.dtarr))
            logger.info('\t Using uvgrid npix=({0}, {1}) and res={2}.'.format(self.npixx, self.npixy, self.uvres))
            logger.info('\t Expect {0} thermal false positives per segment.'.format(self.nfalse))
            
            logger.info('')
            logger.info('\t Visibility memory usage is {0} GB/segment'.format(self.vismem))
#            logger.info('\t Imaging in {0} chunk{1} using max of {2} GB/segment'.format(self.nchunk, "s"[not self.nsegment-1:], immem))
#            logger.info('\t Grand total memory usage: {0} GB/segment'.format(vismem + immem))


    def clearcache(self):
        cached = ['_dmarr', '_t_overlap', '_dmshifts', '_npol', '_blarr', '_segmenttimes']
        for obj in cached:
            try:
                delattr(self, obj)
            except AttributeError:
                pass


    @property
    def fileroot(self):
        # **TODO: update for sdm or config sources

        if self.prefs.fileroot:
            return self.prefs.fileroot
        else:
            return os.path.basename(self.metadata.filename)


    @property
    def dmarr(self):
        if not hasattr(self, '_dmarr'):
            if self.prefs.dmarr:
                self._dmarr = self.prefs.dmarr
            else:
                self._dmarr = calc_dmarr(self)

        return self._dmarr


    @property
    def dtarr(self):
        if self.prefs.dtarr:
            return self.prefs.dtarr
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
        """ List of channel indices to use. Drawn from preferences, with backup to take all defined in metadata. """
        if self.prefs.chans:
            return self.prefs.chans
        else:
            return range(sum(self.metadata.spw_nchan))


    @property
    def nchan(self):
        return len(self.chans)


    @property
    def dmshifts(self):
        """ Calculate max DM delay in units of integrations for each dm trial. Gets cached.
        """

        if not hasattr(self, '_dmshifts'):
            self._dmshifts = [util.calc_delay(self.freq, self.freq.max(), dm, self.metadata.inttime).max() for dm in self.dmarr]
        return self._dmshifts
        

    @property
    def t_overlap(self):
        """ Max DM delay in seconds. Gets cached. """

        if not hasattr(self, '_t_overlap'):
            self._t_overlap = max(self.dmshifts)*self.metadata.inttime
        return self._t_overlap


    @property
    def spw(self):
        if self.prefs.spw:
            return self.prefs.spw
        else:
            return self.metadata.spw_orig


    @property
    def nspw(self):
        return len(self.spw)


#    @property
#    def spw_nchan_select(self):
#        return [len([ch for ch in range(self.metadata.spw_chanr[i][0], self.metadata.spw_chanr[i][1]) if ch in self.chans])
#                for i in range(len(self.metadata.spw_chanr))]


#    @property
#    def spw_chanr_select(self):
#        chanr_select = []
#        i0 = 0
#        for nch in self.spw_nchan_select:
#            chanr_select.append((i0, i0+nch))
#            i0 += nch
#
#        return chanr_select


    @property
    def uvres(self):
        if self.prefs.uvres:
            return self.prefs.uvres
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
        """ Polarizations to use based on preference in prefs.selectpol """

        if self.prefs.selectpol == 'auto':
            return [pp for pp in self.metadata.pols_orig if pp[0] == pp[-1]]
        elif self.prefs.selectpol == 'cross':
            return [pp for pp in self.metadata.pols_orig if pp[0] != pp[-1]]
        elif self.prefs.selectpol == 'all':
            return self.metadata.pols_orig
        else:
            logger.warn('selectpol of {0} not supported'.format(self.prefs.selectpol))


    @property
    def uvres_full(self):
        return np.round(self.metadata.dishdiameter / (3e-1 / self.freq.min()) / 2).astype('int')


    @property
    def npixx_full(self):
        """ Finds optimal uv/image pixel extent in powers of 2 and 3"""

        urange_orig, vrange_orig = self.metadata.uvrange_orig
        urange = urange_orig * (self.freq.max() / self.metadata.freq_orig[0])
        powers = np.fromfunction(lambda i, j: 2**i*3**j, (14, 10), dtype='int')
        rangex = np.round(self.prefs.uvoversample*urange).astype('int')
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
        rangey = np.round(self.prefs.uvoversample*vrange).astype('int')
        largery = np.where(powers - rangey / self.uvres_full > 0,
                           powers, powers[-1, -1])
        p2y, p3y = np.where(largery == largery.min())
        return (2**p2y * 3**p3y)[0]


    @property
    def npixx(self):
        """ Number of x pixels in uv/image grid.
        First defined by input preference set with default to npixx_full
        """

        if self.prefs.npixx:
            return self.prefs.npixx
        else:
            if self.prefs.npix_max:
                npix = min(self.prefs.npix_max, self.npixx_full)
            else:
                npix = self.npixx_full
            return npix


    @property
    def npixy(self):
        """ Number of y pixels in uv/image grid.
        First defined by input preference set with default to npixy_full
        """
        
        if self.prefs.npixy:
            return self.prefs.npixy
        else:
            if self.prefs.npix_max:
                npix = min(self.prefs.npix_max, self.npixy_full)
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
        return sorted([ant for ant in self.metadata.antids if ant not in self.prefs.excludeants])


    @property
    def nants(self):
        return len(self.ants)


    @property
    def nbl(self):
        return int(self.nants*(self.nants-1)/2)


    @property
    def gainfile(self):
        """ Calibration file (telcal) from preferences or found from ".GN" suffix """
        
        if not self.prefs.gainfile:
            # look for gainfile in workdir
            gainfile = os.path.join(self.metadata.workdir, self.metadata.filename, '.GN')

            if os.path.exists(gainfile):
                logger.info('Autodetected telcal file {0}'.format(gainfile))
        else:
            gainfile = self.prefs.gainfile
                
        return gainfile

    
    @property
    def blarr(self):
        if not hasattr(self, '_blarr'):
            self._blarr = np.array([ [int(self.ants[i].lstrip('ea')), int(self.ants[j].lstrip('ea'))] for j in range(self.nants) for i in range(0,j)])

        return self._blarr


    @property
    def blarr_names(self):
        return np.array([ [self.ants[i], self.ants[j]] for j in range(self.nants) for i in range(0,j)])


    @property
    def nsegment(self):
        if self.prefs.nsegment:
            return self.prefs.nsegment
        else:
            return len(self.segmenttimes)


    @property
    def segmenttimes(self):
        """ List of tuples containing MJD times defining segment start and stop.
        Calculated from prefs.nsegment first.
        Alternately, best times found based on fringe time and memory limit
        """

        if not hasattr(self, '_segmenttimes'):
            if len(self.prefs.segmenttimes):
                self._segmenttimes = self.prefs.segmenttimes
            elif self.prefs.nsegment:
                self._segmenttimes = calc_segment_times(self)
            else:
                find_segment_times(self)

        return self._segmenttimes


    def pixtolm(self, pix):
        """ Helper function to calculate (l,m) coords of given pixel.
        Example: st.pixtolm(np.where(im == im.max()))
        """

        assert len(pix) == 2

        peaky, peakx = pix
        if isinstance(peaky, np.ndarray) and len(peaky) == 1: # np.where output
            peaky = peaky[0]
            peakx = peakx[0]

        l1 = (self.npixx/2. - peakx)/(self.npixx*self.uvres)
        m1 = (self.npixy/2. - peaky)/(self.npixy*self.uvres)

        # ** this is flipped relative to rtpipe, but this seems right to me.
        return l1, m1


    def get_segmenttime_string(self, segment):
        mid_mjd = self.segmenttimes[segment].mean()
        return qa.time(qa.quantity(mid_mjd, 'd'), form='ymd', prec=8)[0]


    def get_uvw_segment(self, segment):
        """ Returns uvw in units of baselines for a given segment.
        Tuple of u, v, w given with each a numpy array of (nbl, nchan) shape.
        """

        mjdstr = self.get_segmenttime_string(segment)
        (u, v, w) = util.calc_uvw(datetime=mjdstr, radec=self.metadata.radec, antpos=self.metadata.antpos, telescope=self.metadata.telescope)
#        u = u * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
#        v = v * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
#        w = w * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
        u = np.outer(u, self.freq * (1e9/3e8) * (-1))
        v = np.outer(v, self.freq * (1e9/3e8) * (-1))
        w = np.outer(w, self.freq * (1e9/3e8) * (-1))

        return u.astype('float32'), v.astype('float32'), w.astype('float32')


    @property
    def nints(self):
        if self.metadata.nints:
            return self.metadata.nints  # if nints known (e.g., from sdm) or set (e.g., forced during vys reading)
        elif self.prefs.nsegment:  # if overloading segment time calculation
            return int(round((self.nsegment*self.fringetime - self.t_overlap*(self.nsegment-1))/self.metadata.inttime))  # else this is open ended
        else:
            raise ValueError, "Number of integrations in scan is not known or cannot be inferred. Set metadata.nints of prefs.nsegment."


    @property
    def t_segment(self):
        """ Time read per segment in seconds """

        if self.metadata.nints:
            totaltimeread = 24*3600*(self.segmenttimes[:, 1] - self.segmenttimes[:, 0]).sum()            # not guaranteed to be the same for each segment
            return totaltimeread/self.nsegment
        elif self.prefs.nsegment:
            return self.nints*self.metadata.inttime/self.nsegment
        else:
            raise ValueError, "Number of integrations in scan is not known or cannot be inferred. Set metadata.nints of prefs.nsegment."


    @property
    def readints(self):
        """ Number of integrations read per segment. 
        Defines shape of numpy array for visibilities.
        """

        return int(round(self.t_segment / self.metadata.inttime))


    @property
    def datashape(self):
        return (self.readints/self.prefs.read_tdownsample, self.nbl, self.nchan/self.prefs.read_fdownsample, self.npol)


    @property
    def datasize(self):
        return long(self.readints*self.nbl*self.nchan*self.npol/(self.prefs.read_tdownsample*self.prefs.read_fdownsample))


    @property
    def nfalse(self):
        """ Calculate the number of thermal-noise false positives per segment.
        """

        dtfactor = np.sum([1./i for i in self.dtarr])    # assumes dedisperse-all algorithm
        ntrials = self.readints * dtfactor * len(self.dmarr) * self.npixx * self.npixy
        qfrac = 1 - (erf(self.prefs.sigma_image1/np.sqrt(2)) + 1)/2.
        nfalse = int(qfrac*ntrials)
        return nfalse


    @property
    def search_dimensions(self):
        """ Define dimensions searched for a given piece of data. 
        Actual algorithm defined in pipeline iteration.
        """

        return ('segment', 'integration', 'dmind', 'dtind', 'beamnum')


    @property
    def features(self):
        """ Given searchtype, return features to be extracted in initial analysis """

        if self.prefs.searchtype == 'image1':
            return ('snr1', 'immax1', 'l1', 'm1')
        elif self.prefs.searchtype == 'image1stats':
            return ('snr1', 'immax1', 'l1', 'm1', 'specstd', 'specskew', 'speckurtosis', 'imskew', 'imkurtosis')  # note: spec statistics are all or nothing.


    @property
    def candsfile(self):
        """ File name to write candidates to """

        if self.prefs.candsfile:
            return self.prefs.candsfile
        else:
            return os.path.join(self.metadata.workdir, 'cands_' + self.fileroot + '.pkl')


    @property
    def vismem(self):
        """ Memory required to store read data (in GB)
        """

        toGB = 8/1024.**3   # number of complex64s to GB

        return self.datasize * toGB


    @property
    def vismem_limit(self):
        """ Memory required to store read data (in GB)
        Limit defined for time range equal to the overlap time between segments.
        """

        toGB = 8/1024.**3   # number of complex64s to GB
        return toGB*long(self.t_overlap/self.metadata.inttime*self.nbl*self.nchan*self.npol/(self.prefs.read_tdownsample*self.prefs.read_fdownsample))


    @property
    def immem(self):
        """ Memory required to create all images in a chunk of read integrations
        """

        toGB = 8/1024.**3   # number of complex64s to GB
        immem = self.prefs.nthread * (self.readints/self.prefs.nthread * self.npixx * self.npixy) * toGB

        return immem


    @property
    def immem_limit(self):
        """ Memory required to create all images in a chunk of read integrations
        Limit defined for 
        """

        toGB = 8/1024.**3   # number of complex64s to GB
        nchunk_scale = max(self.dtarr)/min(self.dtarr)
        immem = self.prefs.nthread * ((self.t_overlap/self.metadata.inttime)/(self.prefs.nthread*nchunk_scale) * self.npixx * self.npixy) * toGB

        return immem

# **not sure we need this
#
#    @property
#    def reproducekeys(self):
#        """ Minimal set of state keys required to assure that state can reproduce a given candidate. 
#
#        Should be independent of data? Or assumes post-reading of metadata and applying parameter generation functions?
#        """
#
#        # this set is input, version defines functions that transform this to pipeline state
#        return sorted(['sdmfile', 'excludeants', 'read_tdownsample', 'read_fdownsample',
#                       'selectpol', 'timesub', 'dmarr', 'dtarr', 'searchtype',
##                       'features', 'sigma_image1', 'sigma_image2', 'sigma_bisp',   # not sure about this set
#                       'uvres', 'npixx', 'npixy', 'version',
#                       'flaglist', 'gainfile', 'bpfile', 'onlineflags'])

    # should each of the above actually be considered transformation functions? 
    #  so, input is observation name, next step is transform by ignore ants, next step...
    #
    #   nsegment or dmarr + memory_limit + metadata => segmenttimes
    #   dmarr or dm_parameters + metadata => dmarr
    #   uvoversample + npix_max + metadata => npixx, npixy

    # flagantsol should be merged with flaglist

# ** not sure we need this yet.
#
#    @property
#    def hash(self):
#        """ Hash that identifies pipeline state that produces unique set of output products """
#
#        extant_keys = self.keys()
#        if all([kk in self.reproducekeys for kk in extant_keys]):
#            values = [self[key] for key in self.reproducekeys]
#            return hash(json.dumps(repr(values)))  # is repr safe?
#        else:
#            print('Cannot make hash without minimal set defined in reproducekeys property.')
#            return None


    @property
    def defined(self):
        return [key for key in self.__dict__.keys() if key[0] != '_']


def calc_dmarr(state):
    """ Function to calculate the DM values for a given maximum sensitivity loss.
    dm_maxloss is sensitivity loss tolerated by dm bin width. dm_pulsewidth is assumed pulse width in microsec.
    """

    dm_maxloss = state.prefs.dm_maxloss
    dm_pulsewidth = state.prefs.dm_pulsewidth
    mindm = state.prefs.mindm
    maxdm = state.prefs.maxdm

    # parameters
    tsamp = state.metadata.inttime*1e6  # in microsec
    k = 8.3
    freq = state.freq.mean()  # central (mean) frequency in GHz
    bw = 1e3*(state.freq.max() - state.freq.min())  # in MHz
    ch = 1e-6*state.metadata.spw_chansize[0]  # in MHz ** first spw only

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


def calc_segment_times(state, nsegment=0):
    """ Helper function for set_pipeline to define segmenttimes list, given nsegment definition
    Can optionally overload state.nsegment to calculate new times
    ** TODO: why is this slightly off from rtpipe calculation?
    """

    nsegment = nsegment if nsegment else state.nsegment

    # this casts to int (flooring) to avoid 0.5 int rounding issue. 
    stopdts = np.linspace(state.t_overlap/state.metadata.inttime, state.nints, nsegment+1)[1:]   # nseg+1 assures that at least one seg made
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

    **TODO: this is still pretty awkward. Setting nsegment and nints may produce different outcomes.
    """

    assert state.metadata.nints or state.prefs.nsegment, "Can only find segment times if nints or nsegments defined"

    # initialize at fringe time limit. nsegment must be between 1 and state.nints
    scale_nsegment = 1.
    nsegment = max(1, min(state.metadata.nints, int(round(scale_nsegment*state.metadata.inttime*state.metadata.nints/(state.fringetime-state.t_overlap)))))  # at least 1, at most nints
    state._segmenttimes = calc_segment_times(state, nsegment)

    # calculate memory limit to stop iteration
    assert state.immem_limit+state.vismem_limit < state.prefs.memory_limit, 'memory_limit of {0} is smaller than best solution of {1}. Try forcing nsegment/nchunk larger than {2}/{3} or reducing maxdm/npix'.format(state.prefs.memory_limit, state.immem_limit+state.vismem_limit, state.nsegment, max(state.dtarr)/min(state.dtarr))

    if state.vismem+state.immem > state.prefs.memory_limit:
        logger.info('Over memory limit of {4} when reading {0} segments with {1} chunks ({2}/{3} GB for visibilities/imaging). Searching for solution down to {5}/{6} GB...'.format(state.nsegment, state.prefs.nchunk, state.vismem, state.immem, state.prefs.memory_limit, state.vismem_limit, state.immem_limit))

        while state.vismem+state.immem > state.prefs.memory_limit:
            logger.debug('Using {0} segments with {1} chunks ({2}/{3} GB for visibilities/imaging). Searching for better solution...'.format(state.prefs.nchunk, state.vismem, state.immem, state.prefs.memory_limit))

            scale_nsegment *= (state.vismem+state.immem)/float(state.prefs.memory_limit)
            nsegment = max(1, min(state.metadata.nints, int(round(scale_nsegment*state.metadata.inttime*state.metadata.nints/(state.fringetime-state.t_overlap)))))  # at least 1, at most nints
            state._segmenttimes = calc_segment_times(state, nsegment=nsegment)

            while state.vismem+state.immem > state.prefs.memory_limit:
                logger.debug('Doubling nchunk from %d to fit in %d GB memory limit.' % (state.prefs.nchunk, state.prefs.memory_limit))
                state.prefs.nchunk = 2*state.prefs.nchunk
                if state.prefs.nchunk >= max(state.dtarr)/min(state.dtarr)*state.prefs.nthread: # limit nchunk/nthread to at most the range in dt
                    state.prefs.nchunk = state.prefs.nthread
                    break

    # final set up of memory
    state._segmenttimes = calc_segment_times(state)


def state_vystest(wait, nsegment=0, scantime=0, preffile=None, **kwargs):
    """ Create state to read vys data after wait seconds with nsegment segments.
    kwargs passed in as preferences via inpref argument to State.
    """

    try:
        from evla_mcast import scan_config
    except ImportError:
        logger.error('ImportError for evla_mcast. Need this library to consume multicast messages from CBE.')

    meta = {}
    prefs = {}

    # set start time (and fix antids)
    dt = time.TimeDelta(wait, format='sec')
    t0 = (time.Time.now()+dt).mjd
    meta['starttime_mjd'] = t0
    meta['antids'] = ['ea{0}'.format(i) for i in range(1,26)]  # fixed for scan_config test docs

    # read example scan configuration
    config = scan_config.ScanConfig(vci='/home/cbe-master/realfast/soft/evla_mcast/test/data/test_vci.xml',
                                    obs='/home/cbe-master/realfast/soft/evla_mcast/test/data/test_obs.xml',
                                    ant='/home/cbe-master/realfast/soft/evla_mcast/test/data/test_antprop.xml')

    # define amount of time to catch either by nsegment or by setting total time
    if nsegment and not scantime:
        for key in kwargs:
            prefs[key] = kwargs[key]
        prefs['nsegment'] = nsegment
    elif scantime and not nsegment:
        subband0 = config.get_subbands()[0]
        inttime = subband0.hw_time_res  # assumes that vys stream comes after hw integration
        nints = int(scantime/inttime)
        logger.info('Pipeline will be set to catch {0} s of data ({1} integrations)'.format(scantime, nints))
        meta['nints'] = nints

    st = State(config=config, preffile=preffile, inmeta=meta, inprefs=prefs)

    return st
