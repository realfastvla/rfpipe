from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import os
from datetime import date
import numpy as np
from scipy.special import erf
from astropy import time
from rfpipe import util, preferences, metadata, version
import pwkit.environments.casa.util as casautil

import logging
logger = logging.getLogger(__name__)

qa = casautil.tools.quanta()
logger.info('Using pwkit casa')


class State(object):
    """ Defines initial pipeline preferences and methods for calculating state.
    Uses attributes for preferences and metadata.
    State properties used for derived quantities that depend on
    metadata+preferences.

    Scheme:
    1) initial, minimal state defines either parameters for later use or fixes
        final state
    2) read metadata from observation for given scan (if final value not
        yet set)
    3) run functions to set state (sets hashable state)
    4) may also set convenience attibutes
    5) run data processing for given segment

    these should be modified to change state based on input
    - nsegment or dmarr + memory_limit + metadata => segmenttimes
    - dmarr or dm_parameters + metadata => dmarr
    - uvoversample + npix_max + metadata => npixx, npixy
    """

    def __init__(self, config=None, sdmfile=None, sdmscan=None, inprefs={},
                 inmeta={}, preffile=None, name=None, showsummary=True):
        """ Initialize preference attributes with text file, preffile.
        name can select preference set from within yaml file.
        preferences are overloaded with inprefs.

        Metadata source can be either:
        1) Config object is a scan_config object (see evla_mcast library) or
        2) sdmfile and sdmscan.

        inmeta is a dict with key-value pairs to overload metadata (e.g., to
        mock metadata from a simulation)
        """

        self.config = config
        self.sdmscan = sdmscan
        if sdmfile:
            sdmfile = sdmfile.rstrip('/')
        self.sdmfile = sdmfile

        if isinstance(inprefs, dict):
            # get pipeline preferences as dict
            prefs = preferences.parsepreffile(preffile)

            # optionally overload preferences
            for key in inprefs:
                prefs[key] = inprefs[key]

            for key in prefs:
                logger.debug(key, prefs[key], type(prefs[key]))

            self.prefs = preferences.Preferences(**prefs)
        elif isinstance(inprefs, preferences.Preferences):
            self.prefs = inprefs
        else:
            logger.warn('inprefs should be either a dictionary or \
                         preferences.Preferences object')

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

            for key in meta:
                logger.debug(key, meta[key], type(meta[key]))

            self.metadata = metadata.Metadata(**meta)
        elif isinstance(inmeta, metadata.Metadata):
            self.metadata = inmeta
        else:
            logger.warn('inmeta should be either a dictionary or '
                         'metadata.Metadata object')

        if showsummary:
            self.summarize()

    def summarize(self):
        """ Print summary of pipeline state """

        if self.metadata.atdefaults():
            logger.info('Metadata not set. Cannot calculate properties')
        else:
            logger.info('Metadata summary:')
            logger.info('\t Working directory and fileroot: {0}, {1}'
                        .format(self.metadata.workdir, self.fileroot))
            logger.info('\t Using scan {0}, source {1}'
                        .format(int(self.metadata.scan), self.metadata.source))
            logger.info('\t nants, nbl: {0}, {1}'.format(self.nants, self.nbl))
            logger.info('\t nchan, nspw: {0}, {1}'
                        .format(self.nchan, self.nspw))

            spworder = np.argsort(self.metadata.spw_reffreq)
            if np.any(spworder != np.sort(spworder)):
                logger.warn('Sorting spw frequencies to increasing order from order {0}'
                            .format(spworder))

            logger.info('\t Freq range: {0:.3f} -- {1:.3f}'
                        .format(self.freq.min(), self.freq.max()))
            logger.info('\t Scan has {0} ints ({1:.1f} s) and inttime {2:.3f} s'
                        .format(self.nints, self.nints*self.metadata.inttime,
                                self.metadata.inttime))
            logger.info('\t {0} polarizations: {1}'
                        .format(self.metadata.npol_orig,
                                self.metadata.pols_orig))
            logger.info('\t Ideal uvgrid npix=({0}, {1}) and res={2} (oversample {3:.1f})'
                        .format(self.npixx_full, self.npixy_full,
                                self.uvres_full, self.prefs.uvoversample))

            logger.info('Pipeline summary:')
            if os.path.exists(self.gainfile):
                logger.info('Autodetected telcal file {0}'.format(self.gainfile))
            else:
                logger.warn('telcal file not found at {0}'.format(self.gainfile))
            logger.info('\t Products saved with {0}. telcal calibration with {1}.'
                        .format(self.fileroot,
                                os.path.basename(self.gainfile)))

            logger.info('\t Using {0} segment{1} of {2} ints ({3:.1f} s) with '
                        'overlap of {4:.1f} s'
                        .format(self.nsegment, "s"[not self.nsegment-1:],
                                self.readints, self.t_segment, self.t_overlap))
            if ((self.t_overlap > self.t_segment/3.)
               and (self.t_overlap < self.t_segment)):
                logger.info('\t\t Lots of segments needed, since Max DM sweep '
                            '({0:.1f} s) close to segment size ({1:.1f} s)'
                            .format(self.t_overlap, self.t_segment))
            elif self.t_overlap >= self.t_segment:
                logger.warn('\t\t Max DM sweep ({0:.1f} s) is larger than '
                            'segment size ({1:.1f} s). Pipeline will fail!'
                            .format(self.t_overlap, self.t_segment))

            if self.inttime > self.fringetime_orig:
                logger.warn('\t\t Integration time larger than fringe '
                            'timescale ({0} > {1}). Mean visibility '
                            'subtraction will not work well.'
                            .format(self.inttime, self.fringetime))

            logger.info('\t Downsampling in time/freq by {0}/{1}.'
                        .format(self.prefs.read_tdownsample,
                                self.prefs.read_fdownsample))
            logger.info('\t Excluding ants {0}'.format(self.prefs.excludeants))
            logger.info('\t Using pols {0}'.format(self.pols))
            logger.info('')

            logger.info('\t Search with {0} and threshold {1}.'
                        .format(self.prefs.searchtype,
                                self.prefs.sigma_image1))
            logger.info('\t Using {0} DMs from {1} to {2} and dts {3}.'
                        .format(len(self.dmarr), min(self.dmarr),
                                max(self.dmarr), self.dtarr))
            logger.info('\t Using uvgrid npix=({0}, {1}) and res={2}.'
                        .format(self.npixx, self.npixy, self.uvres))
            logger.info('\t Expect {0} thermal false positives per segment.'
                        .format(self.nfalse))

            logger.info('')
            logger.info('\t Visibility memory usage is {0} GB/segment'
                        .format(self.vismem))
#            logger.info('\t Imaging in {0} chunk{1} using max of {2} GB/segment'.format(self.nchunk, "s"[not self.nsegment-1:], immem))
#            logger.info('\t Grand total memory usage: {0} GB/segment'.format(vismem + immem))

    def clearcache(self):
        cached = ['_dmarr', '_t_overlap', '_dmshifts', '_npol', '_blarr',
                  '_segmenttimes', '_npixx_full', '_npixy_full']
        for obj in cached:
            try:
                delattr(self, obj)
            except AttributeError:
                pass

    @property
    def version(self):
        if self.prefs.rfpipe_version:
            return self.prefs.rfpipe_version
        else:
            return version.__version__

    @property
    def fileroot(self):
        # **TODO: update for sdm or config sources

        if self.prefs.fileroot:
            return self.prefs.fileroot
        else:
            return os.path.basename(self.metadata.filename).rstrip('/')

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
        """ Frequencies for each channel in increasing order.
        Metadata may be out of order, but state/data reading order is sorted.
        """

        # TODO: need to support spw selection and downsampling, e.g.:
        #    if spectralwindow[ii] in d['spw']:
        #    np.array([np.mean(spwch[i:i+d['read_fdownsample']]) for i in range(0, len(spwch), d['read_fdownsample'])], dtype='float32') / 1e9

        return np.sort(self.metadata.freq_orig)[self.chans]

    @property
    def chans(self):
        """ List of channel indices to use. Drawn from preferences,
        with backup to take all defined in metadata.
        """

        # TODO: support for frequency downsampling

        if self.prefs.chans:
            return self.prefs.chans
        else:
            return range(sum(self.metadata.spw_nchan))

    @property
    def inttime(self):
        """ Integration time
        """

        # TODO: suppot for time downsampling

        return self.metadata.inttime

    @property
    def nchan(self):
        return len(self.chans)

    @property
    def dmshifts(self):
        """ Calculate max DM delay in units of integrations for each dm trial.
        Gets cached.
        """

        if not hasattr(self, '_dmshifts'):
            self._dmshifts = [util.calc_delay(self.freq, self.freq.max(), dm,
                              self.inttime).max()
                              for dm in self.dmarr]
        return self._dmshifts

    @property
    def t_overlap(self):
        """ Max DM delay in seconds that is fixed to int mult of integration time.
        Gets cached. """

        if not hasattr(self, '_t_overlap'):
            self._t_overlap = max(self.dmshifts)*self.inttime
        return self._t_overlap

    @property
    def spw(self):
        """ Spectral windows used.
        ** TODO: update for proper naming "basband"+"swindex"
        """

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
            logger.warn('selectpol of {0} not supported'
                        .format(self.prefs.selectpol))

    @property
    def uvres_full(self):
        return int(round(self.metadata.dishdiameter / (3e-1
                     / self.freq.min()) / 2))

    @property
    def npixx_full(self):
        """ Finds optimal uv/image pixel extent in powers of 2 and 3"""

        if not hasattr(self, '_npixx_full'):
            urange_orig, vrange_orig = self.metadata.uvrange_orig
            urange = urange_orig * (self.freq.max()
                                    / self.metadata.freq_orig.min())
            powers = np.fromfunction(lambda i, j: 2**i*3**j, (14, 10),
                                     dtype='int')
            rangex = int(round(self.prefs.uvoversample*urange))
            largerx = np.where(powers - rangex / self.uvres_full > 0,
                               powers, powers[-1, -1])
            p2x, p3x = np.where(largerx == largerx.min())
            self._npixx_full = (2**p2x * 3**p3x)[0]

        return self._npixx_full

    @property
    def npixy_full(self):
        """ Finds optimal uv/image pixel extent in powers of 2 and 3"""

        if not hasattr(self, '_npixy_full'):
            urange_orig, vrange_orig = self.metadata.uvrange_orig
            vrange = vrange_orig * (self.freq.max()
                                    / self.metadata.freq_orig.min())
            powers = np.fromfunction(lambda i, j: 2**i*3**j, (14, 10),
                                     dtype='int')
            rangey = int(round(self.prefs.uvoversample*vrange))
            largery = np.where(powers - rangey / self.uvres_full > 0,
                               powers, powers[-1, -1])
            p2y, p3y = np.where(largery == largery.min())
            self._npixy_full = (2**p2y * 3**p3y)[0]

        return self._npixy_full

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
    def beamsize_deg(self):
        """ Takes gridding spec to estimate beam size in degrees
        ** TODO: check for accuracy
        """

        return (np.degrees(1/(self.npixx*self.uvres/2)),
                np.degrees(1/(self.npixy*self.uvres/2)))

    @property
    def fringetime_orig(self):
        """ Estimate largest time span of a "segment".
        A segment is the maximal time span that can be have a single bg fringe
        subtracted and uv grid definition.  Max fringe window estimated for
        5% amp loss at first null averaged over all baselines. Assumes dec=+90,
        which is conservative. Also can be optimized for memory/compute limits.
        Returns time in seconds that defines good window.
        """

        # max baseline for imaging parameters
        maxbl = self.uvres*max(self.npixx, self.npixy)/2

        # max fringe window in seconds
        fringetime = 0.5*(24*3600)/(2*np.pi*maxbl/25.)

        return fringetime

    @property
    def fringetime(self):
        """ Same as fringetime_orig, but rounded to integer multiple of
        integration time.
        """

        return self.fringetime_orig - np.mod(self.fringetime_orig, self.inttime)

    @property
    def ants(self):
        return sorted([ant for ant in self.metadata.antids
                       if ant not in self.prefs.excludeants])

    @property
    def nants(self):
        return len(self.ants)

    @property
    def nbl(self):
        return int(self.nants*(self.nants-1)/2)

    @property
    def gainfile(self):
        """ Calibration file (telcal) from preferences or found from ".GN"
        suffix
        """

        if not self.prefs.gainfile:
            today = date.today()
            # look for gainfile in mchammer
            gainfile = os.path.join('/home/mchammer/evladata/telcal/'
                                    '{0}/{1:02}/{2}.GN'
                                    .format(today.year, today.month,
                                            self.metadata.filename))
        else:
            gainfile = self.prefs.gainfile

        return gainfile

    @property
    def blarr(self):
        if not hasattr(self, '_blarr'):
            self._blarr = np.array([[int(self.ants[i].lstrip('ea')),
                                     int(self.ants[j].lstrip('ea'))]
                                    for j in range(self.nants)
                                    for i in range(0, j)])

        return self._blarr

    @property
    def blarr_names(self):
        return np.array([[self.ants[i], self.ants[j]]
                         for j in range(self.nants) for i in range(0, j)])

    @property
    def nsegment(self):
        #       if self.prefs.nsegment:
        #           return self.prefs.nsegment
        #       else:
        return len(self.segmenttimes)

    @property
    def segmenttimes(self):
        """ List of tuples containing MJD times defining segment start and stop.
        Calculated from prefs.nsegment first.
        Alternately, best times found based on fringe time and memory limit
        """

        if not hasattr(self, '_segmenttimes'):
            if self.prefs.segmenttimes is not None:
                self._segmenttimes = self.prefs.segmenttimes
#            elif self.prefs.nsegment:
#                self._segmenttimes = calc_segment_times(self, self.prefs.nsegment)
            else:
                find_segment_times(self)

        return self._segmenttimes

    def pixtolm(self, pix):
        """ Helper function to calculate (l,m) coords of given pixel.
        Example: st.pixtolm(np.where(im == im.max()))
        """

        assert len(pix) == 2

        peakx, peaky = pix
        # if np.where output
        if isinstance(peaky, np.ndarray) and len(peaky) == 1:
            peaky = peaky[0]
            peakx = peakx[0]

        l1 = (self.npixx/2. - peakx)/(self.npixx*self.uvres)
        m1 = (self.npixy/2. - peaky)/(self.npixy*self.uvres)

        return l1, m1

    def get_segmenttime_string(self, segment):
        mid_mjd = self.segmenttimes[segment].mean()
        return qa.time(qa.quantity(mid_mjd, 'd'), form='ymd', prec=8)[0]

    def get_uvw_segment(self, segment):
        """ Returns uvw in units of baselines for a given segment.
        Tuple of u, v, w given with each a numpy array of (nbl, nchan) shape.
        """

        mjdstr = self.get_segmenttime_string(segment)
        (u, v, w) = util.calc_uvw(datetime=mjdstr, radec=self.metadata.radec,
                                  antpos=self.metadata.antpos,
                                  telescope=self.metadata.telescope)
#        u = u * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
#        v = v * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
#        w = w * self.metadata.freq_orig[0] * (1e9/3e8) * (-1)
        u = np.outer(u, self.freq * (1e9/3e8) * (-1))
        v = np.outer(v, self.freq * (1e9/3e8) * (-1))
        w = np.outer(w, self.freq * (1e9/3e8) * (-1))

        return u.astype('float32'), v.astype('float32'), w.astype('float32')

    @property
    def nints(self):
        return self.metadata.nints

    @property
    def t_segment(self):
        """ Time read per segment in seconds
        Assumes first segment is same size as all others.
        """

#        totaltimeread = 24*3600*(self.segmenttimes[:, 1] - self.segmenttimes[:, 0]).sum()
#        return totaltimeread/self.nsegment
        # ** not guaranteed to be the same for each segment, but assume so
        return 24*3600*(self.segmenttimes[0, 1] - self.segmenttimes[0, 0])

    @property
    def readints(self):
        """ Number of integrations read per segment.
        Defines shape of numpy array for visibilities.
        """

        return int(round(self.t_segment/self.inttime))

    @property
    def searchints(self):
        """ Number of integrations searched
        """

        return self.readints*self.nsegment

    @property
    def datashape(self):
        return (self.readints//self.prefs.read_tdownsample, self.nbl,
                self.nchan//self.prefs.read_fdownsample, self.npol)

    @property
    def datasize(self):
        return long(self.readints*self.nbl*self.nchan*self.npol /
                    (self.prefs.read_tdownsample*self.prefs.read_fdownsample))

    @property
    def nfalse(self):
        """ Calculate the number of thermal-noise false positives per segment.
        """

        # assumes dedisperse-all algorithm
        dtfactor = np.sum([1/i for i in self.dtarr])
        ntrials = self.readints * dtfactor * len(self.dmarr) * self.npixx * self.npixy
        qfrac = 1 - (erf(self.prefs.sigma_image1/np.sqrt(2)) + 1)/2
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
        """ Given searchtype, return features to be extracted in initial
        analysis
        """

        if self.prefs.searchtype == 'image1':
            return ('snr1', 'immax1', 'l1', 'm1')
        elif self.prefs.searchtype == 'image1stats':
            # note: spec statistics are all or nothing.
            return ('snr1', 'immax1', 'l1', 'm1', 'specstd', 'specskew',
                    'speckurtosis', 'imskew', 'imkurtosis')

    @property
    def candsfile(self):
        """ File name to write candidates into """

        if self.prefs.candsfile:
            return self.prefs.candsfile
        else:
            return os.path.join(self.metadata.workdir,
                                'cands_' + self.fileroot + '.pkl')

    @property
    def vismem(self):
        """ Memory required to store read data (in GB)
        """

        toGB = 8/1024**3   # number of complex64s to GB

        return self.datasize * toGB

    @property
    def vismem_limit(self):
        """ Memory required to store read data (in GB)
        Limit defined for time range equal to the overlap time between
        segments.
        """

        toGB = 8/1024**3   # number of complex64s to GB
        return toGB*long(self.t_overlap /
                         self.inttime*self.nbl*self.nchan*self.npol /
                         (self.prefs.read_tdownsample *
                          self.prefs.read_fdownsample))

    @property
    def immem(self):
        """ Memory required to create all images in a chunk of read integrations
        """

        toGB = 8/1024**3   # number of complex64s to GB
        immem = self.prefs.nthread * (self.readints * self.npixx *
                                      self.npixy) * toGB

        return immem

    @property
    def immem_limit(self):
        """ Memory required to create all images in a chunk of read integrations
        Limit defined for all threads.
        """

        toGB = 8/1024**3   # number of complex64s to GB
#        nchunk_scale = max(self.dtarr)//min(self.dtarr)
        immem = self.prefs.nthread * ((self.t_overlap/self.inttime) *
                                      self.npixx * self.npixy) * toGB

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
    dm_maxloss is sensitivity loss tolerated by dm bin width. dm_pulsewidth is
    assumed pulse width in microsec.
    """

    dm_maxloss = state.prefs.dm_maxloss
    dm_pulsewidth = state.prefs.dm_pulsewidth
    mindm = state.prefs.mindm
    maxdm = state.prefs.maxdm

    # parameters
    tsamp = state.inttime*1e6  # in microsec
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


def calc_segment_times(state, scale_nsegment=1.):
    """ Helper function for set_pipeline to define segmenttimes list.
    Forces segment time windows to be fixed relative to integration boundaries.
    Can optionally push nsegment scaling up.
    """

#    stopdts = np.linspace(int(round(state.t_overlap/state.inttime)), state.nints,
#                          nsegment+1)[1:]  # nseg+1 keeps at least one seg
#    startdts = np.concatenate(([0],
#                              stopdts[:-1]-int(round(state.t_overlap/state.inttime))))
    # or force on integer boundaries?

    stopdts = np.arange(int(round(state.t_overlap/state.inttime)), state.nints+1,
                        min(max(1,
                            int(round(state.fringetime/state.inttime/scale_nsegment))),
                        state.nints-int(round(state.t_overlap/state.inttime))),
                        dtype=int)[1:]
    startdts = np.concatenate(([0],
                              stopdts[:-1]-int(round(state.t_overlap/state.inttime))))

    assert all([len(stopdts), len(startdts)]), ('Could not set segment times.'
                                                't_overlap may be longer than '
                                                'nints or fringetime shorter '
                                                'than inttime.')

    segmenttimes = []
    for (startdt, stopdt) in zip(state.inttime*startdts, state.inttime*stopdts):
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

    scale_nsegment = 1.
    state._segmenttimes = calc_segment_times(state, scale_nsegment)

    # calculate memory limit to stop iteration
    assert state.immem_limit+state.vismem_limit < state.prefs.memory_limit, 'memory_limit of {0} is smaller than best solution of {1}. Try setting maxdm/npix_max lower.'.format(state.prefs.memory_limit, state.immem_limit+state.vismem_limit)

    if state.vismem+state.immem > state.prefs.memory_limit:
        logger.info('Over memory limit of {0} when reading {1} segments '
                    '({2}/{3} GB for visibilities/imaging). Searching for '
                    'solution down to {5}/{6} GB...'
                    .format(state.prefs.memory_limit, state.nsegment,
                            state.vismem, state.immem, state.vismem_limit,
                            state.immem_limit))

        while state.vismem+state.immem > state.prefs.memory_limit:
            logger.debug('Using {0} segments requires {1}/{2} GB for '
                         'visibilities/images. Searching for better solution.'
                         .format(state.prefs.nchunk, state.vismem, state.immem,
                                 state.prefs.memory_limit))

            scale_nsegment *= (state.vismem+state.immem)/float(state.prefs.memory_limit)
            state._segmenttimes = calc_segment_times(state, scale_nsegment)


def state_vystest(wait, catch, scantime=0, preffile=None, **kwargs):
    """ Create state to read vys data after wait seconds with nsegment segments.
    kwargs passed in as preferences via inpref argument to State.
    """

    try:
        from evla_mcast import scan_config
    except ImportError:
        logger.error('ImportError for evla_mcast. Need this library to consume multicast messages from CBE.')

    _install_dir = os.path.abspath(os.path.dirname(__file__))

    meta = {}
    prefs = {}

    # set start time (and fix antids)
    dt = time.TimeDelta(wait, format='sec')
    onesec = time.TimeDelta(1, format='sec')
    t0 = (time.Time.now()+dt).mjd
    meta['starttime_mjd'] = t0
    meta['stopime_mjd'] = t0+onesec*catch
    meta['antids'] = ['ea{0}'.format(i) for i in range(1,26)]  # fixed for scan_config test docs

    # read example scan configuration
    config = scan_config.ScanConfig(vci=os.path.join(_install_dir, 'data/vci.xml'),
                                    obs=os.path.join(_install_dir, 'data/obs.xml'),
                                    ant=os.path.join(_install_dir, 'data/antprop.xml'),
                                    requires=['ant', 'vci', 'obs'])
    config.stopTime = config.startTime+1/(24*3600.)

    st = State(config=config, preffile=preffile, inmeta=meta, inprefs=prefs)

    return st
