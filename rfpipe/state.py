from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
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

    def __init__(self, config=None, sdmfile=None, sdmscan=None, bdfdir=None,
                 inprefs=None, inmeta=None, preffile=None, name=None,
                 showsummary=True, lock=None):
        """ Initialize preference attributes with text file, preffile.
        name can select preference set from within yaml file.
        preferences are overloaded with inprefs.

        Metadata source can be either:
        1) Config object is a scan_config object (see evla_mcast library) or
        2) sdmfile and sdmscan (optional bdfdir for reading from CBE).

        inmeta is a dict with key-value pairs to overload metadata (e.g., to
        mock metadata from a simulation)
        """

        self.config = config
        self.sdmscan = sdmscan
        if sdmfile:
            sdmfile = sdmfile.rstrip('/')
        self.sdmfile = sdmfile
        self.lock = lock

        # set prefs according to inprefs type
        if isinstance(inprefs, preferences.Preferences):
            self.prefs = inprefs
        else:
            # default values will result in empty dict
            prefs = preferences.parsepreffile(preffile, inprefs=inprefs,
                                              name=name)
            self.prefs = preferences.Preferences(**prefs)

        # TODO: not working
        logger.parent.setLevel(getattr(logging, self.prefs.loglevel))

        if isinstance(inmeta, metadata.Metadata):
            self.metadata = inmeta
        else:
            if inmeta is None:
                inmeta = {}

            # get metadata
            if (self.sdmfile and self.sdmscan) and not self.config:
                meta = metadata.sdm_metadata(sdmfile, sdmscan, bdfdir=bdfdir)
            elif self.config and not (self.sdmfile or self.sdmscan):
                # config datasource can be vys or simulated data
                datasource = inmeta['datasource'] if 'datasource' in inmeta else 'vys'
                meta = metadata.config_metadata(config, datasource=datasource)
            else:
                meta = {}

            # optionally overload metadata
            if isinstance(inmeta, dict):
                for key in inmeta:
                    meta[key] = inmeta[key]

                for key in meta:
                    logger.debug(key, meta[key], type(meta[key]))
            else:
                logger.warn("inmeta not dict, Metadata, or None. Not parsed.")

            self.metadata = metadata.Metadata(**meta)

        if showsummary:
            self.summarize()

    def __repr__(self):
        return ('rfpipe state with metadata/prefs ({0}/{1})'
                .format(self.metadata.datasetId, self.prefs.name))

    def summarize(self):
        """ Print summary of pipeline state """

        if self.metadata.atdefaults():
            logger.info('Metadata not set. Cannot calculate properties')
        else:
            logger.info('Metadata summary:')
            logger.info('\t Working directory and fileroot: {0}, {1}'
                        .format(self.prefs.workdir, self.fileroot))
            logger.info('\t Using scan {0}, source {1}'
                        .format(int(self.metadata.scan), self.metadata.source))
            logger.info('\t nants, nbl: {0}, {1}'.format(self.nants, self.nbl))
            logger.info('\t nchan, nspw: {0}, {1}'
                        .format(self.nchan, self.nspw))

            spworder = np.argsort(self.metadata.spw_reffreq)
            if np.any(spworder != np.sort(spworder)):
                logger.warn('BDF spw sorted to increasing order from {0}'
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
            logger.info('\t Using {0} segment{1} of {2} ints ({3:.1f} s) with '
                        'overlap of {4:.1f} s'
                        .format(self.nsegment, "s"[not self.nsegment-1:],
                                self.readints, self.t_segment, self.t_overlap))
            logger.info('\t Searching {0} of {1} ints in scan'
                        .format(self.searchints, self.metadata.nints))
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

            if (self.prefs.read_tdownsample > 1 or self.prefs.read_fdownsample > 1):
                logger.info('\t Downsampling in time/freq by {0}/{1}.'
                            .format(self.prefs.read_tdownsample,
                                    self.prefs.read_fdownsample))
            if len(self.prefs.excludeants):
                logger.info('\t Excluding ants {0}'
                            .format(self.prefs.excludeants))

            logger.info('\t Using pols {0}'.format(self.pols))
            if os.path.exists(self.gainfile) and os.path.isfile(self.gainfile):
                logger.info('\t Found telcal file {0}'.format(self.gainfile))
            else:
                logger.warn('\t Gainfile preference ({0}) is not a telcal file'
                            .format(self.gainfile))

            logger.info('')

            logger.info('\t Using {0} for {1} search at {2} sigma using {3} thread{4}.'
                        .format(self.fftmode, self.prefs.searchtype,
                                self.prefs.sigma_image1, self.prefs.nthread,
                                's'[not self.prefs.nthread-1:]))
            logger.info('\t Using {0} DMs from {1} to {2} and dts {3}.'
                        .format(len(self.dmarr), min(self.dmarr),
                                max(self.dmarr), self.dtarr))
            logger.info('\t Using uvgrid npix=({0}, {1}) and res={2} with {3} int chunks.'
                        .format(self.npixx, self.npixy, self.uvres, self.chunksize))
            logger.info('\t Expect {0} thermal false positives per segment.'
                        .format(self.nfalse))

            logger.info('')
            if self.fftmode == "fftw":
                logger.info('\t Visibility/image memory usage is {0}/{1} '
                            'GB/segment when using fftw imaging.'
                            .format(self.vismem, self.immem))
            elif self.fftmode == "cuda":
                logger.info('\t Visibility memory usage is {0} '
                            'GB/segment when using cuda imaging.'
                            .format(self.vismem))

    def clearcache(self):
        cached = ['_dmarr', '_t_overlap', '_dmshifts', '_npol', '_blarr',
                  '_segmenttimes', '_npixx_full', '_npixy_full']
        for obj in cached:
            try:
                delattr(self, obj)
            except AttributeError:
                pass

    def get_search_ints(self, segment, dmind, dtind):
        """ Helper function to get list of integrations
        to be searched after correcting for DM and resampling.
        """

        if segment == 0:
            return list(range((self.readints-self.dmshifts[dmind])//self.dtarr[dtind]))
        else:
            return list(range((self.dmshifts[-1]-self.dmshifts[dmind])//self.dtarr[dtind],
                              (self.readints-self.dmshifts[dmind])//self.dtarr[dtind]))

    @property
    def version(self):
        if self.prefs.rfpipe_version:
            return self.prefs.rfpipe_version
        else:
            return version.__version__

    @property
    def fileroot(self):
        if self.prefs.fileroot:
            return self.prefs.fileroot
        else:
            return self.metadata.scanId

    @property
    def dmarr(self):
        if not hasattr(self, '_dmarr'):
            if self.prefs.dmarr:
                self._dmarr = self.prefs.dmarr
            else:
                self._dmarr = util.calc_dmarr(self)

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
            return list(range(sum(self.metadata.spw_nchan)))

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
        """ List of spectral windows used.
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
    def fftmode(self):
        """ Should the FFT be done with fftw or cuda?
        *Could overload here based on local configuration*
        """

        return self.prefs.fftmode

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

        if self.prefs.gainfile is None:
            today = date.today()
            # look for gainfile in mchammer
            gainfile = os.path.join('/home/mchammer/evladata/telcal/'
                                    '{0}/{1:02}/{2}.GN'
                                    .format(today.year, today.month,
                                            self.metadata.datasetId))
        else:
            if os.path.dirname(self.prefs.gainfile):  # use full path if given
                gainfile = self.prefs.gainfile
            else:  # else assume workdir
                gainfile = os.path.join(self.prefs.workdir,
                                        self.prefs.gainfile)

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
                self._segmenttimes = util.calc_segment_times(self, 1.)
                if self.memory_total > self.prefs.memory_limit:
                    logger.info('Total memory of {0} is over limit of {1} '
                                'with {2} segments. Searching to vis/im limits'
                                ' of {3}/{4} GB...'
                                .format(self.memory_total,
                                        self.prefs.memory_limit,
                                        self.nsegment, self.vismem_limit,
                                        self.immem_limit))

                    util.find_segment_times(self)

        return self._segmenttimes

    def pixtolm(self, pix):
        """ Helper function to calculate (l,m) coords of given pixel.
        Example: st.pixtolm(np.where(im == im.max()))
        """

        assert len(pix) == 2

        peakx, peaky = pix
        # if np.where output
        if isinstance(peaky, np.ndarray):
            if len(peakx) > 1 or len(peaky) > 1:
                logger.warn("More than one peak pixel ({0}, {1}). Using first."
                            .format(peakx, peaky))
            peaky = peaky[0]
            peakx = peakx[0]

        l1 = (self.npixx/2. - peakx)/(self.npixx*self.uvres)
        m1 = (self.npixy/2. - peaky)/(self.npixy*self.uvres)

        return l1, m1

    def get_segmenttime_string(self, segment):
        mid_mjd = self.segmenttimes[segment].mean()
        return qa.time(qa.quantity(mid_mjd, 'd'), form='ymd', prec=8)[0]

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

        return self.readints + \
        (self.readints - int(round(self.t_overlap/self.metadata.inttime))) * \
        max(0, (self.nsegment-1))

    @property
    def datashape(self):
        return (self.readints//self.prefs.read_tdownsample, self.nbl,
                self.nchan//self.prefs.read_fdownsample, self.npol)

    @property
    def datashape_orig(self):
        return (self.readints, self.metadata.nbl_orig,
                self.metadata.nchan_orig, self.metadata.npol_orig)

    @property
    def datasize(self):
        return np.int32(self.readints*self.nbl*self.nchan*self.npol /
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
            return os.path.join(self.prefs.workdir,
                                'cands_' + self.fileroot + '.pkl')

    @property
    def noisefile(self):
        """ File name to write noises into """

        return self.candsfile.replace('cands_', 'noise_')

    @property
    def mockfile(self):
        """ File name to write mocks into """

        return self.candsfile.replace('cands_', 'mock_')

    @property
    def vismem(self):
        """ Memory required to store read data (in GB)
        """

        toGB = 8/1000**3   # number of complex64s to GB

        return self.datasize * toGB

    @property
    def vismem_limit(self):
        """ Memory required to store read data (in GB)
        Limit defined for time range equal to the overlap time between
        segments.
        """

        toGB = 8/1000**3   # number of complex64s to GB
        return toGB*np.int32(self.t_overlap /
                             self.inttime*self.nbl*self.nchan*self.npol /
                             (self.prefs.read_tdownsample *
                              self.prefs.read_fdownsample))

    @property
    def immem(self):
        """ Memory required to create all images in a chunk of read integrations
        """

        toGB = 8/1000**3   # number of complex64s to GB
        return (self.chunksize * self.npixx * self.npixy) * toGB

    @property
    def immem_limit(self):
        """ Memory required to create all images in a chunk of read integrations
        Limit defined for all threads.
        """

        toGB = 8/1000**3   # number of complex64s to GB
        return (min(self.chunksize, (self.t_overlap/self.inttime)) * self.npixx * self.npixy) * toGB

    @property
    def memory_total(self):
        """ Total memory (in GB) required to read and process.
        Depends on data source and search algorithm.
        """

        if self.fftmode == "fftw":
            return self.immem + self.vismem
        elif self.fftmode == "cuda":
            return self.vismem

    @property
    def memory_total_limit(self):
        """ Minimum memory (in GB) required to read and process.
        Depends on data source and search algorithm.
        """

        if self.fftmode == "fftw":
            return self.immem_limit + self.vismem_limit
        elif self.fftmode == "cuda":
            return self.vismem_limit

    @property
    def chunksize(self):
        toGB = 8/1000**3   # number of complex64s to GB

        if self.prefs.maximmem is not None:
            return min(max(1,
                           int(self.prefs.maximmem/(self.npixx*self.npixy*toGB))),
                       self.readints)
        else:
            return self.readints

    @property
    def defined(self):
        return [key for key in list(self.__dict__.keys()) if key[0] != '_']

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
    meta['antids'] = ['ea{0}'.format(i) for i in range(1, 26)]  # fixed for scan_config test docs

    # read example scan configuration
    config = scan_config.ScanConfig(vci=os.path.join(_install_dir, 'data/vci.xml'),
                                    obs=os.path.join(_install_dir, 'data/obs.xml'),
                                    ant=os.path.join(_install_dir, 'data/antprop.xml'),
                                    requires=['ant', 'vci', 'obs'])
    config.stopTime = config.startTime+1/(24*3600.)

    st = State(config=config, preffile=preffile, inmeta=meta, inprefs=prefs)

    return st
