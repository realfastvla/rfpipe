from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import os
import numpy as np
from astropy import time
from rfpipe import util, version
import pwkit.environments.casa.util as casautil

import logging
logger = logging.getLogger(__name__)

qa = casautil.tools.quanta()


class State(object):
    """ Defines initial search from preferences and methods.
    State properties are used to calculate quantities for the search.
    
    Approach:
    1) initial preferences can overload state properties
    2) read metadata from observation for given scan
    3) set state (and a few cached values)
    4) state can optionally include attributes for later convenience
    5) run search on a segment.
    """

    def __init__(self, config=None, sdmfile=None, sdmscan=None, bdfdir=None,
                 inprefs=None, inmeta=None, preffile=None, name=None,
                 showsummary=True, lock=None, validate=True):
        """ Initialize preference attributes with text file, preffile.
        name can select preference set from within yaml file.
        preferences are overloaded with inprefs.
        
        Metadata source can be either:
        1) Config object is a scan_config object (see evla_mcast library) or
        2) sdmfile and sdmscan (optional bdfdir for reading from CBE).
        
        inmeta is a dict with key-value pairs to overload metadata (e.g., to
        mock metadata from a simulation)
        validate argument will use assertions to test state.
        """

        from rfpipe import preferences, metadata

        self.config = config
        self.sdmscan = sdmscan
        if sdmfile:
            sdmfile = sdmfile.rstrip('/')
        self.sdmfile = sdmfile
        self.lock = lock
        self._corrections = None

        # set prefs according to inprefs type
        if isinstance(inprefs, preferences.Preferences):
            self.prefs = inprefs
        else:
            # default values will result in empty dict
            prefs = preferences.parsepreffile(preffile, inprefs=inprefs,
                                              name=name)
            try:
                self.prefs = preferences.Preferences(**prefs)
            except TypeError as exc:
                from fuzzywuzzy import fuzz
                badarg = exc.args[0].split('\'')[1]
                closeprefs = [pref for pref in list(preferences.Preferences().__dict__) if fuzz.ratio(badarg, pref) > 50]
                raise TypeError("Preference {0} not recognized. Did you mean {1}?".format(badarg, ', '.join(closeprefs)))

        # TODO: not working
        logger.parent.setLevel(getattr(logging, self.prefs.loglevel))

        self.metadata = metadata.make_metadata(config=config, sdmfile=sdmfile,
                                               sdmscan=sdmscan, inmeta=inmeta,
                                               bdfdir=bdfdir)

        if validate:
            assert self.validate() is True

        if showsummary:
            self.summarize()

    def __repr__(self):
        return ('rfpipe state with metadata/prefs ({0}/{1})'
                .format(self.metadata.datasetId, self.prefs.name))

    def validate(self):
        """ Test validity of state (metadata + preferences) with a assertions
        and module imports.
        """

        if self.metadata.datasource == 'sdm':
            assert self.metadata.bdfstr is not None, "No bdf found."            

        # limits on time boundaries/sizes
        assert self.t_overlap < self.nints*self.inttime, ('t_overlap must be'
                                                          ' less than scan '
                                                          'length ({0:.3f} < {1:.3f}) '
                                                          .format(self.t_overlap,
                                                                  self.nints*self.inttime))
        assert self.t_overlap < self.t_segment, ('Max DM sweep ({0:.3f} s)'
                                                 ' is larger than segment '
                                                 'size ({1:.3f} s). '
                                                 'Pipeline will fail!'
                                                 .format(self.t_overlap,
                                                         self.t_segment))
        assert self.readints >= max(self.dtarr)

        if self.prefs.timesub is not None:
            assert self.inttime < self.fringetime_orig, ('Integration time '
                                                         'must be less than '
                                                         'fringe time '
                                                         '({0:.3f} < {1:.3f}) for vis '
                                                         'subtraction'
                                                         .format(self.inttime,
                                                                 self.fringetime))
        # required parameters
        if self.prefs.searchtype == 'image':
            assert self.prefs.sigma_image1 is not None, "Must set sigma_image1"
        elif self.prefs.searchtype == 'imagek':
            assert self.prefs.sigma_image1 is not None, "Must set sigma_image1"
            assert self.prefs.sigma_kalman is not None, "Must set sigma_kalman"
        elif self.prefs.searchtype == 'armkimage':
            assert self.prefs.sigma_arm is not None, "Must set sigma_arm"
            assert self.prefs.sigma_arms is not None, "Must set sigma_arms"
            assert self.prefs.sigma_kalman is not None, "Must set sigma_kalman"

        # supported algorithms for gpu/cpu
        if self.prefs.fftmode == 'cuda' and self.prefs.searchtype is not None:
            assert self.prefs.searchtype in ['image', 'imagek']
        elif self.prefs.fftmode == 'fftw' and self.prefs.searchtype is not None:
            assert self.prefs.searchtype in ['image', 'imagek', 'armkimage', 'armk']

        return True

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

            # TODO: remove for datasource=vys or sim? or remove altogether?
#            spworder = np.argsort(self.metadata.spw_reffreq)
#            if np.any(spworder != np.sort(spworder)) and self.metadata.datasource == 'sdm':
#                logger.warning('BDF spw sorted to increasing order from {0}'
#                            .format(spworder))

            logger.info('\t Freq range: {0:.3f} -- {1:.3f}'
                        .format(self.freq.min(), self.freq.max()))
            logger.info('\t Scan has {0} ints ({1:.1f} s) and inttime {2:.3f} s'
                        .format(self.nints, self.nints*self.inttime,
                                self.inttime))
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
            if self.t_overlap > self.t_segment/2.:
                logger.info('\t\t Highly redundant reading. Max DM sweep '
                            '({0:.1f} s) > 1/2 segment size ({1:.1f} s)'
                            .format(self.t_overlap, self.t_segment))

            if (self.prefs.read_tdownsample > 1 or self.prefs.read_fdownsample > 1):
                logger.info('\t Downsampling in time/freq by {0}/{1}.'
                            .format(self.prefs.read_tdownsample,
                                    self.prefs.read_fdownsample))
            if len(self.prefs.excludeants):
                logger.info('\t Excluding ants {0}'
                            .format(self.prefs.excludeants))

            logger.info('\t Using pols {0}'.format(self.pols))
            if self.gainfile is not None:
                logger.info('\t Found telcal file {0}'.format(self.gainfile))
            else:
                logger.info("No gainfile specified or found.")

            logger.info('')

            logger.info('\t Using {0} for {1} search using {2} thread{3}.'
                        .format(self.fftmode, self.prefs.searchtype,
                                self.prefs.nthread,
                                's'[not self.prefs.nthread-1:]))
            if self.prefs.searchtype == 'image':
                logger.info('\t Image threshold of {0} sigma.'
                            .format(self.prefs.sigma_image1,
                                    self.prefs.sigma_kalman))
            elif self.prefs.searchtype == 'imagek':
                logger.info('\t Image/kalman threshold of {0}/{1} sigma.'
                            .format(self.prefs.sigma_image1,
                                    self.prefs.sigma_kalman))
            elif self.prefs.searchtype == 'armkimage':
                logger.info('\t Arm/arms/kalman/image thresholds of {0}/{1}/{2}/{3} sigma.'
                            .format(self.prefs.sigma_arm,
                                    self.prefs.sigma_arms,
                                    self.prefs.sigma_kalman,
                                    self.prefs.sigma_image1))

            logger.info('\t Using {0} DMs from {1} to {2} and dts {3}.'
                        .format(len(self.dmarr), min(self.dmarr),
                                max(self.dmarr), self.dtarr))
            logger.info('\t Using uvgrid npix=({0}, {1}) and res={2} with {3} int chunks.'
                        .format(self.npixx, self.npixy, self.uvres, self.chunksize))
            logger.info('\t Expect {0} thermal false positives per scan.'
                        .format(self.nfalse(self.prefs.sigma_image1)))
            if self.prefs.clustercands is not None:
                if isinstance(self.prefs.clustercands, tuple):
                    min_cluster_size, min_samples = self.prefs.clustercands
                    logger.info('\t Clustering candidates wth min_cluster_size={0} and min_samples={1}'
                                .format(min_cluster_size, min_samples))
                    if min_cluster_size <= len(self.dtarr):
                        logger.warning("min_cluster_size should be > len(dtarr) for best results")
                elif isinstance(self.prefs.clustercands, bool):
                    if self.prefs.clustercands:
                        logger.info('\t Clustering candidates wth default parameters')

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
        cached = ['_dmarr', '_dmshifts', '_npol', '_blarr',
                  '_segmenttimes', '_npixx_full', '_npixy_full',
                  '_corrections']
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
            if self.prefs.fftmode == 'fftw':
                return list(range((self.readints-self.dmshifts[dmind])//self.dtarr[dtind]))
            elif self.prefs.fftmode == 'cuda':
                return list(range(self.readints//self.dtarr[dtind]-self.dmshifts[dmind]//self.dtarr[dtind]))
        elif segment < self.nsegment:
            if self.prefs.fftmode == 'fftw':
                return list(range((self.dmshifts[-1]-self.dmshifts[dmind])//self.dtarr[dtind],
                                  (self.readints-self.dmshifts[dmind])//self.dtarr[dtind]))
            elif self.prefs.fftmode == 'cuda':
                # TODO: fix this
                return list(range((self.dmshifts[-1]//self.dtarr[dtind]-self.dmshifts[dmind]//self.dtarr[dtind]),
                                  (self.readints//self.dtarr[dtind]-self.dmshifts[dmind]//self.dtarr[dtind])))
        else:
            logger.warning("No segment {0} in scan".format(segment))

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
        if self.prefs.dmarr:
            return self.prefs.dmarr
        else:
            if not hasattr(self, '_dmarr'):
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
        TODO: test effect of metadata spw out of order (but not data reading order?)
        """

        # TODO: add support for frequency downsampling

        return self.metadata.freq_orig[self.chans]

    @property
    def chans(self):
        """ List of channel indices to use. Drawn from preferences,
        with backup to take all those for preferred spw.
        """

        if self.prefs.chans:
            return self.prefs.chans
        else:
#            list(range(sum(self.metadata.spw_nchan)))
            chanlist = []
            nch = np.unique(self.metadata.spw_nchan)[0]  # assume 1 nchan/spw
            if self.prefs.ignore_spwedge:
                edge = int(self.prefs.ignore_spwedge*nch)
            else:
                edge = 0
            for spw in self.spw:
                spwi = self.metadata.spw_orig.index(spw)
                chanlist += list(range(nch*spwi+edge, nch*(spwi+1)-edge))
            return chanlist

    @property
    def spw(self):
        """ List of spectral windows used.
        ** TODO: update for proper naming "basband"+"swindex"
        ** OR: reflect spworder here and select for spw
        """

        if self.prefs.spw:
            return self.prefs.spw
        else:
            return self.metadata.spw_orig

    @property
    def nspw(self):
        return len(self.spw)

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
            from rfpipe import util
            self._dmshifts = [util.calc_delay(self.freq, self.freq.max(), dm,
                              self.inttime).max()
                              for dm in self.dmarr]
        return self._dmshifts

    @property
    def t_overlap(self):
        """ Max DM delay in seconds that is fixed to int mult of integration time.
        Gets cached. """

        return max(self.dmshifts)*self.inttime

    @property
    def spw_chan_select(self):
        """ List of lists with selected channels per spw.
        Channel numbers assume selected data.
        """

        reffreq, nchan, chansize = self.metadata.spw_sorted_properties
        chanlist = []
        for spwi in range(len(reffreq)):
            nch = nchan[spwi]
            chans = [self.chans.index(ch) for ch in list(range(nch*spwi, nch*(spwi+1))) if ch in self.chans]
            chanlist.append(chans)

        return chanlist

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
        """
        Polarizations to use based on preference in prefs.selectpol
        Can provide description ('auto', 'all', 'cross') or list of pol
        names to select from all listed in metadata.
        """

        if self.prefs.selectpol == 'auto':
            return [pp for pp in self.metadata.pols_orig if pp[0] == pp[-1]]
        elif self.prefs.selectpol == 'cross':
            return [pp for pp in self.metadata.pols_orig if pp[0] != pp[-1]]
        elif self.prefs.selectpol == 'all':
            return self.metadata.pols_orig
        elif isinstance(self.prefs.selectpol, list):
            return [pp for pp in self.metadata.pols_orig if pp in self.prefs.selectpol]
        else:
            logger.warning('selectpol of {0} not supported'
                        .format(self.prefs.selectpol))

    @property
    def uvres_full(self):
        return int(round(self.metadata.dishdiameter / (3e-1/self.freq.min()) / 2))

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
    def fieldsize_deg(self):
        """ Takes gridding spec to estimate field of view in degrees
        """

        return np.degrees(1/(self.uvres*2.))

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
        suffix with datasetId. Default behavior is to find file in workdir.
        Value of None means none will be applied.
        """

        if self.prefs.gainfile is None:
            gainfile = os.path.join(self.prefs.workdir,
                                    self.metadata.datasetId) + ".GN"
        else:
            if os.path.dirname(self.prefs.gainfile):  # use full path if given
                gainfile = self.prefs.gainfile
            else:  # else assume workdir
                gainfile = os.path.join(self.prefs.workdir,
                                        self.prefs.gainfile)

        if (not os.path.exists(gainfile)) or (not os.path.isfile(gainfile)):
            gainfile = None

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
    def blarr_arms(self):
        return np.array([[self.metadata.stationids[i][0], self.metadata.stationids[j][0]]
                         for j in range(self.nants) for i in range(0, j)])

    def blind_arm(self, arm):
        """ Give the index of baseline with given arm "N", "E", or "W".
        """

        return np.where(np.all(self.blarr_arms == arm, axis=1))[0]

    @property
    def nsegment(self):
        #       if self.prefs.nsegment:
        #           return self.prefs.nsegment
        #       else:
        return len(self.segmenttimes)

    @property
    def segmenttimes(self):
        """ Array of float pairs containing MJD times defining segment start and stop.
        Calculated from prefs.nsegment first.
        Alternately, best times found based on fringe time and memory limit
        """

        if not hasattr(self, '_segmenttimes'):
            if self.prefs.segmenttimes is not None:
                self._segmenttimes = np.array(self.prefs.segmenttimes)
#            elif self.prefs.nsegment:
#                self._segmenttimes = calc_segment_times(self, self.prefs.nsegment)
            else:
                from rfpipe import util
                self._segmenttimes = util.calc_segment_times(self, 1.)
                if self.memory_total > self.prefs.memory_limit:
                    util.find_segment_times(self)

        return self._segmenttimes

    @property
    def otfcorrections(self):
        """ Use otf phasecenters (if set) to calc the phase shift from 
        First radec (set in metadata) to actual radec for phase center in segment.
        """

        if self.metadata.phasecenters is None:
            return None

        if self._corrections is None and self.metadata.phasecenters is not None:
            self._corrections = {}
            for segment in range(self.nsegment):
                segmenttime0, segmenttime1 = self.segmenttimes[segment]
                bintimes = segmenttime0 + self.inttime*(0.5+np.arange(self.readints))/(24*3600)
                pcts = {i: [] for i in range(len(self.metadata.phasecenters))}
                corrs = []

                # assign integration to a window
                for i, bintime in enumerate(bintimes):
                    for j, (startmjd, stopmjd, ra_deg, dec_deg) in enumerate(self.metadata.phasecenters):
                        if (bintime >= startmjd) and (bintime < stopmjd):
                            pcts[j].append(i)

                # calculate corrections
                for j in range(len(self.metadata.phasecenters)):
                    (startmjd, stopmjd, ra_deg, dec_deg) = self.metadata.phasecenters[j]
                    if len(pcts[j]):
                        ints0 = pcts[j]
                        logger.info("Segment {0}, ints {1} will have phase center at {2},{3}"
                                    .format(segment, ints0, ra_deg, dec_deg))
                        corrs.append((ints0, ra_deg, dec_deg),)
                    else:
                        logger.debug("Phase center ({0},{1}) not in segment ({2}-{3})"
                                     .format(ra_deg, dec_deg, segmenttime0,
                                             segmenttime1))
                if not any(pcts.values()):
                    logger.warning("phasecenters found, but not overlapping with segment {0}"
                                   .format(segment))

                self._corrections[segment] = corrs

        return self._corrections

    def pixtolm(self, pix):
        """ Helper function to calculate (l,m) coords of given pixel.
        Example: st.pixtolm(np.where(im == im.max()))
        """

        assert len(pix) == 2

        peakx, peaky = pix
        # if np.where output
        if isinstance(peaky, np.ndarray):
            if len(peakx) > 1 or len(peaky) > 1:
                logger.warning("More than one peak pixel ({0}, {1}). Using first."
                            .format(peakx, peaky))
            peaky = peaky[0]
            peakx = peakx[0]

        l1, m1 = self.calclm(self.npixx, self.npixy, self.uvres, peakx, peaky)

        return l1, m1

    @staticmethod
    def calclm(npixx, npixy, uvres, peakx, peaky):
        l1 = (npixx/2. - peakx)/(npixx*uvres)
        m1 = (npixy/2. - peaky)/(npixy*uvres)
        return l1, m1

    @staticmethod
    def calcpix(candl, candm, npixx, npixy, uvres):
        """Convert from candidate l,m to x,y pixel number
        """
        peakx = np.round(npixx/2. - candl*(npixx*uvres)).astype(int)
        peaky = np.round(npixy/2. - candm*(npixy*uvres)).astype(int)
        return peakx, peaky

    def get_segmenttime_string(self, segment):
        mid_mjd = self.segmenttimes[segment].mean()
        return qa.time(qa.quantity(mid_mjd, 'd'), form='ymd', prec=8)[0]

    @property
    def nints(self):
        assert self.metadata.nints > 0, "metadata.nints must be greater than zero"
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
        """ Number of integrations searched per scan
        """

        return self.readints + \
               (self.readints - int(round(self.t_overlap/self.inttime))) * \
               max(0, (self.nsegment-1))

    @property
    def ntrials(self):
        """ Number of search trials per scan
        """

        dtfactor = np.sum([1/i for i in self.dtarr])
        return self.searchints * dtfactor * len(self.dmarr) * self.npixx * self.npixy

    def nfalse(self, sigma):
        """ Number of thermal-noise false positives per scan at given sigma
        """
        import scipy.stats

        qfrac = scipy.stats.norm.sf(sigma)
        return int(qfrac*self.ntrials)

    def thresholdlevel(self, nfalse):
        """ Sigma threshold for a given number of false positives per scan
        """
        import scipy.stats

        return scipy.stats.norm.isf(nfalse/self.ntrials)

#    @property
#    def sigma_image1(self):
#        """ Use either sigma_image1 or nfalse
#        nfalse value calculates total number of pixels formed if imaging all
#        data, which is not true for some searchtypes.
#        """
#
#        assert (self.prefs.sigma_image1 is not None) or (self.prefs.nfalse  is not None), "Must set either prefs.sigma_image1 or prefs.nfalse"
#
#        if self.prefs.sigma_image1 is not None:
#            return self.prefs.sigma_image1
#        elif self.prefs.nfalse is not None:
#            return self.thresholdlevel(self.prefs.nfalse)
#        else:
#            logger.warning("Cannot set sigma_image1 from given preferences")

    @property
    def datashape(self):
        return (self.readints//self.prefs.read_tdownsample, self.nbl,
                self.nchan//self.prefs.read_fdownsample, self.npol)

    @property
    def datasize(self):
        return np.prod(self.datashape)

    @property
    def datashape_orig(self):
        return (self.readints, self.metadata.nbl_orig,
                self.metadata.nchan_orig, self.metadata.npol_orig)

    @property
    def datasize_orig(self):
        return np.prod(self.datashape_orig)

    @property
    def search_dimensions(self):
        """ Define dimensions searched for a given piece of data.
        Actual algorithm defined in pipeline iteration.
        """

        return ('segment', 'integration', 'dmind', 'dtind', 'beamnum')

    @property
    def searchfeatures(self):
        """ Given searchtype, return features to be extracted during search.
        """
        # TODO: overload with preference for features as a list of names

        if self.prefs.searchfeatures is not None:
            return self.prefs.searchfeatures
        elif self.prefs.searchtype in ['image', 'image1', 'imagek']:
            return ('snr1', 'snrk', 'immax1', 'l1', 'm1')
        elif self.prefs.searchtype == 'armk':
            return ('snrarms', 'snrk', 'l1', 'm1')
        elif self.prefs.searchtype == 'armkimage':
            return ('snrarms', 'snrk', 'snr1', 'immax1', 'l1', 'm1')
        else:
            return ()

    @property
    def features(self):
        """ Sum of search features (those used to detect candidate) and
        features calculated in second step (more computationally demanding).
        """

        return tuple(self.searchfeatures) + tuple(self.prefs.calcfeatures)

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

        return toGB * self.datasize_orig

    @property
    def vismem_limit(self):
        """ Memory required to store read data (in GB)
        Limit defined for time range equal to the overlap time between
        segments.
        """

        toGB = 8/1000**3   # number of complex64s to GB
        return toGB * ((self.datasize_orig//self.readints *
                       int(round(self.t_overlap/self.inttime))))

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
        return (min(self.chunksize, int(round(self.t_overlap/self.inttime))) * self.npixx * self.npixy) * toGB

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
