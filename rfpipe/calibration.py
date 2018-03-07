from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input#, str # not numpy/python2 compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
import os.path
from numba import jit

import logging
logger = logging.getLogger(__name__)


### Functional form

def apply_telcal(st, data, threshold=1/50., onlycomplete=True, sign=+1):
    """ Wrap all telcal functions to parse telcal file and apply it to data
    sign defines if calibration is applied (+1) or backed out (-1).
    assumes dual pol and that each spw has same nch and chansize.
    Threshold is minimum ratio of gain amp to median gain amp.
    """

    assert sign in [-1, +1], 'sign must be +1 or -1'

    if (not os.path.exists(st.gainfile)) or (not os.path.isfile(st.gainfile)):
        logger.warn('{0} is not a telcal file. No {1} calibration to apply.'
                    .format(st.gainfile, ['', 'forward', 'inverse'][sign]))
        return data

    pols = [0, 1]
    reffreq = np.array(st.metadata.spw_reffreq)
    chansize = np.array(st.metadata.spw_chansize)
    nchan = np.array(st.metadata.spw_nchan)

    sols = parseGN(st.gainfile)
    # must run time select before flagants for complete solutions
    sols = select(sols, time=st.segmenttimes.mean())
    sols = flagants(sols, threshold=threshold, onlycomplete=onlycomplete)  

    skyfreqs = np.around(reffreq + (chansize*nchan/2), -6)/1e6  # GN skyfreq is band center
    if len(sols):
#        print(sols, st.blarr, skyfreqs, pols, chansize[0], nchan[0], sign)
#        print(type(sols), type(st.blarr), type(skyfreqs), type(pols), type(chansize[0]), type(nchan[0]), type(sign))
        gaindelay = calcgaindelay(sols, st.blarr, skyfreqs, pols, chansize[0],
                                  nchan[0], sign=sign)

        # data should have nchan_orig because no selection done yet
        # TODO: make nchan, npol, nbl selection consistent for all data types
        return data*gaindelay.reshape((st.nbl, st.metadata.nchan_orig,
                                       len(pols)))
    else:
        return data


def parseGN(telcalfile):
    """Takes .GN telcal file and places values in numpy arrays.
    threshold and onlycomplete define flagging of low gains and incomplete
    solutions.
    """

    # Define telcal file formatting
    skip = 3   # skip first three header lines
    MJD = 0; UTC = 1; LSTD = 2; LSTS = 3; IFID = 4; SKYFREQ = 5; ANT = 6; AMP = 7; PHASE = 8
    RESIDUAL = 9; DELAY = 10; FLAGGED = 11; ZEROED = 12; HA = 13; AZ = 14; EL = 15
    SOURCE = 16; FLAGREASON = 17

    mjd = []
    ifid = []
    skyfreq = []
    antnum = []
    amp = []
    phase = []
    delay = []
    flagged = []
    source = []

    i = 0
    with open(telcalfile, 'r') as fp:
        for line in fp:

            fields = line.split()
            if i < skip:
                i += 1
                continue

            try:
                mjd.append(float(fields[MJD]))
                ifid.append(str(fields[IFID]))
                skyfreq.append(float(fields[SKYFREQ]))
                antnum.append(int(fields[ANT].lstrip('ea')))
                amp.append(float(fields[AMP]))
                phase.append(float(fields[PHASE]))
                delay.append(float(fields[DELAY]))
                flagged.append('true' == (fields[FLAGGED]))
                source.append(str(fields[SOURCE]))
            except ValueError:
                logger.warn('Trouble parsing line of telcal file. Skipping.')
                continue

    # TODO: assumes dual pol. update to full pol
    polarization = [('C' in i0 or 'D' in i0) for i0 in ifid]
    fields = [str(ff) for ff in ['mjd', 'ifid', 'skyfreq', 'antnum', 'polarization', 'source', 'amp', 'phase', 'delay', 'flagged']]
    types = [str(tt) for tt in ['<f8', 'U4', '<f8', 'i8', 'i8', 'U20', '<f8', '<f8', '<f8', '?']]
    dtype = list(zip(fields, types))
    if (len(mjd) == len(phase)) and (len(phase) > 0):
        sols = np.zeros(len(mjd), dtype=dtype)

        for i in range(len(mjd)):
            sols[i] = (mjd[i], ifid[i], skyfreq[i], antnum[i], polarization[i],
                       source[i], amp[i], phase[i], delay[i], flagged[i])

        logger.info('Read telcalfile {0} with {1} sources, {2} times, {3} '
                    'IFIDs, and {4} antennas'
                    .format(telcalfile,
                            len(np.unique(sols['source'])),
                            len(np.unique(sols['mjd'])),
                            len(np.unique(sols['ifid'])),
                            len(np.unique(sols['antnum']))))
    else:
        logger.warn('Bad telcalfile {0}. Not parsed properly'.format(telcalfile))
        sols = np.array([])

    return sols


def flagants(solsin, threshold, onlycomplete):
    """ Flags solutions with amplitude more than threshold larger than median.
    onlycomplete defines whether to flag times with incomplete solutions.
    """

    if not len(solsin):
        return solsin

    sols = solsin.copy()

    # identify very low gain amps not already flagged
    badsols = np.where((sols['amp']/np.median(sols['amp']) < threshold) &
                       (sols['flagged'] == False))[0]
    if len(badsols):
        logger.info('Flagging {0} solutions at MJD {1}, ant {2}, and freqs '
                    '{3}) for low gain amplitude.'
                    .format(len(badsols), np.unique(sols[badsols]['mjd']),
                            np.unique(sols[badsols]['antnum']),
                            np.unique(sols[badsols]['ifid'])))
        for sol in badsols:
            sols['flagged'][sol] = True

    if onlycomplete:
        ifids = np.unique(sols['ifid'])
        antnums = np.unique(sols['antnum'])
        mjds = sols['mjd']

        completecount = len(ifids) * len(antnums)
        for mjd0 in np.unique(mjds):
            sols0 = np.where(mjd0 == mjds)[0]
            if len(sols0) < completecount:
                logger.info("Solution set at MJD {0} has only {1} of {2} "
                            "solutions. Flagging..."
                            .format(mjd0, len(sols0), completecount))
                sols[sols0]['flagged'] = True

    return sols


def select(sols, time=None, freqs=None, polarization=None):
    """ Selects a solution set based on given time and freqs.
    time (in mjd) defines the time to find solutions.
    freqs (in Hz) is frequencies in data.
    """

    if not len(sols):
        return sols

    # select freq if solution band center is in (rounded) array of chan freqs
    if freqs is not None:
        freqselect = [ff in np.around(freqs, -6)
                      for ff in np.around(1e6*sols['skyfreq'], -6)]
    else:
        freqselect = np.ones(len(sols), dtype=bool)

    # select by smallest time distance for source
    if time is not None:
        mjddist = np.abs(time - sols['mjd'])
        mjdselect = mjddist == mjddist.min()
    else:
        mjdselect = np.ones(len(sols), dtype=bool)

    if polarization is not None:
        polselect = sols['polarization'] == polarization
    else:
        polselect = np.ones(len(sols), dtype=bool)

    selection = np.where(freqselect*mjdselect*polselect)

    sources = np.unique(sols['source'][selection])
    times = np.unique(sols['mjd'][selection])
    if (len(sources) == 1) and (len(times) == 1):
        logger.info('Selecting {0} solutions from calibrator {1} separated by {2} min.'
                    .format(len(selection[0]), sources[0],
                            mjddist[np.where(mjdselect)][0]*24*60))
    else:
        logger.info('Existing calibration selection includes multiple solutions.')

    logger.debug('Mid frequency (MHz): {0}'
                 .format(np.unique(sols['skyfreq'][selection])))
    logger.debug('IFID: {0}'.format(np.unique(sols['ifid'][selection])))
    logger.debug('Ants: {0}'.format(np.unique(sols['antnum'][selection])))

    return sols[selection]


# @jit  # this fails after unicode_literals change
def calcgaindelay(sols, bls, freqarr, pols, chansize, nch, sign=1):
    """ Build gain calibraion array with shape to project into data
    freqarr is a list of reffreqs in MHz.
    """

    assert sign in [-1, +1], 'sign must be +1 or -1'

    nspw = len(freqarr)
    gaindelay = np.zeros((len(bls), nspw, nch, len(pols)), dtype=np.complex64)

    for bi in range(len(bls)):
        ant1, ant2 = bls[bi]
        for fi in range(len(freqarr)):
            relfreq = chansize*(np.arange(nch) - nch//2)

            for pi in range(len(pols)):
                g1 = 0.
                g2 = 0.
                d1 = 0.
                d2 = 0.
                for sol in sols:
                    if ((sol['polarization'] == pols[pi]) and (sol['skyfreq'] == freqarr[fi]) and (not sol['flagged'])):
                        if sol['antnum'] == ant1:
                            g1 = sol['amp']*np.exp(1j*np.radians(sol['phase']))
                            d1 = sol['delay']
                        if sol['antnum'] == ant2:
                            g2 = sol['amp']*np.exp(-1j*np.radians(sol['phase']))
                            d2 = sol['delay']

                if (g1 != 0.) and (g2 != 0.):
                    if sign == 1:
                        g1g2 = 1./(g1*g2)
                    elif sign == -1:
                        g1g2 = (g1*g2)
                else:
                    g1g2 = 0.

                d1d2 = sign*2*np.pi*((d1-d2) * 1e-9) * relfreq
                gaindelay[bi, fi, :, pi] = g1g2*np.exp(-1j*d1d2)

    return gaindelay


### Class form

def apply_telcal_class(st, data, calname=None, sign=+1):
    """ Wrap all telcal functions to parse telcal file and apply it to data
    sign defines if calibration is applied (+1) or backed out (-1).
    """

    assert sign in [-1, +1], 'sign must be +1 or -1'

    sols = telcal_sol(st.gainfile)
    sols.set_selection(st.segmenttimes.mean(), st.freq*1e9, st.blarr,
                       calname=calname)
    sols.apply(data, sign=sign)

    return data


class telcal_sol():
    """ Instantiated with on telcalfile.
    Parses .GN file and provides tools for applying to data of shape (nints,
    nbl, nch, npol)
    """

    def __init__(self, telcalfile, flagants=True):
        self.logger = logging.getLogger(__name__)

        if os.path.exists(telcalfile):
            self.parseGN(telcalfile)
            self.logger.info('Read telcalfile {0}'.format(telcalfile))
            if flagants:
                self.flagants()
        else:
            self.logger.warn('Gainfile {0} not found.'.format(telcalfile))
            raise IOError

    def flagants(self, threshold=50):
        """ Flags solutions with amplitude more than threshold larger than median.
        """

        # identify very low gain amps not already flagged
        badsols = np.where( (np.median(self.amp)/self.amp > threshold) & (self.flagged == False))[0]
        if len(badsols):
            self.logger.info('Solutions %s flagged (times %s, ants %s, freqs %s) for low gain amplitude.' % (str(badsols), self.mjd[badsols], self.antname[badsols], self.ifid[badsols]))
            for sol in badsols:
                self.flagged[sol] = True

    def set_selection(self, time, freqs, blarr, calname='', radec=(), dist=0, spwind=[], pols=['XX','YY']):
        """ Set select parameter that defines spectral window, time, or any other selectionp.
        time (in mjd) defines the time to find solutions near for given calname.
        freqs (in Hz) is frequencies in data.
        blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
        calname defines the name of the calibrator to use. if blank, uses only the time selection.
        pols is from d['pols'] (e.g., ['RR']). single or dual parallel allowed. not yet implemented.
        radec, dist, spwind not used. here for uniformity with casa_sol.
        """

        self.freqs = freqs
        self.chansize = freqs[1]-freqs[0]
        self.select = self.complete   # use only complete solution sets (set during parse)
        self.blarr = blarr
        if spwind:
            self.logger.warn('spwind option not used for telcal_sol. Applied based on freqs.')
        if radec:
            self.logger.warn('radec option not used for telcal_sol. Applied based on calname.')
        if dist:
            self.logger.warn('dist option not used for telcal_sol. Applied based on calname.')

        # define pol index
        if 'X' in ''.join(pols) or 'Y' in ''.join(pols):
            polord = ['XX', 'YY']
        elif 'R' in ''.join(pols) or 'L' in ''.join(pols):
            polord = ['RR', 'LL']
        self.polind = [polord.index(pol) for pol in pols]

        if calname:
            nameselect = []
            for ss in np.unique(self.source[self.select]):
                if calname in ss:
                    nameselect = np.where(self.source[self.select] == ss)   # define selection for name
                    self.select = self.select[nameselect]       # update overall selection
                    self.logger.debug('Selection down to %d solutions with %s' % (len(self.select), calname))
            if not nameselect:
                self.logger.warn('Calibrator name %s not found. Ignoring.' % (calname))

        # select freq
        freqselect = np.where([ff in np.around(self.freqs, -6) for ff in np.around(1e6*self.skyfreq[self.select], -6)])   # takes solution if band center is in (rounded) array of chan freqs
        if len(freqselect[0]) == 0:
            raise Exception('No complete set of telcal solutions at that frequency.')
        self.select = self.select[freqselect[0]]    # update overall selection
        self.logger.info('Frequency selection cut down to %d solutions' % (len(self.select)))

        # select pol
#        ifids = self.ifid[self.select]
#        if (polstr == 'RR') or (polstr == 'XX'):
#            polselect = np.where(['A' in ifid or 'B' in ifid for ifid in ifids])
#        elif (polstr == 'LL') or (polstr == 'YY'):
#            polselect = np.where(['C' in ifid or 'D' in ifid for ifid in ifids])
#        self.select = self.select[polselect]    # update overall selection
        self.polarization = np.empty(len(self.ifid))
        for i in range(len(self.ifid)):
            if ('A' in self.ifid[i]) or ('B' in self.ifid[i]):
                self.polarization[i] = 0
            elif ('C' in self.ifid[i]) or ('D' in self.ifid[i]):
                self.polarization[i] = 1

        # select by smallest time distance for source
        mjddist = np.abs(time - np.unique(self.mjd[self.select]))
        closest = np.where(mjddist == mjddist.min())
        if len(closest[0]) > 1:
            self.logger.info('Multiple closest solutions in time (%s). Taking first.' % (str(closest[0])))
            closest = closest[0][0]
        timeselect = np.where(self.mjd[self.select] == np.unique(self.mjd[self.select])[closest])   # define selection for time
        self.select = self.select[timeselect[0]]    # update overall selection
        self.logger.info('Selection down to %d solutions separated from given time by %d minutes' % (len(self.select), mjddist[closest]*24*60))

        self.logger.debug('Selected solutions: %s' % str(self.select))
        self.logger.info('MJD: %s' % str(np.unique(self.mjd[self.select])))
        self.logger.debug('Mid frequency (MHz): %s' % str(np.unique(self.skyfreq[self.select])))
        self.logger.debug('IFID: %s' % str(np.unique(self.ifid[self.select])))
        self.logger.info('Source: %s' % str(np.unique(self.source[self.select])))
        self.logger.debug('Ants: %s' % str(np.unique(self.antname[self.select])))

    def parseGN(self, telcalfile, onlycomplete=True):
        """Takes .GN telcal file and places values in numpy arrays.
        onlycomplete defines whether to toss times with less than full set of solutions (one per spw, pol, ant).
        """

        skip = 3   # skip first three header lines
        MJD = 0; UTC = 1; LSTD = 2; LSTS = 3; IFID = 4; SKYFREQ = 5; ANT = 6; AMP = 7; PHASE = 8
        RESIDUAL = 9; DELAY = 10; FLAGGED = 11; ZEROED = 12; HA = 13; AZ = 14; EL = 15
        SOURCE = 16
        #FLAGREASON = 17

        mjd = []; utc = []; lstd = []; lsts = []; ifid = []; skyfreq = []; 
        antname = []; amp = []; phase = []; residual = []; delay = []; 
        flagged = []; zeroed = []; ha = []; az = []; el = []; source = []
        #flagreason = []

        i = 0
        with open(telcalfile, 'r') as fp:
            for line in fp:

                fields = line.split()
                if i < skip:
                    i += 1
                    continue

                if ('NO_ANTSOL_SOLUTIONS_FOUND' in line):
                    # keep ERROR solutions now that flagging works
                    continue

                try:
                    mjd.append(float(fields[MJD])); utc.append(fields[UTC]); lstd.append(float(fields[LSTD])); lsts.append(fields[LSTS])
                    ifid.append(fields[IFID]); skyfreq.append(float(fields[SKYFREQ])); antname.append(fields[ANT])
                    amp.append(float(fields[AMP])); phase.append(float(fields[PHASE])); residual.append(float(fields[RESIDUAL]))
                    delay.append(float(fields[DELAY])); flagged.append('true' == (fields[FLAGGED]))
                    zeroed.append('true' == (fields[ZEROED])); ha.append(float(fields[HA])); az.append(float(fields[AZ]))
                    el.append(float(fields[EL])); source.append(fields[SOURCE])
#                   flagreason.append('')  # 18th field not yet implemented
                except ValueError:
                    self.logger.warn('Trouble parsing line of telcal file. Skipping.')
                    continue

        self.mjd = np.array(mjd); self.utc = np.array(utc); self.lstd = np.array(lstd); self.lsts = np.array(lsts)
        self.ifid = np.array(ifid); self.skyfreq = np.array(skyfreq); self.antname = np.array(antname); self.amp = np.array(amp) 
        self.phase = np.array(phase); self.residual = np.array(residual); self.delay = np.array(delay)
        self.flagged = np.array(flagged); self.zeroed = np.array(zeroed); self.ha = np.array(ha); self.az = np.array(az)
        self.el = np.array(el); self.source = np.array(source); 
        #self.flagreason = np.array(flagreason)

        # purify list to keep only complete solution sets
        if onlycomplete:
            completecount = len(np.unique(self.ifid)) * len(np.unique(self.antname))
            complete = []
            for mjd in np.unique(self.mjd):
                mjdselect = list(np.where(mjd == self.mjd)[0])
                if len(mjdselect) == completecount:
                    complete = complete + mjdselect
            self.complete = np.array(complete)
        else:
            self.complete = np.arange(len(self.mjd))

        # make another version of ants array
        antnum = []
        for aa in self.antname:
            antnum.append(int(aa[2:]))    # cuts the 'ea' from start of antenna string to get integer
        self.antnum = np.array(antnum)

    def calcgain(self, ant1, ant2, skyfreq, pol):
        """ Calculates the complex gain product (g1*g2) for a pair of antennas.
        """

        select = self.select[np.where((self.skyfreq[self.select] == skyfreq) &
                                      (self.polarization[self.select] == pol))[0]]

        if len(select):  # for when telcal solutions don't exist
            ind1 = np.where(ant1 == self.antnum[select])
            ind2 = np.where(ant2 == self.antnum[select])
            g1 = self.amp[select][ind1]*np.exp(1j*np.radians(self.phase[select][ind1])) * (not self.flagged.astype(int)[select][ind1][0])
            g2 = self.amp[select][ind2]*np.exp(-1j*np.radians(self.phase[select][ind2])) * (not self.flagged.astype(int)[select][ind2][0])
        else:
            g1 = [0]
            g2 = [0]

        try:
            assert (g1[0] != 0j) and (g2[0] != 0j)
            invg1g2 = 1./(g1[0]*g2[0])
        except (AssertionError, IndexError):
            invg1g2 = 0.
        return invg1g2

    def calcdelay(self, ant1, ant2, skyfreq, pol):
        """ Calculates the relative delay (d1-d2) for a pair of antennas in ns.
        """

        select = self.select[np.where( (self.skyfreq[self.select] == skyfreq) & (self.polarization[self.select] == pol) )[0]]

        ind1 = np.where(ant1 == self.antnum[select])
        ind2 = np.where(ant2 == self.antnum[select])
        d1 = self.delay[select][ind1]
        d2 = self.delay[select][ind2]
        if len(d1-d2) > 0:
            return d1-d2
        else:
            return np.array([0])

    def apply(self, data, sign=1):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        Sign defines direction of calibration solution. +1 is correcting, -1 will corrupt.
        """

        # find best skyfreq for each channel
        skyfreqs = np.unique(self.skyfreq[self.select])    # one per spw
        nch_tot = len(self.freqs)
        chan_bandnum = [list(range(nch_tot*i/len(skyfreqs), nch_tot*(i+1)/len(skyfreqs)) for i in range(len(skyfreqs)))]  # divide chans by number of spw in solution
        self.logger.info('Solutions for %d spw: (%s)' % (len(skyfreqs), skyfreqs))

        for j in range(len(skyfreqs)):
            skyfreq = skyfreqs[j]
            chans = chan_bandnum[j]
            self.logger.info('Applying gain solution for chans from %d-%d' % (chans[0], chans[-1]))

            # define freq structure to apply delay solution
            nch = len(chans)
            chanref = nch/2    # reference channel at center
            relfreq = self.chansize*(np.arange(nch) - chanref)   # relative frequency

            for i in range(len(self.blarr)):
                ant1, ant2 = self.blarr[i]  # ant numbers (1-based)
                for pol in self.polind:
                    # apply gain correction
                    invg1g2 = sign*self.calcgain(ant1, ant2, skyfreq, pol)
                    data[:, i, chans, pol-self.polind[0]] = data[:, i, chans,
                                                                 pol-self.polind[0]] * invg1g2    # hack: lousy data pol indexing

                    # apply delay correction
                    d1d2 = sign*self.calcdelay(ant1, ant2, skyfreq, pol)
                    delayrot = 2*np.pi*(d1d2[0] * 1e-9) * relfreq      # phase to rotate across band
                    data[:,i,chans,pol-self.polind[0]] = data[:,i,chans,pol-self.polind[0]] * np.exp(-1j*delayrot[None, None, :])     # do rotation
