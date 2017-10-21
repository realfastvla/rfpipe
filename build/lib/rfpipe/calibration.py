from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import numpy as np
import os.path

import logging
logger = logging.getLogger(__name__)


def apply_telcal(st, data, calname=None):
    """ Wrap all telcal functions to parse telcal file and apply it to data
    """

    sols = telcal_sol(st.gainfile)
    sols.set_selection(st.segmenttimes.mean(), st.freq*1e9, st.blarr,
                       calname=calname)
    sols.apply(data)


class telcal_sol():
    """ Instantiated with on telcalfile.
    Parses .GN file and provides tools for applying to data of shape (nints,
    nbl, nch, npol)
    """

    def __init__(self, telcalfile, flagants=True):
        self.logger = logging.getLogger(__name__)

        if os.path.exists(telcalfile):
            self.parseGN(telcalfile)
            self.logger.info('Read telcalfile %s' % telcalfile)
            if flagants:
                self.flagants()
        else:
            self.logger.warn('Gainfile not found.')
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
            raise StandardError('No complete set of telcal solutions at that frequency.')
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
        for line in open(telcalfile,'r'):

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
#                flagreason.append('')  # 18th field not yet implemented
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

        select = self.select[np.where( (self.skyfreq[self.select] == skyfreq) & (self.polarization[self.select] == pol) )[0]]

        if len(select):  # for when telcal solutions don't exist
            ind1 = np.where(ant1 == self.antnum[select])
            ind2 = np.where(ant2 == self.antnum[select])
            g1 = self.amp[select][ind1]*np.exp(1j*np.radians(self.phase[select][ind1])) * (not self.flagged.astype(int)[select][ind1][0])
            g2 = self.amp[select][ind2]*np.exp(-1j*np.radians(self.phase[select][ind2])) * (not self.flagged.astype(int)[select][ind2][0])
        else:
            g1 = [0]; g2 = [0]

        try:
            assert (g1[0] != 0j) and (g2[0] != 0j)
            invg1g2 = 1./(g1[0]*g2[0])
        except (AssertionError, IndexError):
            invg1g2 = 0
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

    def apply(self, data):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        """

        # find best skyfreq for each channel
        skyfreqs = np.unique(self.skyfreq[self.select])    # one per spw
        nch_tot = len(self.freqs)
        chan_bandnum = [range(nch_tot*i/len(skyfreqs), nch_tot*(i+1)/len(skyfreqs)) for i in range(len(skyfreqs))]  # divide chans by number of spw in solution
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
                    invg1g2 = self.calcgain(ant1, ant2, skyfreq, pol)
                    data[:,i,chans,pol-self.polind[0]] = data[:,i,chans,pol-self.polind[0]] * invg1g2    # hack: lousy data pol indexing

                    # apply delay correction
                    d1d2 = self.calcdelay(ant1, ant2, skyfreq, pol)
                    delayrot = 2*np.pi*(d1d2[0] * 1e-9) * relfreq      # phase to rotate across band
                    data[:,i,chans,pol-self.polind[0]] = data[:,i,chans,pol-self.polind[0]] * np.exp(-1j*delayrot[None, None, :])     # do rotation


###
### Trying to clean up and put in functions. Not there yet...
###


def parseGN(telcalfile, onlycomplete=True, threshold=50):
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
    for line in open(telcalfile,'r'):

        fields = line.split()
        if i < skip:
            i += 1
            continue

        try:
            mjd.append(float(fields[MJD])); utc.append(fields[UTC]); lstd.append(float(fields[LSTD])); lsts.append(fields[LSTS])
            ifid.append(fields[IFID]); skyfreq.append(float(fields[SKYFREQ])); antname.append(fields[ANT])
            amp.append(float(fields[AMP])); phase.append(float(fields[PHASE])); residual.append(float(fields[RESIDUAL]))
            delay.append(float(fields[DELAY])); flagged.append('true' == (fields[FLAGGED]))
            zeroed.append('true' == (fields[ZEROED])); ha.append(float(fields[HA])); az.append(float(fields[AZ]))
            el.append(float(fields[EL])); source.append(fields[SOURCE])
#            flagreasonp.append('')  # 18th field not yet implemented
        except ValueError:
            logger.warn('Trouble parsing line of telcal file. Skipping.')
            continue

    mjd = np.array(mjd); utc = np.array(utc); lstd = np.array(lstd); lsts = np.array(lsts)
    ifid = np.array(ifid); skyfreq = np.array(skyfreq); antname = np.array(antname); amp = np.array(amp) 
    phase = np.array(phase); residual = np.array(residual); delay = np.array(delay)
    flagged = np.array(flagged); zeroed = np.array(zeroed); ha = np.array(ha); az = np.array(az)
    el = np.array(el); source = np.array(source); 
    #flagreason = np.array(flagreason)

    # purify list to keep only complete solution sets
    if onlycomplete:
        completecount = len(np.unique(ifid)) * len(np.unique(antname))
        complete = []
        for mjd0 in np.unique(mjd):
            mjdselect = list(np.where(mjd0 == mjd)[0])
            if len(mjdselect) == completecount:
                complete = complete + mjdselect
        complete = np.array(complete)
    else:
        complete = np.arange(len(mjd))

    antnum = np.array([aa.lstrip('ea') for aa in antname]) # cut the 'ea' from start of antenna string to get integer

    sols = {'mjd': mjd, 'utc': utc, 'lstd': lstd, 'lsts': lsts, 'ifid': ifid, 'skyfreq': skyfreq,
            'antname': antname, 'amp': amp, 'phase': phase, 'residual': residual, 'delay': delay,
            'flagged': flagged, 'zeroed': zeroed, 'ha': ha, 'az': az, 'el': el, 'source': source,
            'complete': complete}

    flagants(sols, threshold=threshold)

    return sols


def flagants(sols, threshold=50):
    """ Flags solutions with amplitude more than threshold larger than median.
    """

    # identify very low gain amps not already flagged
    badsols = np.where( (np.median(sols['amp'])/sols['amp'] > threshold) & (sols['flagged'] == False))[0]
    if len(badsols):
        logger.info('Solutions %s flagged (times %s, ants %s, freqs %s) for low gain amplitude.' % (str(badsols), sols['mjd'][badsols], sols['antname'][badsols], sols['ifid'][badsols]))
        for sol in badsols:
            sols['flagged'][sol] = True


def set_selection(sols, time, freqs, blarr, calname=None):
    """ Set select parameter that defines spectral window, time, or any other selection.
    time (in mjd) defines the time to find solutions near for given calname.
    freqs (in Hz) is frequencies in data.
    blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
    calname defines the name of the calibrator to use. if blank, uses only the time selection.
    """

    sols['freqs'] = freqs
    sols['chansize'] = freqs[1]-freqs[0]
    sols['select'] = sols['complete']   # use only complete solution sets (set during parse)
    sols['blarr'] = blarr

    if calname:
        nameselect = []
        for ss in np.unique(sols['source'][sols['select']]):
            if calname in ss:
                nameselect = np.where(sols['source'][sols['select']] == ss)   # define selection for name
                sols['select'] = sols['select'][nameselect]       # update overall selection
                logger.debug('Selection down to %d solutions with %s' % (len(sols['select']), calname))
        if not nameselect:
            logger.warn('Calibrator name %s not found. Ignoring.' % (calname))

    # select freq
    freqselect = np.where([ff in np.around(sols['freqs'], -6) for ff in np.around(1e6*sols['skyfreq'][sols['select']], -6)])   # takes solution if band center is in (rounded) array of chan freqs
    if len(freqselect[0]) == 0:
        raise StandardError('No complete set of telcal solutions at that frequency.')
    sols['select'] = sols['select'][freqselect[0]]    # update overall selection
    logger.info('Frequency selection cut down to %d solutions' % (len(sols['select'])))

    # select by smallest time distance for source
    mjddist = np.abs(time - np.unique(sols['mjd'][sols['select']]))
    closest = np.where(mjddist == mjddist.min())
    if len(closest[0]) > 1:
        logger.info('Multiple closest solutions in time (%s). Taking first.' % (str(closest[0])))
        closest = closest[0][0]
    timeselect = np.where(sols['mjd'][sols['select']] == np.unique(sols['mjd'][sols['select']])[closest])   # define selection for time
    sols['select'] = sols['select'][timeselect[0]]    # update overall selection
    logger.info('Selection down to %d solutions separated from given time by %d minutes' % (len(sols['select']), mjddist[closest]*24*60))

    logger.debug('Selected solutions: %s' % str(sols['select']))
    logger.info('MJD: %s' % str(np.unique(sols['mjd'][sols['select']])))
    logger.debug('Mid frequency (MHz): %s' % str(np.unique(sols['skyfreq'][sols['select']])))
    logger.debug('IFID: %s' % str(np.unique(sols['ifid'][sols['select']])))
    logger.info('Source: %s' % str(np.unique(sols['source'][sols['select']])))
    logger.debug('Ants: %s' % str(np.unique(sols['antname'][sols['select']])))


def apply(sols, data):
    """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
    """

    # find best skyfreq for each channel
    skyfreqs = np.unique(sols['skyfreq'][sols['select']])    # one per spw
    nch_tot = len(sols['freqs'])
    chan_bandnum = [range(nch_tot*i/len(skyfreqs), nch_tot*(i+1)/len(skyfreqs)) for i in range(len(skyfreqs))]  # divide chans by number of spw in solution
    logger.info('Solutions for %d spw: (%s)' % (len(skyfreqs), skyfreqs))

    for j in range(len(skyfreqs)):
        skyfreq = skyfreqs[j]
        chans = chan_bandnum[j]
        logger.info('Applying gain solution for chans from %d-%d' % (chans[0], chans[-1]))

        # define freq structure to apply delay solution
        nch = len(chans)
        chanref = nch/2    # reference channel at center
        relfreq = sols['chansize']*(np.arange(nch) - chanref)   # relative frequency

        for i in range(len(sols['blarr'])):
            ant1, ant2 = sols['blarr'][i]  # ant numbers (1-based)
            for pol in sols['polind']:
                # apply gain correction
                invg1g2 = calcgain(sols, ant1, ant2, skyfreq, pol)
                data[:,i,chans,pol-sols['polind'][0]] = data[:,i,chans,pol-sols['polind'][0]] * invg1g2    # hack: lousy data pol indexing

                # apply delay correction
                d1d2 = calcdelay(sols, ant1, ant2, skyfreq, pol)
                delayrot = 2*np.pi*(d1d2[0] * 1e-9) * relfreq      # phase to rotate across band
                data[:,i,chans,pol-sols['polind'][0]] = data[:,i,chans,pol-sols['polind'][0]] * np.exp(-1j*delayrot[None, None, :])     # do rotation


def calcgain(sols, ant1, ant2, skyfreq, pol):
    """ Calculates the complex gain product (g1*g2) for a pair of antennas.
    """

    select = sols['select'][np.where( (sols['skyfreq'][sols['select']] == skyfreq) & (sols['polarization'][sols['select']] == pol) )[0]]

    if len(select):  # for when telcal solutions don't exist
        ind1 = np.where(ant1 == sols['antnum'][select])
        ind2 = np.where(ant2 == sols['antnum'][select])
        g1 = sols['amp'][select][ind1]*np.exp(1j*np.radians(sols['phase'][select][ind1])) * (not sols['flagged'].astype(int)[select][ind1][0])
        g2 = sols['amp'][select][ind2]*np.exp(-1j*np.radians(sols['phase'][select][ind2])) * (not sols['flagged'].astype(int)[select][ind2][0])
    else:
        g1 = [0]; g2 = [0]
        
    try:
        assert (g1[0] != 0j) and (g2[0] != 0j)
        invg1g2 = 1./(g1[0]*g2[0])
    except (AssertionError, IndexError):
        invg1g2 = 0

    return invg1g2


def calcdelay(sols, ant1, ant2, skyfreq, pol):
    """ Calculates the relative delay (d1-d2) for a pair of antennas in ns.
    """

    select = sols['select'][np.where( (sols['skyfreq'][sols['select']] == skyfreq) & (sols['polarization'][sols['select']] == pol) )[0]]
    
    ind1 = np.where(ant1 == sols['antnum'][select])
    ind2 = np.where(ant2 == sols['antnum'][select])
    d1 = sols['delay'][select][ind1]
    d2 = sols['delay'][select][ind2]
    if len(d1-d2) > 0:
        return d1-d2
    else:
        return np.array([0])
