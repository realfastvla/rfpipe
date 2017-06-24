from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)


def apply_telcal(st, data):
    """ Wrap all telcal functions to parse telcal file and apply it to data
    """

    sols = parseGN(st.gainfile)
    sols = flagants(sols)
    sols = set_selection(sols, st.segmenttimes[segment].mean(), st.freq*1e9, st.blarr, calname=calname, pols=st.pols, radec=radec, spwind=spwind)

#    sols.apply(data)

    return data


def parseGN(telcalfile, onlycomplete=True):
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
#            flagreason.append('')  # 18th field not yet implemented
        except ValueError:
            logger.warn('Trouble parsing line of telcal file. Skipping.')
            continue

    mjd = n.array(mjd); utc = n.array(utc); lstd = n.array(lstd); lsts = n.array(lsts)
    ifid = n.array(ifid); skyfreq = n.array(skyfreq); antname = n.array(antname); amp = n.array(amp) 
    phase = n.array(phase); residual = n.array(residual); delay = n.array(delay)
    flagged = n.array(flagged); zeroed = n.array(zeroed); ha = n.array(ha); az = n.array(az)
    el = n.array(el); source = n.array(source); 
    #flagreason = n.array(flagreason)

    # purify list to keep only complete solution sets
    if onlycomplete:
        completecount = len(n.unique(ifid)) * len(n.unique(antname))
        complete = []
        for mjd in n.unique(mjd):
            mjdselect = list(n.where(mjd == mjd)[0])
            if len(mjdselect) == completecount:
                complete = complete + mjdselect
        complete = n.array(complete)
    else:
        complete = n.arange(len(mjd))

    # make another version of ants array
    antnum = []
    for aa in antname:
        antnum.append(int(aa[2:]))    # cuts the 'ea' from start of antenna string to get integer
    antnum = n.array(antnum)

    sols = {'mjd': mjd, 'utc': utc, 'lstd': lstd, 'lsts': lsts, 'ifid': ifid, 'skyfreq': skyfreq,
            'antname': antname, 'amp': amp, 'phase': phase, 'residual': residual, 'delay': delay,
            'flagged': flagged, 'zeroed': zeroed, 'ha': ha, 'az': az, 'el': el, 'source': source}

    return sols


def flagants(sols, threshold=50):
    """ Flags solutions with amplitude more than threshold larger than median.
    """

    # identify very low gain amps not already flagged
    badsols = n.where( (n.median(sols['amp'])/sols['amp'] > threshold) & (sols['flagged'] == False))[0]
    if len(badsols):
        logger.info('Solutions %s flagged (times %s, ants %s, freqs %s) for low gain amplitude.' % (str(badsols), sols['mjd'][badsols], sols['antname'][badsols], sols['ifid'][badsols]))
        for sol in badsols:
            sols['flagged'][sol] = True

    return sols


def set_selection(sols, time, freqs, blarr):
    """ Set select parameter that defines spectral window, time, or any other selection.
    time (in mjd) defines the time to find solutions near for given calname.
    freqs (in Hz) is frequencies in data.
    blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
    calname defines the name of the calibrator to use. if blank, uses only the time selection.
    radec, dist, spwind not used. here for uniformity with casa_sol.
    """

    sols['freqs'] = freqs
    sols['chansize'] = freqs[1]-freqs[0]
    sols['select'] = sols['complete']   # use only complete solution sets (set during parse)
    sols['blarr'] = blarr
    if spwind:
        logger.warn('spwind option not used for telcal_sol. Applied based on freqs.')
    if radec:
        logger.warn('radec option not used for telcal_sol. Applied based on calname.')
    if dist:
        logger.warn('dist option not used for telcal_sol. Applied based on calname.')

    if calname:
        nameselect = []
        for ss in n.unique(sols['source'][sols['select']]):
            if calname in ss:
                nameselect = n.where(sols['source'][sols['select']] == ss)   # define selection for name
                sols['select'] = sols['select'][nameselect]       # update overall selection
                logger.debug('Selection down to %d solutions with %s' % (len(sols['select']), calname))
        if not nameselect:
            logger.warn('Calibrator name %s not found. Ignoring.' % (calname))

    # select freq
    freqselect = n.where([ff in n.around(sols['freqs'], -6) for ff in n.around(1e6*sols['skyfreq'][sols['select']], -6)])   # takes solution if band center is in (rounded) array of chan freqs
    if len(freqselect[0]) == 0:
        raise StandardError('No complete set of telcal solutions at that frequency.')
    sols['select'] = sols['select'][freqselect[0]]    # update overall selection
    logger.info('Frequency selection cut down to %d solutions' % (len(sols['select'])))

    # select by smallest time distance for source
    mjddist = n.abs(time - n.unique(sols['mjd'][sols['select']]))
    closest = n.where(mjddist == mjddist.min())
    if len(closest[0]) > 1:
        logger.info('Multiple closest solutions in time (%s). Taking first.' % (str(closest[0])))
        closest = closest[0][0]
    timeselect = n.where(sols['mjd'][sols['select']] == n.unique(sols['mjd'][sols['select']])[closest])   # define selection for time
    sols['select'] = sols['select'][timeselect[0]]    # update overall selection
    logger.info('Selection down to %d solutions separated from given time by %d minutes' % (len(sols['select']), mjddist[closest]*24*60))

    logger.debug('Selected solutions: %s' % str(sols['select']))
    logger.info('MJD: %s' % str(n.unique(sols['mjd'][sols['select']])))
    logger.debug('Mid frequency (MHz): %s' % str(n.unique(sols['skyfreq'][sols['select']])))
    logger.debug('IFID: %s' % str(n.unique(sols['ifid'][sols['select']])))
    logger.info('Source: %s' % str(n.unique(sols['source'][sols['select']])))
    logger.debug('Ants: %s' % str(n.unique(sols['antname'][sols['select']])))


def apply(sols, data):
    """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
    """

    # find best skyfreq for each channel
    skyfreqs = n.unique(sols['skyfreq'][sols['select']])    # one per spw
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
        relfreq = sols['chansize']*(n.arange(nch) - chanref)   # relative frequency

        for i in range(len(sols['blarr'])):
            ant1, ant2 = sols['blarr'][i]  # ant numbers (1-based)
            for pol in sols['polind']:
                # apply gain correction
                invg1g2 = calcgain(sols, ant1, ant2, skyfreq, pol)
                data[:,i,chans,pol-sols['polind'][0]] = data[:,i,chans,pol-sols['polind'][0]] * invg1g2    # hack: lousy data pol indexing

                # apply delay correction
                d1d2 = calcdelay(sols, ant1, ant2, skyfreq, pol)
                delayrot = 2*n.pi*(d1d2[0] * 1e-9) * relfreq      # phase to rotate across band
                data[:,i,chans,pol-sols['polind'][0]] = data[:,i,chans,pol-sols['polind'][0]] * n.exp(-1j*delayrot[None, None, :])     # do rotation


def calcgain(sols, ant1, ant2, skyfreq, pol):
    """ Calculates the complex gain product (g1*g2) for a pair of antennas.
    """

    select = sols['select'][n.where( (sols['skyfreq'][sols['select']] == skyfreq) & (sols['polarization'][sols['select']] == pol) )[0]]

    if len(select):  # for when telcal solutions don't exist
        ind1 = n.where(ant1 == sols['antnum'][select])
        ind2 = n.where(ant2 == sols['antnum'][select])
        g1 = sols['amp'][select][ind1]*n.exp(1j*n.radians(sols['phase'][select][ind1])) * (not sols['flagged'].astype(int)[select][ind1][0])
        g2 = sols['amp'][select][ind2]*n.exp(-1j*n.radians(sols['phase'][select][ind2])) * (not sols['flagged'].astype(int)[select][ind2][0])
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

    select = sols['select'][n.where( (sols['skyfreq'][sols['select']] == skyfreq) & (sols['polarization'][sols['select']] == pol) )[0]]
    
    ind1 = n.where(ant1 == sols['antnum'][select])
    ind2 = n.where(ant2 == sols['antnum'][select])
    d1 = sols['delay'][select][ind1]
    d2 = sols['delay'][select][ind2]
    if len(d1-d2) > 0:
        return d1-d2
    else:
        return n.array([0])
