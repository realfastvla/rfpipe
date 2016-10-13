from __future__ import print_function, division, absolute_import

import logging
logger = logging.getLogger(__name__)

import os, attr
from lxml.etree import XMLSyntaxError
import numpy as np
import sdmpy
from . import util

import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
me = casautil.tools.measures()

# source.py will:
# - define data sources for pipeline
# - have first rate support for sdm files
# - can generalize to include streaming data from CBE?


@attr.s
class Metadata(object):
    """ Metadata we need to translate parameters into a pipeline state.
    Called from a function defined for a given source (e.g., an sdm file).
    Built from *nominally* immutable attributes and properties.
    To modify metadata, use attr.assoc(inst, key=newval)

    TODO: can we freeze attributes while still having cached values?
    """

    # basics
    filename = attr.ib(default=None)
    scan = attr.ib(default=None)
    bdfdir = attr.ib(default=None)
    bdfstr = attr.ib(default=None)

    # data structure and source properties
    source = attr.ib(default=None)
    radec = attr.ib(default=None)
    inttime = attr.ib(default=None)
    nints = attr.ib(default=None)
    telescope = attr.ib(default=None)

    # array/antenna info
    starttime_mjd = attr.ib(default=None)
    endtime_mjd = attr.ib(default=None)
    dishdiameter = attr.ib(default=None)
    intent = attr.ib(default=None)

    # spectral info
    spw_orig = attr.ib(default=None)
    spw_nchan = attr.ib(default=None)
    spw_reffreq = attr.ib(default=None)
    spw_chansize = attr.ib(default=None)
    pols_orig = attr.ib(default=None)


    def atdefaults(self):
        """ Is metadata still set at default values? """
        return not any([self.__dict__[ab] for ab in self.__dict__])


    @property
    def workdir(self):
        return os.path.dirname(os.path.abspath(self.filename))


    @property
    def spw_chanr(self):
        chanr = []
        i0 = 0
        for nch in self.spw_nchan:
            chanr.append((i0, i0+nch))
            i0 = nch
        return chanr

    @property
    def freq_orig(self):
        """Spacing of channel centers in GHz"""

        return np.array([np.linspace(self.spw_reffreq[ii], self.spw_reffreq[ii] +
                                     (self.spw_nchan[ii]-1) * self.spw_chansize[ii], self.spw_nchan[ii])
                        for ii in range(len((self.spw_reffreq)))], dtype='float32').flatten()/1e9


    @property
    def nchan_orig(self):
        return len(self.freq_orig)


    @property
    def configid(self):
        sdm = getsdm(self.filename, bdfdir=self.bdfdir)
        return [str(row.configDescriptionId) for row in sdm['Main']
                if self.scan == int(row.scanNumber)][0]


    @property
    def antids(self):
        sdm = getsdm(self.filename, bdfdir=self.bdfdir)
        return [str(row.antennaId) for row in sdm['ConfigDescription']
                if self.configid == row.configDescriptionId][0].split(' ')[2:]


    @property
    def ants_orig(self):
        sdm = getsdm(self.filename, bdfdir=self.bdfdir)
        return [int(str(row.name).lstrip('ea')) for antid in self.antids for row in sdm['Antenna'] 
                if antid == str(row.antennaId)]


    @property
    def nants_orig(self):
        return len(self.ants_orig)


    @property
    def nbl_orig(self):
        return self.nants_orig*(self.nants_orig-1)/2


    @property
    def antpos(self):
        sdm = getsdm(self.filename, bdfdir=self.bdfdir)
        logger.debug('Assuming all antennas used.')
        stationidlist = [str(ant.stationId) for ant in sdm['Antenna']]

        positions = [str(station.position).strip().split(' ')
                     for station in sdm['Station'] 
                     if station.stationId in stationidlist]
        x = [float(positions[i][2]) for i in range(len(positions))]
        y = [float(positions[i][3]) for i in range(len(positions))]
        z = [float(positions[i][4]) for i in range(len(positions))]
        ap = me.position('itrf', qa.quantity(x, 'm'), qa.quantity(y, 'm'), qa.quantity(z, 'm'))

        return ap


    @property
    def starttime_string(self):
        return qa.time(qa.quantity(self.starttime_mjd,'d'), form="ymd", prec=8)[0]


    @property
    def uvrange_orig(self):
        if not hasattr(self, '_uvrange_orig'):
            (u, v, w) = util.calc_uvw(datetime=self.starttime_string, radec=self.radec, antpos=self.antpos, telescope=self.telescope)
            u = u * self.freq_orig[0] * (1e9/3e8) * (-1)
            v = v * self.freq_orig[0] * (1e9/3e8) * (-1)
            self._uvrange_orig = (u.max() - u.min(), v.max() - v.min())

        return self._uvrange_orig


    @property
    def npol_orig(self):
        return len(self.pols_orig)


def sdm_metadata(sdmfile, scan, bdfdir=None):
    """ Wraps Metadata call to provide immutable, attribute-filled class instance.
    """

    logger.info('Reading metadata from {0}, scan {1}'.format(sdmfile, scan))

    sdm = getsdm(sdmfile, bdfdir=bdfdir)
    scanobj = sdm.scan(scan)

    sdmmeta = {}
    sdmmeta['filename'] = sdmfile
    sdmmeta['scan'] = scan
    sdmmeta['bdfdir'] = bdfdir

    starttime_mjd = scanobj.bdf.startTime
    nints = scanobj.bdf.numIntegration
    inttime = scanobj.bdf.get_integration(0).interval
    endtime_mjd = starttime_mjd + (nints*inttime)/(24*3600)
    bdfstr = scanobj.bdf.fname

    sdmmeta['starttime_mjd'] = starttime_mjd
    sdmmeta['endtime_mjd'] = endtime_mjd
    sdmmeta['inttime'] = inttime
    sdmmeta['nints'] = nints
    sdmmeta['source'] = str(scanobj.source)
    sdmmeta['intent'] = ' '.join(scanobj.intents)
    sdmmeta['telescope'] = str(sdm['ExecBlock'][0]['telescopeName']).strip()
    bdfstr = scanobj.bdf.fname
    if (not os.path.exists(bdfstr)) or ('X1' in bdfstr):
        sdmmeta['bdfstr'] = None
    else:
        sdmmeta['bdfstr'] = bdfstr

    sources = sdm_sources(sdmfile)
    sdmmeta['radec'] = [(prop['ra'], prop['dec']) for (sr, prop) in sources.iteritems() if prop['source'] == scanobj.source][0]
    sdmmeta['dishdiameter'] = float(str(sdm['Antenna'][0].dishDiameter).strip())

    sdmmeta['spw_orig'] = [int(str(row.spectralWindowId).split('_')[1]) for row in sdm['SpectralWindow']]
    sdmmeta['spw_nchan'] = [int(row.numChan) for row in sdm['SpectralWindow']]
    sdmmeta['spw_reffreq'] = [float(row.chanFreqStart) for row in sdm['SpectralWindow']]
    sdmmeta['spw_chansize'] = [float(row.chanFreqStep) for row in sdm['SpectralWindow']]


    sdmmeta['pols_orig'] = [pol for pol in (str(sdm['Polarization'][0]
                                               .corrType).strip()
                                           .split(' '))
                           if pol in ['XX', 'YY', 'XY', 'YX',
                                      'RR', 'LL', 'RL', 'LR']]

    # any need to overload with provided kw args?
#    for key, value in kwargs.iteritems():
#        sdmmeta[key] = value

    return sdmmeta


def sdm_sources(sdmname):
    """ Use sdmpy to get all sources and ra,dec per scan as dict """

    sdm = getsdm(sdmname)
    sourcedict = {}

    for row in sdm['Field']:
        src = str(row.fieldName)
        sourcenum = int(row.sourceId)
        direction = str(row.referenceDir)
        (ra,dec) = [float(val) for val in direction.split(' ')[3:]]  # skip first two values in string

        sourcedict[sourcenum] = {}
        sourcedict[sourcenum]['source'] = src
        sourcedict[sourcenum]['ra'] = ra
        sourcedict[sourcenum]['dec'] = dec

    return sourcedict


def getsdm(*args, **kwargs):
    """ Wrap sdmpy.SDM to get around schema change error """

    try:
        sdm = sdmpy.SDM(*args, **kwargs)
    except XMLSyntaxError:
        kwargs['use_xsd'] = False
        sdm = sdmpy.SDM(*args, **kwargs)

    return sdm


def dataprep(st, segment):
    data_read = read_bdf_segment(st, segment)
    return data_read


def read_bdf(st, nskip=0):
    """ Uses sdmpy to read a given range of integrations from sdm of given scan.
    readints=0 will read all of bdf (skipping nskip).
    """

    assert os.path.exists(st.metadata.filename), 'sdmfile {0} does not exist'.format(st.metadata.filename)
    assert st.metadata.bdfstr, 'bdfstr not defined for scan {0}'.format(st.metadata.scan)

    logger.info('Reading %d ints starting at int %d' % (st.readints, nskip))
    sdm = getsdm(st.metadata.filename, bdfdir=st.metadata.bdfdir)
    scan = sdm.scan(st.metadata.scan)

    data = np.empty( (st.readints, st.metadata.nbl_orig, st.metadata.nchan_orig, st.metadata.npol_orig), dtype='complex64', order='C')
    data[:] = scan.bdf.get_data(trange=[nskip, nskip+st.readints]).reshape(data.shape)

    return data


def read_bdf_segment(st, segment):
    """ Uses sdmpy to reads bdf (sdm) format data into numpy array in given segment.
    """

    assert segment < st.nsegments, 'segment {0} is too big for nsegments {1}' % (segment, st.nsegments)

    # define integration range
    nskip = (24*3600*(st.segmenttimes[segment, 0] - st.metadata.starttime_mjd) / st.metadata.inttime).astype(int)
    logger.info('Reading scan {0}, segment {1}/{2}, times {3} to {4}'.format(st.metadata.scan, segment, len(st.segmenttimes)-1,
                                                                             qa.time(qa.quantity(st.segmenttimes[segment, 0], 'd'),
                                                                                     form=['hms'], prec=9)[0],
                                                                             qa.time(qa.quantity(st.segmenttimes[segment, 1], 'd'),
                                                                                     form=['hms'], prec=9)[0]))
    data = read_bdf(st, nskip=nskip).astype('complex64')

    # read Flag.xml and apply flags for given ant/time range
    if st.preferences.applyonlineflags:
        raise NotImplementedError

        sdm = getsdm(d['filename'], bdfdir=d['bdfdir'])

        allantdict = dict(zip([str(ant.antennaId) for ant in sdm['Antenna']],
                              [int(str(ant.name).lstrip('ea'))
                               for ant in sdm['Antenna']]))
        antflags = [(allantdict[str(flag.antennaId).split(' ')[2]],
                     int(flag.startTime)/(1e9*24*3600),
                     int(flag.endTime)/(1e9*24*3600))
                    for flag in sdm['Flag']]  # assumes one flag per entry
        logger.info('Found online flags for %d antenna/time ranges.'
                    % (len(antflags)))
        blarr = calc_blarr(d)  # d may define different ants than in allantdict
        timearr = np.linspace(d['segmenttimes'][segment][0],
                              d['segmenttimes'][segment][1], d['readints'])
        badints_cum = []
        for antflag in antflags:
            antnum, time0, time1 = antflag
            badbls = np.where((blarr == antnum).any(axis=1))[0]
            badints = np.where((timearr >= time0) & (timearr <= time1))[0]
            for badint in badints:
                data[badint, badbls] = 0j
            badints_cum = badints_cum + list(badints)
        logger.info('Applied online flags to %d ints.'
                    % (len(set(badints_cum))))
    else:
        logger.info('Not applying online flags.')

    # test that spw are in freq sorted order
    # only one use case supported: rolled spw
    dfreq = np.array([st.metadata.spw_reffreq[i+1] - st.metadata.spw_reffreq[i]
                      for i in range(len(st.metadata.spw_reffreq)-1)])
    dfreqneg = [df for df in dfreq if df < 0]
    # if spw are permuted, then roll them.
    # !! not a perfect test of permutability!!
    if len(dfreqneg) <= 1:
        if len(dfreqneg) == 1:
            logger.info('Rolling spw frequencies to increasing order: %s'
                        % str(st.metadata.spw_reffreq))
            rollch = np.sum([st.metadata.spw_nchan[ss]
                             for ss in range(np.where(dfreq < 0)[0][0]+1)])
            data = np.roll(data, rollch, axis=2)
    else:
        raise StandardError('SPW out of order and can\'t be permuted '
                            'to increasing order: %s'
                            % str(st.metadata.spw_reffreq))

    # optionally integrate (downsample)
    if ((st.preferences.read_tdownsample > 1) or (st.preferences.read_fdownsample > 1)):
        raise NotImplementedError

        sh = data.shape
        tsize = sh[0]/d['read_tdownsample']
        fsize = sh[2]/d['read_fdownsample']
        data2 = np.zeros((tsize, sh[1], fsize, sh[3]), dtype='complex64')
        if d['read_tdownsample'] > 1:
            logger.info('Downsampling in time by %d' % d['read_tdownsample'])
            for i in range(tsize):
                data2[i] = data[
                    i*d['read_tdownsample']:(i+1)*d['read_tdownsample']
                    ].mean(axis=0)
        if d['read_fdownsample'] > 1:
            logger.info('Downsampling in frequency by %d'
                        % d['read_fdownsample'])
            for i in range(fsize):
                data2[:, :, i, :] = data[
                    :, :, i * d['read_fdownsample']:(i+1)*d['read_fdownsample']
                    ].mean(axis=2)
        data = data2

    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
    logger.debug('Selecting pols {0}'.format(st.pols))

    return data.take(st.chans, axis=2).take(takepol, axis=3)

