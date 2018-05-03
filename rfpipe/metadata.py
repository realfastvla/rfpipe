from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import os.path
import attr

import numpy as np
from rfpipe import source, util
import pwkit.environments.casa.util as casautil
import sdmpy

import logging
logger = logging.getLogger(__name__)

qa = casautil.tools.quanta()
me = casautil.tools.measures()


@attr.s
class Metadata(object):
    """ Metadata we need to translate parameters into a pipeline state.
    Called from a function defined for a given source (e.g., an sdm file).
    Built from nominally immutable attributes and properties.
    To modify metadata, use attr.assoc(inst, key=newval)
    """

    # basics
    datasource = attr.ib(default=None)
    datasetId = attr.ib(default=None)
    filename = attr.ib(default=None)  # full path to SDM (optional)
    scan = attr.ib(default=None)  # int
    subscan = attr.ib(default=None)  # int
    bdfdir = attr.ib(default=None)
    bdfstr = attr.ib(default=None)

    # data structure and source properties
    source = attr.ib(default=None)
    radec = attr.ib(default=None)  # (radians, radians)
    inttime = attr.ib(default=None)  # seconds
    nints_ = attr.ib(default=None)
    telescope = attr.ib(default=None)

    # array/antenna info
    starttime_mjd = attr.ib(default=None)  # float
    endtime_mjd_ = attr.ib(default=None)
    dishdiameter = attr.ib(default=None)
    intent = attr.ib(default=None)
    antids = attr.ib(default=None)
    stationids = attr.ib(default=None) # needed?
    xyz = attr.ib(default=None)  # in m, geocentric

    # spectral info
    spworder = attr.ib(default=None)
    spw_orig = attr.ib(default=None)  # indexes for spw
    spw_nchan = attr.ib(default=None)  # channels per spw
    spw_reffreq = attr.ib(default=None)  # reference frequency (ch0) in Hz
    spw_chansize = attr.ib(default=None)  # channel size in Hz
    pols_orig = attr.ib(default=None)

    def atdefaults(self):
        """ Is metadata still set at default values? """
        return not any([self.__dict__[ab] for ab in self.__dict__])

#    @property
#    def spw_chanr(self):
#        chanr = []
#        i0 = 0
#        for nch in self.spw_nchan:
#            chanr.append((i0, i0+nch))
#            i0 = nch
#        return chanr

    @property
    def freq_orig(self):
        """Spacing of channel centers in GHz.
        Out of order metadata order is sorted in state/data reading"""

        return np.concatenate([np.linspace(self.spw_reffreq[ii],
                                           self.spw_reffreq[ii] +
                                           (self.spw_nchan[ii]-1) *
                                           self.spw_chansize[ii],
                                           self.spw_nchan[ii])
                              for ii in range(len((self.spw_reffreq)))]).astype('float32')/1e9

    @property
    def nspw_orig(self):
        return len(self.spw_reffreq)

    @property
    def nchan_orig(self):
        return len(self.freq_orig)

    @property
    def nants_orig(self):
        return len(self.antids)

    @property
    def nbl_orig(self):
        return int(self.nants_orig*(self.nants_orig-1)/2)

    @property
    def antpos(self):
        x = self.xyz[:, 0].tolist()
        y = self.xyz[:, 1].tolist()
        z = self.xyz[:, 2].tolist()
        return me.position('itrf', qa.quantity(x, 'm'),
                           qa.quantity(y, 'm'), qa.quantity(z, 'm'))

    @property
    def starttime_string(self):
        return qa.time(qa.quantity(self.starttime_mjd, 'd'),
                       form='ymd', prec=8)[0]

    @property
    def endtime_mjd(self):
        """ If nints\_ is defined (e.g., for SDM data), then endtime_mjd is calculated.
        Otherwise (e.g., for scan_config/vys data), it looks for endtime_mjd\_
        attribute
        """

        if self.endtime_mjd_:
            return self.endtime_mjd_
        elif self.nints_:
            assert self.nints_ > 0, "nints_ must be greater than zero"
            return self.starttime_mjd + (self.nints_*self.inttime)/(24*3600)
        else:
            raise AttributeError("Either endtime_mjd_ or nints_ need to be "
                                 "defined.")

    @property
    def nints(self):
        """ If endtime_mjd\_ is defined (e.g., for scan_config/vys
        data), then endtime_mjd is calculated.
        Otherwise (e.g., for SDM data), it looks for nints\_ attribute
        """

        if self.nints_:
            return self.nints_
        elif self.endtime_mjd_:
            assert self.endtime_mjd > self.starttime_mjd, "endtime_mjd must be larger than starttime_mjd"
            return np.round((self.endtime_mjd_ - self.starttime_mjd)*(24*3600)/self.inttime).astype(int)
        else:
            raise AttributeError("Either endtime_mjd_ or nints_ need to be "
                                 "defined.")

    @property
    def uvrange_orig(self):
        (ur, vr, wr) = util.calc_uvw(datetime=self.starttime_string,
                                     radec=self.radec,
                                     antpos=self.antpos,
                                     telescope=self.telescope)
        u = ur * self.freq_orig.min() * (1e9/3e8) * (-1)
        v = vr * self.freq_orig.min() * (1e9/3e8) * (-1)

        return (u.max() - u.min(), v.max() - v.min())

    @property
    def npol_orig(self):
        return len(self.pols_orig)

    @property
    def scanId(self):
        assert self.datasetId is not None
        return '{0}.{1}.{2}'.format(self.datasetId, self.scan, self.subscan)


def config_metadata(config, datasource='vys'):
    """ Creates dict holding metadata from evla_mcast scan config object.
    Parallel structure to sdm_metadata, so this inherits some of its
    nomenclature. datasource defines expected data source (vys expected when
    using scan config)
    spworder is required for proper indexing of vys data.
    """

    logger.info('Reading metadata from config object')

    meta = {}
    meta['datasource'] = datasource
    meta['datasetId'] = config.datasetId
    meta['scan'] = config.scanNo
    meta['subscan'] = config.subscanNo
#    meta['configid'] = config.Id

    meta['starttime_mjd'] = config.startTime
    meta['endtime_mjd_'] = config.stopTime
    meta['source'] = str(config.source)
    meta['intent'] = str(config.scan_intent)
    meta['telescope'] = str(config.telescope)
    antennas = config.get_antennas()
    meta['antids'] = [str(ant.name) for ant in antennas]
    meta['stationids'] = config.listOfStations  # TODO: check type
    meta['xyz'] = np.array([ant.xyz for ant in antennas])

    meta['radec'] = (np.radians(config.ra_deg), np.radians(config.dec_deg))
    meta['dishdiameter'] = 25.  # ?

    subbands = config.get_subbands()
    subband0 = subbands[0]  # **parsing single subband for now
    meta['inttime'] = subband0.hw_time_res  # assumes vys stream post-hw-integ
    meta['pols_orig'] = subband0.pp
    meta['spw_nchan'] = [sb.spectralChannels for sb in subbands]
    meta['spw_chansize'] = [1e6*sb.bw/subband0.spectralChannels for sb in subbands]
    meta['spw_orig'] = ['{0}-{1}'.format(sb.IFid, sb.sbid) for sb in subbands]
    meta['spw_reffreq'] = [(sb.sky_center_freq-sb.bw/subband0.spectralChannels*(sb.spectralChannels/2))*1e6 for sb in subbands]
    meta['spworder'] = sorted([('{0}-{1}'.format(sb.IFid, sb.sbid),
                                meta['spw_reffreq'][subbands.index(sb)])
                               for sb in subbands], key=lambda x: x[1])

    return meta


def sdm_metadata(sdmfile, scan, bdfdir=None):
    """ Wraps Metadata call to provide immutable, attribute-filled class instance.
    """

    logger.info('Reading metadata from {0}, scan {1}'.format(sdmfile, scan))

    sdm = source.getsdm(sdmfile, bdfdir=bdfdir)
    scanobj = sdm.scan(scan)

    meta = {}
    meta['datasource'] = 'sdm'
    meta['datasetId'] = os.path.basename(sdmfile.rstrip('/'))
    meta['filename'] = sdmfile
    meta['scan'] = int(scan)
    meta['subscan'] = 1  # TODO: update for more than one subscan per scan
    meta['bdfdir'] = bdfdir
#    meta['configid'] = scanobj.configDescriptionId
    bdfstr = scanobj.bdf.fname
    if (not os.path.exists(bdfstr)) or ('X1' in bdfstr):
        meta['bdfstr'] = None
    else:
        meta['bdfstr'] = bdfstr

    meta['starttime_mjd'] = scanobj.startMJD
    meta['nints_'] = scanobj.numIntegration

    try:
        inttime = scanobj.bdf.get_integration(0).interval
        meta['inttime'] = inttime
    except AttributeError:
        logger.warn("No BDF found. inttime not set.")

    meta['source'] = str(scanobj.source)
    meta['intent'] = ' '.join(scanobj.intents)
    meta['telescope'] = str(sdm['ExecBlock'][0]['telescopeName']).strip()
    meta['antids'] = [str(ant) for ant in scanobj.antennas]
    meta['stationids'] = [str(station) for station in scanobj.stations]
    meta['xyz'] = np.array(scanobj.positions)

    meta['radec'] = scanobj.coordinates.tolist()
    meta['dishdiameter'] = float(str(sdm['Antenna'][0].dishDiameter).strip())
    meta['spw_orig'] = [int(str(spw).split('_')[1]) for spw in scanobj.spws]
    meta['spw_nchan'] = scanobj.numchans
    meta['spw_reffreq'] = scanobj.reffreqs
    meta['spw_chansize'] = scanobj.chanwidths
    try:
        meta['pols_orig'] = scanobj.bdf.spws[0].pols('cross')
    except AttributeError:
        logger.warn("No BDF found. Inferring pols from xml.")
        meta['pols_orig'] = [pol for pol in (str(sdm['Polarization'][0]
                                                 .corrType)).strip().split(' ')
                             if pol in ['XX', 'YY', 'XY', 'YX',
                                        'RR', 'LL', 'RL', 'LR',
                                        'A*A', 'A*B', 'B*A', 'B*B']]
    try:
        # TODO: remove for datasource=vys or sim?
        meta['spworder'] = sorted(zip(['{0}-{1}'.format(spw.swbb.rstrip('_8BIT'),
                                                        spw.sw-1)
                                       for spw in scanobj.bdf.spws],
                                      np.array(scanobj.reffreqs)/1e6),
                                  key=lambda x: x[1])
    except AttributeError:
        logger.warn("No BDF found. spworder not defined.")

    return meta


def getfirstscan(sdmfile):
    """ Function that returns first scan with bdf
    """

    sdm = sdmpy.SDM(sdmfile)
    scannum = 1
    while True:
        try:
            if sdm.scan(scannum).bdf.exists:
                break
            else:
                scannum += 1
        except KeyError:
            logger.info("Reached last scan but found no bdf.")
            break

    return scannum


def mock_metadata(t0, t1, nants, nspw, chans, npol, inttime_micros, scan=1,
                  subscan=1, datasource='vys', datasetid=None, **kwargs):
    """ Wraps Metadata call to provide immutable, attribute-filled class instance.
    Parallel structure to sdm_metadata, so this inherits some of its
    nomenclature. t0, t1 are times in mjd. Supports up to nant=27, npol=4, and
    nspw=32. chans is total number of channels over all spw (equal per spw).
    datasource is expected source of data (typically vys when mocking).
    datasetid default is "test_<t0>".
    """

    logger.info('Generating mock metadata')

    meta = {}
    meta['datasource'] = datasource
    if datasetid is None:
        datasetid = 'test_{0}'.format(t0)
    meta['datasetId'] = datasetid
    meta['scan'] = scan
    meta['subscan'] = subscan
    meta['bdfdir'] = ''

    meta['starttime_mjd'] = t0
    meta['endtime_mjd_'] = t1
    meta['inttime'] = inttime_micros/1e6
    meta['source'] = 'testsource'
    meta['intent'] = 'OBSERVE_TARGET'
    meta['telescope'] = 'VLA'
    meta['antids'] = ['ea{0:02}'.format(ant) for ant in range(1, nants+1)]
    meta['stationids'] = ['W24', 'W04', 'W28', 'E04', 'E36', 'N12', 'N16',
                          'W12', 'N28', 'E20', 'N20', 'E28', 'E08', 'N24',
                          'W20', 'N04', 'W36', 'W16', 'E16', 'E24', 'W32',
                          'E12', 'W08', 'N08', 'E32', 'N36', 'N32'][:nants]
    meta['xyz'] = np.array([[-1604008.7444, -5042135.8251,  3553403.7108],
                            [-1601315.9005, -5041985.30747,  3554808.311],
                            [-1604865.6575, -5042190.032,  3552962.3635],
                            [-1601068.806, -5042051.9327,  3554824.8388],
                            [-1596127.7308, -5045193.7421,  3552652.4197],
                            [-1601110.022, -5041488.0826,  3555597.4446],
                            [-1601061.9544, -5041175.8753,  3556058.0267],
                            [-1602044.9123, -5042025.8014,  3554427.8357],
                            [-1600863.6922, -5039885.3167,  3557965.3178],
                            [-1599340.8001, -5043150.963,  3554065.2315],
                            [-1601004.6988, -5040802.801,  3556610.1493],
                            [-1597899.8959, -5044068.6847,  3553432.4502],
                            [-1600801.9314, -5042219.3826,  3554706.4294],
                            [-1600930.0836, -5040316.3864,  3557330.39],
                            [-1603249.6721, -5042091.4281,  3553797.7842],
                            [-1601173.9647, -5041902.6458,  3554987.5342],
                            [-1606841.961, -5042279.6752,  3551913.0214],
                            [-1602592.8535, -5042054.9924,  3554140.7028],
                            [-1599926.1041, -5042772.9772,  3554319.8011],
                            [-1598663.082, -5043581.3912,  3553767.0141],
                            [-1605808.6341, -5042230.084,  3552459.1978],
                            [-1600416.518, -5042462.4305,  3554536.0417],
                            [-1601614.0832, -5042001.6569,  3554652.5059],
                            [-1601147.9425, -5041733.8336,  3555235.947],
                            [-1597053.1244, -5044604.675,  3553058.9927],
                            [-1600690.6125, -5038758.7161,  3559632.0571],
                            [-1600781.0607, -5039347.4391,  3558761.5271]])[:nants]
    meta['radec'] = [0., 0.]
    meta['dishdiameter'] = 25
    meta['spw_orig'] = list(range(nspw))
    meta['spw_reffreq'] = np.linspace(2e9, 4e9, 33)[:nspw]
    meta['spw_chansize'] = [2000000]*nspw
    chanperspw = chans//nspw
    meta['spw_nchan'] = [chanperspw]*nspw
    if npol == 4:
        meta['pols_orig'] = ['A*A', 'A*B', 'B*A', 'B*B']
    elif npol == 2:
        meta['pols_orig'] = ['A*A', 'B*B']
    else:
        logger.warn("npol must be 2 or 4 (autos or full pol)")
    meta['spworder'] = sorted([('{0}-{1}'.format('AC1', sbid),
                                meta['spw_reffreq'][sbid])
                               for sbid in range(nspw)], key=lambda x: x[1])

    return meta


def oldstate_metadata(d, scan=None, bdfdir=None):
    """ Parses old state function ("d", a dictionary) into new metadata instance
    Note: d from merged candidate file will have some parameters defined by
    last scan.
    If scan is None, it will assume d is from single scan, not merged
    """

    meta = {}
    meta['datasource'] = 'sdm'
    meta['datasetId'] = d['fileroot']
    meta['subscan'] = 1
    meta['bdfdir'] = bdfdir

    if scan is not None:
        meta['starttime_mjd'] = d['starttime_mjddict'][scan]
        meta['scan'] = scan
    else:
        meta['starttime_mjd'] = d['starttime_mjd']
        meta['scan'] = d['scan']

    meta['inttime'] = d['inttime']
    meta['nints_'] = d['nints']

    meta['source'] = str(d['source'])
    meta['telescope'] = 'VLA'
    meta['antids'] = ['ea'+str(ant) for ant in d['ants']]  # ** test that these are the same as what we expected with rtpipe **
#    meta['xyz'] = #

    meta['radec'] = d['radec']  # ** for last scan! **
    meta['dishdiameter'] = d['dishdiameter']
    meta['spw_orig'] = d['spw_orig']
    meta['spw_nchan'] = d['spw_nchan']
    meta['spw_reffreq'] = d['spw_reffreq']
    meta['spw_chansize'] = d['spw_chansize']

    meta['pols_orig'] = d['pols_orig']

    logger.info('Read metadata from old state dictionary for scan {0}'
                .format(scan))

    return meta
