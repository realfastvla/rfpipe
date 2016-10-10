import logging
logger = logging.getLogger(__name__)

import os, attr
from lxml.etree import XMLSyntaxError
import numpy as np
import rtpipe, sdmpy

# source.py will:
# - define data sources for pipeline
# - have first rate support for sdm files
# - can generalize to include streaming data from CBE?


@attr.s(frozen=True)
class Metadata(object):
    """ Metadata we need to translate parameters into a pipeline state.
    Called from a function defined for a given source (e.g., an sdm file).
    Built from immutable attributes and properties.
    To modify metadata, use attr.assoc(inst, key=newval)
    """

    # basics
    filename = attr.ib()
    scan = attr.ib()
    bdfdir = attr.ib(default=None)
    bdfstr = attr.ib(default=None)
    _sdm = None  # cached sdmpy sdm object

    # data structure and source properties
    source = attr.ib(default=None)
    radec = attr.ib(default=None)
    inttime = attr.ib(default=None)
    nints = attr.ib(default=None)

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


    @property
    def workdir(self):
        return os.path.dirname(os.path.abspath(filename))


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

        return np.array([list(np.linspace(self.reffreq[ii], self.reffreq[ii] + (self.spw_nchan[ii]-1) * self.spw_chansize[ii], self.spw_numchan[ii])) 
                         for ii in range(len((self.spw_reffreq)))], dtype='float32')/1e9


    @property
    def sdm(self):
        """ returns sdmpy sdm object with caching """

        if not self._sdm:
            self._sdm = sdmpy.SDM(self.filename)

        return self._sdm


    @property
    def configid(self):
        return [str(row.configDescriptionId) for row in self.sdm['Main']
                if self.scan == int(row.scanNumber)][0]


    @property
    def antids(self):
        return [str(row.antennaId) for row in self.sdm['ConfigDescription']
                if configid == row.configDescriptionId][0].split(' ')[2:]


    @property
    def ants_orig(self):
        return [int(str(row.name).lstrip('ea')) for antid in antids for row in self.sdm['Antenna'] 
                if antid == str(row.antennaId)]


    @property
    def uvrange_orig(self):
        (u, v, w) = sdm_uvw(self.filename, self.scan, bdfdir=self.bdfdir)  # default uses time at start
        u = u * self.freq_orig[0] * (1e9/3e8) * (-1)
        v = v * self.freq_orig[0] * (1e9/3e8) * (-1)
        return (u.max() - u.min(), v.max() - v.min())


def sdm_metadata(sdmfile, scannum, bdfdir=None, **kw):
    """ Wraps Metadata call to provide immutable, attribute-filled class instance.
    """

    sdm = getsdm(sdmfile, bdfdir=bdfdir)
    scan = sdm.scan(scannum)

    kwargs = {}
    kwargs['filename'] = sdmfile
    kwargs['scan'] = scannum
    kwargs['bdfdir'] = bdfdir

    starttime_mjd = scan.bdf.startTime
    nints = scan.bdf.numIntegration
    inttime = scan.bdf.get_integration(0).interval
    endtime_mjd = starttime_mjd + (nints*inttime)/(24*3600)
    bdfstr = scan.bdf.fname

    kwargs['starttime_mjd'] = starttime_mjd
    kwargs['endtime_mjd'] = endtime_mjd
    kwargs['inttime'] = inttime
    kwargs['nints'] = nints
    kwargs['source'] = scan.source
    kwargs['intent'] = ' '.join(scan.intents)
    bdfstr = scan.bdf.fname
    if (not os.path.exists(bdfstr)) or ('X1' in bdfstr):
        kwargs['bdfstr'] = None
    else:
        kwargs['bdfstr'] = bdfstr

    sources = sdm_sources(sdmfile)
    kwargs['radec'] = [(prop['ra'], prop['dec']) for (sr, prop) in sources.iteritems() if prop['source'] == scan.source][0]
    kwargs['dishdiameter'] = float(str(sdm['Antenna'][0].dishDiameter).strip())

    kwargs['spw_orig'] = [int(str(row.spectralWindowId).split('_')[1]) for row in sdm['SpectralWindow']]
    kwargs['spw_nchan'] = [int(row.numChan) for row in sdm['SpectralWindow']]
    kwargs['spw_reffreq'] = [float(row.chanFreqStart) for row in sdm['SpectralWindow']]
    kwargs['spw_chansize'] = [float(row.chanFreqStep) for row in sdm['SpectralWindow']]

    # finally, overload with provided kw args
    for key, value in kw.iteritems():
        kwargs[key] = value

    return Metadata(**kwargs)


def sdm_uvw(sdmfile, scan=0, datetime=0, radec=(), bdfdir=''):
    """ Calculates and returns uvw in meters for a given SDM, time, and pointing direction.
    sdmfile is path to sdm directory that includes "Station.xml" file.
    scan is scan number defined by observatory.
    datetime is time (as string) to calculate uvw (format: '2014/09/03/08:33:04.20')
    radec is (ra,dec) as tuple in units of degrees (format: (180., +45.))
    bdfdir is path to bdfs (optional, for pre-archive SDMs)
    """

    assert os.path.exists(os.path.join(sdmfile, 'Station.xml')), 'sdmfile %s has no Station.xml file. Not an SDM?' % sdmfile

    # get scan info
    sources = sdm_sources(sdmfile)

    # default is to use scan info
    if (datetime == 0) and (len(radec) == 0):
        assert scan != 0, 'scan must be set when using datetime and radec'   # default scan value not valid

        logger.info('Calculating uvw for first integration of scan %d of source %s' % (scan, scan['source']))
        datetime = qa.time(qa.quantity(scan['startmjd'],'d'), form="ymd", prec=8)[0]
        sourcenum = [kk for kk in sources.keys() if sources[kk]['source'] == scan['source']][0]
        direction = me.direction('J2000', str(np.degrees(sources[sourcenum]['ra']))+'deg', str(np.degrees(sources[sourcenum]['dec']))+'deg')

    # secondary case is when datetime is also given
    elif (datetime != 0) and (len(radec) == 0):
        assert scan != 0, 'scan must be set when using datetime and radec'   # default scan value not valid
        assert '/' in datetime, 'datetime must be in yyyy/mm/dd/hh:mm:ss.sss format'

        logger.info('Calculating uvw at %s for scan %d of source %s' % (datetime, scan, scan['source']))
        sourcenum = [kk for kk in sources.keys() if sources[kk]['source'] == scan['source']][0]
        direction = me.direction('J2000', str(np.degrees(sources[sourcenum]['ra']))+'deg', str(np.degrees(sources[sourcenum]['dec']))+'deg')

    else:
        assert '/' in datetime, 'datetime must be in yyyy/mm/dd/hh:mm:ss.sss format'
        assert len(radec) == 2, 'radec must be (ra,dec) tuple in units of degrees'

        logger.info('Calculating uvw at %s in direction %s' % (datetime, direction))
        logger.info('This mode assumes all antennas used.')
        ra = radec[0]; dec = radec[1]
        direction = me.direction('J2000', str(ra)+'deg', str(dec)+'deg')

    # define metadata "frame" for uvw calculation
    sdm = getsdm(sdmfile)
    telescopename = str(sdm['ExecBlock'][0]['telescopeName']).strip()
    logger.debug('Found observatory name %s' % telescopename)

    me.doframe(me.observatory(telescopename))
    me.doframe(me.epoch('utc', datetime))
    me.doframe(direction)

    # read antpos
    if scan != 0:
        configid = [str(row.configDescriptionId) for row in sdm['Main'] if scan == int(row.scanNumber)][0]
        antidlist = [str(row.antennaId) for row in sdm['ConfigDescription'] if configid == str(row.configDescriptionId)][0].split(' ')[2:]
        stationidlist = [ant.stationId for antid in antidlist for ant in sdm['Antenna'] if antid == str(ant.antennaId)]
    else:
        stationidlist = [str(ant.stationId) for ant in sdm['Antenna']]

    positions = [str(station.position).strip().split(' ')
                 for station in sdm['Station'] 
                 if station.stationId in stationidlist]
    x = [float(positions[i][2]) for i in range(len(positions))]
    y = [float(positions[i][3]) for i in range(len(positions))]
    z = [float(positions[i][4]) for i in range(len(positions))]
    ants = me.position('itrf', qa.quantity(x, 'm'), qa.quantity(y, 'm'), qa.quantity(z, 'm'))

    # calc bl
    bls = me.asbaseline(ants)
    uvwlist = me.expand(me.touvw(bls)[0])[1]['value']

    # define new bl order to match sdm binary file bl order
    u = np.empty(len(uvwlist)/3); v = np.empty(len(uvwlist)/3); w = np.empty(len(uvwlist)/3)
    nants = len(ants['m0']['value'])
    ord1 = [i*nants+j for i in range(nants) for j in range(i+1,nants)]
    ord2 = [i*nants+j for j in range(nants) for i in range(j)]
    key=[]
    for new in ord2:
        key.append(ord1.index(new))
    for i in range(len(key)):
        u[i] = uvwlist[3*key[i]]
        v[i] = uvwlist[3*key[i]+1]
        w[i] = uvwlist[3*key[i]+2]

    return u, v, w


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
    data_read = rtpipe.parsesdm.read_bdf_segment(st, segment)
    return data_read


def randomdata(st):
    data = np.zeros(shape=(st['readints'], st['nbl'], st['nchan'], st['npol']), dtype='complex64')
    data.real = np.random.normal(size=data.shape)
    data.imag = np.random.normal(size=data.shape)
    return data


def randomuvw(st):
    return np.random.randint(-100, 100, size=st['nbl'])



