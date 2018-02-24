from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import attr
import yaml
import json
from os import getcwd
from collections import OrderedDict
import hashlib
from rfpipe.version import __version__

import logging
logger = logging.getLogger(__name__)


@attr.s
class Preferences(object):
    """ Preferences *should* be immutable and express half of info needed to
    define state. Using preferences with metadata produces a unique state and
    pipeline outcome.

    TODO: can we freeze attributes while still having cached values?
    """

    rfpipe_version = attr.ib(default=__version__)

    # data selection
    chans = attr.ib(default=None)
    spw = attr.ib(default=None)
    excludeants = attr.ib(default=())
    selectpol = attr.ib(default='auto')  # 'auto', 'all', 'cross'
    fileroot = attr.ib(default=None)

    # preprocessing
    read_tdownsample = attr.ib(default=1)
    read_fdownsample = attr.ib(default=1)
    l0 = attr.ib(default=0.)  # in radians
    m0 = attr.ib(default=0.)  # in radians
    timesub = attr.ib(default=None)
    flaglist = attr.ib(default=[('badchtslide', 4., 10), ('blstd', 3.0, 0.05)])
    flagantsol = attr.ib(default=True)
    badspwpol = attr.ib(default=2.)  # 0 means no flagging done
    applyonlineflags = attr.ib(default=True)
    gainfile = attr.ib(default=None)
    # simulate transients from list of tuples with
    # values/units: (segment, i0/int, dm/pc/cm3, dt/s, amp/sys, dl/rad, dm/rad)
    simulated_transient = attr.ib(default=None)

    # processing
    nthread = attr.ib(default=1)
#    nsegment = attr.ib(default=0)
    segmenttimes = attr.ib(default=None)
    memory_limit = attr.ib(default=16)  # in GB; includes typical freqs/configs
    maximmem = attr.ib(default=16)  # in GB; defines chunk for fftw imaging

    # search
    fftmode = attr.ib(default='fftw')  # either 'fftw' or 'cuda'. defines segment size and algorithm used.
    dmarr = attr.ib(default=None)  # in pc/cm3
    dtarr = attr.ib(default=None)  # in samples
    dm_maxloss = attr.ib(default=0.05)  # fractional sensitivity loss
    mindm = attr.ib(default=0)  # in pc/cm3
    maxdm = attr.ib(default=0)  # in pc/cm3
    dm_pulsewidth = attr.ib(default=3000)   # in microsec
    searchtype = attr.ib(default='image1')  # supported: image1, image1stat
    sigma_image1 = attr.ib(default=7.)  # threshold for image1 algorithm
    sigma_image2 = attr.ib(default=7.)  # second threshold (no algo yet)
    sigma_plot = attr.ib(default=7.)
    uvres = attr.ib(default=0)  # in lambda
    npixx = attr.ib(default=0)  # set number of x pixels in image
    npixy = attr.ib(default=0)  # set number of y pixels in image
    npix_max = attr.ib(default=0)  # set max number of pixels in image
    uvoversample = attr.ib(default=1.)  # scale factor for to overresolve grid

    savenoise = attr.ib(default=False)
    savecands = attr.ib(default=False)
    candsfile = attr.ib(default=None)
    workdir = attr.ib(default=getcwd())  # set upon import
    timewindow = attr.ib(default=30)
#    logfile = attr.ib(default=True)
    loglevel = attr.ib(default='INFO')

    @property
    def ordered(self):
        """ Get OrderedDict of preferences sorted by key
        """

        keys = sorted(self.__dict__)
        return OrderedDict([(key, self.__dict__[key]) for key in keys])

    @property
    def json(self):
        """ json string that can be loaded into elasticsearch or hashed.
        """

        return json.dumps(self.ordered).encode('utf-8')

    @property
    def name(self):
        """ Unique name for an instance of preferences.
        To be used to look up preference set for a given candidate or data set.
        """

        return hashlib.md5(self.json).hexdigest()


def parsejson(jsonstring):
    """ Take json string (as from elasticsearch) and creates preference object.
    """

    inprefs = json.loads(jsonstring)
    return Preferences(**inprefs)


def parsepreffile(preffile=None, name=None, inprefs=None):
    """ Read preference file and set parameter values.
    File should have python-like syntax. Full file name needed.
    name can be used to select a parameter set if multiple are defined in the
    yaml file.
    """

    # define baseline dicts
    prefs = {}
    if inprefs is None:
        inprefs = {}

    # optionally overload
    if preffile:
        ptype = preffiletype(preffile)

        if ptype == 'yaml':
            prefs = _parsepref_yaml(preffile, name=name)
        elif ptype == 'old':
            prefs = _parsepref_old(preffile)
        else:
            logger.warn('Preffile type ({0}) not recognized.'.format(preffile))

    # optionall overload
    for key in inprefs:
        prefs[key] = inprefs[key]

    return prefs


def _parsepref_old(preffile):
    """ Parse parameter file of old type (using exec of Python commands)
    """

    pars = {}
    with open(preffile, 'r') as fp:
        for line in fp.readlines():
            # trim out comments and trailing cr
            line_clean = line.rstrip('\n').split('#')[0]
            if line_clean and '=' in line:   # use valid lines only
                attribute, value = line_clean.split('=')
                try:
                    value_eval = eval(value.strip())
                except NameError:
                    value_eval = value.strip()
                finally:
                    pars[attribute.strip()] = value_eval

    return pars


def _parsepref_yaml(preffile, name=None):
    """ Parse parameter file from yaml format.
    """

    name = 'default' if not name else name
    logger.info("Parsing preffile for preference set {0}".format(name))

    with open(preffile, 'r') as fp:
        yamlpars = yaml.load(fp)
        pars = yamlpars['rfpipe'][name]

    return pars


def preffiletype(preffile):
    """ Infer type from first uncommented line in preffile
    """

    with open(preffile, 'r') as fp:
        while True:
            line = fp.readline().split('#')[0]
            if line:
                break

    if '=' in line:
        return 'old'
    elif ':' in line:
        return 'yaml'
    else:
        return None


def oldstate_preferences(d, scan=None):
    """ Parse old state dictionary "d" and to define preferences instance
    If no scan is given, assumed to be a single scan state dict.
    """

    prefs = {}
    allowed = [kk for kk in attr.asdict(Preferences()).keys()]
    for key in list(d.keys()):
        if key in allowed:
            if key == 'segmenttimes':
                prefs[key] = d[key].tolist()
            else:
                prefs[key] = d[key]

#    prefs['nsegment'] = d['nsegments']
    prefs['selectpol'] = 'auto'
    if scan is not None:
        prefs['segmenttimes'] = d['segmenttimesdict'][scan].tolist()

    return prefs
