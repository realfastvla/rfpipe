from __future__ import print_function, division, absolute_import #, unicode_literals # not casa compatible
from builtins import bytes, dict, object, range, map, input#, str # not casa compatible
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import attr
import yaml
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
    flaglist = attr.ib(default=[('badchtslide', 4., 0.), ('badap', 3., 0.2),
                                ('blstd', 3.0, 0.05)])
    flagantsol = attr.ib(default=True)
    badspwpol = attr.ib(default=2.)  # 0 means no flagging done
    applyonlineflags = attr.ib(default=True)
    gainfile = attr.ib(default=None)
    # simulate transients from list of tuples with (amp, i0, dm, dt, dl, dm)
    simulated_transient = attr.ib(default=None)

    # processing
    nthread = attr.ib(default=1)
    nchunk = attr.ib(default=0)
#    nsegment = attr.ib(default=0)
    segmenttimes = attr.ib(default=None)
    memory_limit = attr.ib(default=32)

    # search
    dmarr = attr.ib(default=None)
    dtarr = attr.ib(default=None)
    dm_maxloss = attr.ib(default=0.05)  # fractional sensitivity loss
    mindm = attr.ib(default=0)
    maxdm = attr.ib(default=0)  # in pc/cm3
    dm_pulsewidth = attr.ib(default=3000)   # in microsec
    searchtype = attr.ib(default='image1')  # supported: image1, image1stat
    sigma_image1 = attr.ib(default=7.)
    sigma_image2 = attr.ib(default=7.)
    sigma_plot = attr.ib(default=7.)
    uvres = attr.ib(default=0)
    npixx = attr.ib(default=0)
    npixy = attr.ib(default=0)
    npix_max = attr.ib(default=0)
    uvoversample = attr.ib(default=1.)

    savenoise = attr.ib(default=False)
    savecands = attr.ib(default=False)
    candsfile = attr.ib(default=None)
    timewindow = attr.ib(default=30)
#    logfile = attr.ib(default=True)
    loglevel = attr.ib(default='INFO')


def parsepreffile(preffile=None, name=None):
    """ Read preference file and set parameter values.
    File should have python-like syntax. Full file name needed.
    name can be used to select a parameter set if multiple are defined in the
    yaml file.
    """

    if preffile:
        ptype = preffiletype(preffile)

        if ptype == 'yaml':
            return _parsepref_yaml(preffile, name=name)
        elif ptype == 'old':
            return _parsepref_old(preffile)
        else:
            logger.warn('Preffile type ({0}) not recognized.'.format(preffile))
            return {}
    else:
        return {}


def _parsepref_old(preffile):
    """ Parse parameter file of old type (using exec of Python commands)
    """

    pars = {}
    with open(preffile, 'r') as f:
        for line in f.readlines():
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

    yamlpars = yaml.load(open(preffile, 'r'))
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
    """

    prefs = {}
    allowed = attr.asdict(Preferences()).keys()

    for key in d.keys():
        if key in allowed:
            prefs[key] = d[key]

#    prefs['nsegment'] = d['nsegments']
    prefs['selectpol'] = 'auto'

    if not scan:
        scan = d['scan']
    prefs['segmenttimes'] = d['segmenttimesdict'][scan]

    return prefs
