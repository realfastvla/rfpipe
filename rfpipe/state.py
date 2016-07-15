import json

class State(object):

    def  __init__(self, paramfile='', version=2, **kwargs):
        """ Define rfpipe state 
    
        For version==1, this parses in the rtpipe-style. Input uses python exec to run each line.
        For version>=2, this will make more general rfpipe-style pipeline state.
        """

        self.version = version

        if version == 1:
            # default values
            self.chans = []; self.spw = []    
            self.nskip = 0; self.excludeants = []; self.read_tdownsample = 1; self.read_fdownsample = 1
            self.selectpol = ['RR', 'LL', 'XX', 'YY']   # default processing assumes dual-pol
            self.nthread = 1; self.nchunk = 0; self.nsegments = 0; self.scale_nsegments = 1
            self.timesub = ''
            self.dmarr = []; self.dtarr = [1]    # dmarr = [] will autodetect, given other parameters
            self.dm_maxloss = 0.05; self.maxdm = 0; self.dm_pulsewidth = 3000   # dmloss is fractional sensitivity loss, maxdm in pc/cm3, width in microsec
            self.searchtype = 'image1'; self.sigma_image1 = 7.; self.sigma_image2 = 7.
            self.l0 = 0.; self.m0 = 0.
            self.uvres = 0; self.npix = 0; self.npix_max = 0; self.uvoversample = 1.
            self.flaglist = [('badchtslide', 4., 0.) , ('badap', 3., 0.2), ('blstd', 3.0, 0.05)]
            self.flagantsol = True; self.gainfile = ''; self.bpfile = ''; self.fileroot = ''; self.applyonlineflags = True
            self.savenoise = False; self.savecands = False; self.logfile = True; self.loglevel = 'INFO'
            self.writebdfpkl = False; self.mock = 0
                           
            # overload with the parameter file values, if provided
            if len(paramfile):
                self.parseparamfile(paramfile)

        elif version == 2:
            # parse paramfile

            self.__dict__['featureind'] = ['scan', 'segment', 'int', 'dmind', 'dtind', 'beamnum']  # feature index. should be stable.

            # parse kwargs and overload defaults
            for key in kwargs:
                setattr(self, key, value)

            raise NotImplementedError


    def parseparamfile(self, paramfile):
        """ Read parameter file and set parameter values.
        File should have python-like syntax. Full file name needed.
        """

        with open(paramfile, 'r') as f:
            for line in f.readlines():
                line_clean = line.rstrip('\n').split('#')[0]   # trim out comments and trailing cr
                if line_clean and '=' in line:   # use valid lines only
                    attribute, value = line_clean.split('=')
                    try:
                        value_eval = eval(value.strip())
                    except NameError:
                        value_eval = value.strip()
                    finally:
                        setattr(self, attribute.strip(), value_eval)


    def parseyaml(self, paramfile, name='default'):
        # maybe use pyyaml to parse parameters more reliably
        # could save multiple per yml paramfile
        pass


    @property
    def reproducekeys(self):
        """ Minimal set of state keys required to assure that state can reproduce a given candidate. 

        Should be independent of data? Or assumes post-reading of metadata and applying parameter generation functions?
        """

        # this set is input, version defines functions that transform this to pipeline state
        return sorted(['sdmfile', 'excludeants', 'read_tdownsample', 'read_fdownsample',
                       'selectpol', 'timesub', 'dmarr', 'dtarr', 'searchtype',
#                       'features', 'sigma_image1', 'sigma_image2', 'sigma_bisp',   # not sure about this set
                       'uvres', 'npixx', 'npixy', 'version',
                       'flaglist', 'gainfile', 'bpfile', 'onlineflags'])

    # should each of the above actually be considered transformation functions? 
    #  so, input is observation name, next step is transform by ignore ants, next step...
    #
    #   nsegments or dmarr + memory_limit + metadata => segmenttimes
    #   dmarr or dm_parameters + metadata => dmarr
    #   uvoversample + npix_max + metadata => npixx, npixy

    # flagantsol should be merged with flaglist


    @property
    def hash(self):
        """ Hash that identifies pipeline state that produces unique set of output products """

        extant_keys = self.keys()
        if all([kk in self.reproducekeys for kk in extant_keys]):
            values = [self[key] for key in self.reproducekeys]
            return hash(json.dumps(repr(values)))  # is repr safe?
        else:
            print('Cannot make hash without minimal set defined in reproducekeys property.')
            return None


    @property
    def defined(self):
        return self.__dict__.keys()


    def __getitem__(self, key):
        return self.__dict__[key]


    def __setitem__(self, key, value):
        self.__dict__[key] = value


    def __delitem__(self, key):
        del self.__dict__[key]


    def __str__(self):
        return str(self.__dict__)
