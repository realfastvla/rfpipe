from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import pickle
import os
from copy import deepcopy
import numpy as np
from math import cos, radians
from numpy.lib.recfunctions import append_fields
from collections import OrderedDict
import matplotlib as mpl
from astropy import time
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rfpipe import version, fileLock
from bokeh.plotting import ColumnDataSource, Figure, save, output_file
import scipy.stats.mstats as mstats
from bokeh.models import HoverTool
from bokeh.models import Row
from collections import OrderedDict
import hdbscan

import logging
logger = logging.getLogger(__name__)

class CandData(object):
    """ Object that bundles data from search stage to candidate visualization.
    Provides some properties for the state of the phased data and candidate.
    """

    def __init__(self, state, loc, image, data, **kwargs):
        """ Instantiate with pipeline state, candidate location tuple,
        image, and resampled data phased to candidate.
        TODO: Need to use search_dimensions to infer candloc meaning
        """

        self.state = state
        self.loc = tuple(loc)
        self.image = image
        self.data = data
        if 'snrk' in kwargs:  # hack to allow detection level calculation in
            self.snrk = kwargs['snrk']
        else:
            self.snrk = None
        if 'snrarms' in kwargs:  # hack to allow detection level calculation in
            self.snrarms = kwargs['snrarms']
        else:
            self.snrarms = None
        if 'cluster' in kwargs:
            self.cluster = kwargs['cluster']
        else:
            self.cluster = None
        if 'clustersize' in kwargs:
            self.clustersize = kwargs['clustersize']
        else:
            self.clustersize = None

        assert len(loc) == len(self.state.search_dimensions), ("candidate location "
                                                          "should set each of "
                                                          "the st.search_dimensions")

    def __repr__(self):
        return 'CandData for scanId {0} at loc {1}'.format(self.state.metadata.scanId, self.loc)

    @property
    def searchtype(self):
        return self.state.prefs.searchtype

    @property
    def features(self):
        return self.state.features

    @property
    def snrtot(self):
        """ Optimal SNR given searchtype (e.g., snr1 with snrk, if snrk measured)
        Note that snrk can be calclated after detection, so snrtot represents post detection
        significance.
        """

        if self.state.prefs.searchtype in ['image', 'imagek', 'armkimage']:
            return (self.snrk**2 + self.snr1**2)**0.5
        elif self.state.prefs.searchtype == 'armk':
            return (self.snrk**2 + self.snrarms**2)**0.5

    @property
    def snr1(self):
# TODO: find good estimate for both CPU and GPU
#       imstd = util.madtostd(image)  # outlier resistant
        return self.image.max()/self.image.std()

    @property
    def immax1(self):
        return self.image.max()

    @property
    def l1(self):
        return self.peak_lm[0]

    @property
    def m1(self):
        return self.peak_lm[1]

    @property
    def spec(self):
        return self.data.real.mean(axis=2)[self.integration_rel]

    @property
    def specstd(self):
        return self.spec.std()

    @property
    def specskew(self):
        return mstats.skew(self.spec)

    @property
    def speckur(self):
        return mstats.kurtosis(self.spec)

    @property
    def imskew(self):
        return mstats.skew(self.image.flatten())

    @property
    def imkur(self):
        return mstats.kurtosis(self.image.flatten())

    @property
    def lc(self):
        return self.data.real.mean(axis=2).mean(axis=1)

    @property
    def tskew(self):
        return mstats.skew(self.lc)

    @property
    def tkur(self):
        return mstats.kurtosis(self.lc)

    @property
    def integration_rel(self):
        """ Candidate integration relative to data time window
        """

        if self.loc[1] < self.state.prefs.timewindow//2:
            return self.loc[1]
        else:
            return self.state.prefs.timewindow//2

    @property
    def peak_lm(self):
        """
        """

        return self.state.pixtolm(self.peak_xy)

    @property
    def peak_xy(self):
        """ Peak pixel in image
        Only supports positive peaks for now.
        """

        return np.where(self.image == self.image.max())

    @property
    def time_top(self):
        """ Time in mjd where burst is at top of band
        """

        return (self.state.segmenttimes[self.loc[0]][0] +
                (self.loc[1]*self.state.inttime)/(24*3600))


class CandCollection(object):
    """ Wrap candidate array with metadata and
    prefs to be attached and pickled.
    """

    def __init__(self, array=np.array([]), prefs=None, metadata=None):
        self.array = array
        self.prefs = prefs
        self.metadata = metadata
        self.rfpipe_version = version.__version__
        self._state = None
        # TODO: pass in segmenttimes here to avoid recalculating during search?

    def __repr__(self):
        if self.metadata is not None:
            return ('CandCollection for {0}, scan {1}, segment {2} with {3} candidate{4}'
                    .format(self.metadata.datasetId, self.metadata.scan, self.segment,
                            len(self), 's'[not len(self)-1:]))
        else:
            return ('CandCollection with {0} rows'.format(len(self.array)))

    def __len__(self):
        return len(self.array)

    def __add__(self, cc):
        """ Allow candcollections to be added within a given scan.
        (same dmarr, dtarr, segmenttimes)
        Adding empty cc ok, too.
        """

        later = self
        # TODO: update to allow different simulated_transient fields that get added into single list
        assert self.prefs.name == cc.prefs.name, "Cannot add collections with different preference name/hash"
        assert self.state.dmarr == cc.state.dmarr,  "Cannot add collections with different dmarr"
        assert self.state.dtarr == cc.state.dtarr,  "Cannot add collections with different dmarr"

        # standard case
        if self.state.nsegment == cc.state.nsegment:
            assert (self.state.segmenttimes == cc.state.segmenttimes).all(),  "Cannot add collections with different segmenttimes"
        # OTF case (one later than the other)
        else:
            if self.state.nsegment > cc.state.nsegment:
                assert self.metadata.starttime_mjd == cc.metadata.starttime_mjd, "OTF segments should have same start time"
                assert (self.state.segmenttimes[:cc.state.nsegment] == cc.state.segmenttimes).all(),  "OTF segments should have shared segmenttimes"
                later = self
            elif self.state.nsegment < cc.state.nsegment:
                assert self.metadata.starttime_mjd == cc.metadata.starttime_mjd, "OTF segments should have same start time"
                assert (self.state.segmenttimes == cc.state.segmenttimes[:self.state.nsegment]).all(),  "OTF segments should have shared segmenttimes"
                later = cc

        # combine candidate arrays
        if len(later) and len(cc):
            later.array = np.concatenate((later.array, cc.array))
        elif not len(later) and len(cc):
            later.array = cc.array

        # combine prefs simulated_transient
        later.prefs.simulated_transient = later.prefs.simulated_transient or cc.prefs.simulated_transient

        return later

    def __radd__(self, other):
        """ Support recursive add so we can sum(ccs)
        """

        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __getitem__(self, key):
        return CandCollection(array=self.array.take([key]), prefs=self.prefs,
                              metadata=self.metadata)

    @property
    def scan(self):
        if self.metadata is not None:
            return self.metadata.scan
        else:
            return None

    @property
    def segment(self):
        if len(self):
            segments = np.unique(self.array['segment'])
            if len(segments) == 1:
                return int(segments[0])
            elif len(segments) > 1:
                logger.warning("Multiple segments in this collection")
                return segments
        else:
            return None

    @property
    def locs(self):
        if len(self.array):
            return self.array[['segment', 'integration', 'dmind', 'dtind',
                               'beamnum']].tolist()
        else:
            return np.array([], dtype=int)

    @property
    def candmjd(self):
        """ Candidate MJD at top of band
        """

#        dt_inf = util.calc_delay2(1e5, self.state.freq.max(), self.canddm)
        t_top = np.array(self.state.segmenttimes)[self.array['segment'], 0] + (self.array['integration']*self.canddt)/(24*3600)

        return t_top

    @property
    def canddm(self):
        """ Candidate DM in pc/cm3
        """

        dmarr = np.array(self.state.dmarr)
        return dmarr[self.array['dmind']]

    @property
    def canddt(self):
        """ Candidate dt in seconds
        """

        dtarr = np.array(self.state.dtarr)
        return self.metadata.inttime*dtarr[self.array['dtind']]

    @property
    def candl(self):
        """ Return l1 for candidate (offset from phase center in RA direction)
        """
        #  beamnum not yet supported

        return self.array['l1']

    @property
    def candm(self):
        """ Return m1 for candidate (offset from phase center in Dec direction)
        """
        #  beamnum not yet supported

        return self.array['m1']

    @property
    def cluster(self):
        """ Return cluster label
        """

        if self.prefs.clustercands:
            return self.array['cluster']
        else:
            return None

    @property
    def clustersize(self):
        """ Return size of cluster
        """

        if self.prefs.clustercands:
            return self.array['clustersize']
        else:
            return None

    @property
    def snrtot(self):
        """ Optimal SNR, given fields in cc (quadrature sum)
        """
        fields = self.array.dtype.fields
        snr = 0.

        if 'snr1' in fields:
            snr += self.array['snr1']**2
        if 'snrk' in fields:
            snr += self.array['snrk']**2
        if 'snrarms' in fields:
            snr += self.array['snrkarms']**2

        return snr**0.5

    @property
    def state(self):
        """ Sets state by regenerating from the metadata and prefs.
        """

        from rfpipe import state

        if self._state is None:
            self._state = state.State(inmeta=self.metadata, inprefs=self.prefs,
                                      showsummary=False, validate=False)

        return self._state

    @property
    def mock_map(self):
        """ Look for mock in candcollection
        TODO: return values that help user know mocks found and missed.
        """
        
        if self.prefs.simulated_transient is not None:
            clusters = self.array['cluster'].astype(int)
            cl_rank, cl_count = calc_cluster_rank(self)
            mock_labels = []
            map_mocks = {}
            for mock in self.prefs.simulated_transient:
                (segment, integration, dm, dt, amp, l0, m0) = mock
                dmind0 = np.abs((np.array(self._state.dmarr)-dm)).argmin()
                dtind0 = np.abs((np.array(self._state.dtarr)-dt)).argmin()
                mockloc = (segment, integration, dmind0, dtind0, 0)

                if mockloc in self.locs:
                    label = clusters[self.locs.index(mockloc)]
                    mock_labels.append(label)
                    clustersize = cl_count[self.locs.index(mockloc)]
                    map_mocks[mock] = np.array(self.locs)[clusters == label].tolist()
                    logger.info("Found mock ({0}, {1}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.4f}, {6:.4f}) at loc {7} with label {8} of size {9}"\
                     .format(segment, integration, dm, dt, amp, l0,\
                             m0 ,mockloc, label, clustersize))
                else:
                    map_mocks[mock] = []
                    mock_labels.append(-2)
                    logger.info("The mock ({0}, {1}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.4f}, {6:.4f}) wasn't found at loc {7}"\
                     .format(segment, integration, dm, dt, amp, l0, m0 ,mockloc))
            return map_mocks, mock_labels
        else:
            return None

    def sdmname(self):
        """ Get name of SDM created by realfast based on naming convention
        """

        segment = self.segment
        segmenttimes = self.state.segmenttimes
        startTime = segmenttimes[segment][0]
        bdftime = int(time.Time(startTime, format='mjd').unix*1e3)
        return 'realfast_{0}_{1}'.format(self.state.metadata.datasetId, bdftime)


def save_and_plot(canddatalist):
    """ Converts a canddata list into a plots and a candcollection.
    Calculates candidate features from CandData instance(s).
    Returns structured numpy array of candidate features labels defined in
    st.search_dimensions.
    Generates png plot for peak cands, if so defined in preferences.
    """

    if isinstance(canddatalist, CandData):
        canddatalist = [canddatalist]
    elif isinstance(canddatalist, list):
        if not len(canddatalist):
            return CandCollection()
    else:
        logger.warning("argument must be list of CandData object")

    logger.info('Calculating features for {0} candidate{1}.'
                .format(len(canddatalist), 's'[not len(canddatalist)-1:]))

    st = canddatalist[0].state

    # TODO: should this be all features, calcfeatures, searchfeatures?
    featurelists = []
    for feature in st.searchfeatures:
        ff = []
        for i, canddata in enumerate(canddatalist):
            ff.append(canddata_feature(canddata, feature))
        featurelists.append(ff)
    kwargs = dict(zip(st.searchfeatures, featurelists))

    candlocs = []
    for i, canddata in enumerate(canddatalist):
        candlocs.append(canddata_feature(canddata, 'candloc'))
        kwargs['candloc'] = candlocs

    if canddata.cluster is not None:
        clusters = []
        clustersizes = []
        for i, canddata in enumerate(canddatalist):
            clusters.append(canddata_feature(canddata, 'cluster'))
            clustersizes.append(canddata_feature(canddata, 'clustersize'))
        kwargs['cluster'] = clusters
        kwargs['clustersize'] = clustersizes

    candcollection = make_candcollection(st, **kwargs)

    if (st.prefs.savecanddata or st.prefs.savecandcollection or st.prefs.saveplots) and len(candcollection):
        if len(candcollection) > 1:
            snrs = candcollection.array['snr1'].flatten()
        elif len(candcollection) == 1:
            snrs = None

        # save cc and save/plot each canddata
        for i, canddata in enumerate(canddatalist):
            if st.prefs.savecanddata:
                save_cands(st, canddata=canddata)
            if st.prefs.saveplots:
                if canddata.cluster is not None:
                    clustertuple = (clusters[i], clustersizes[i])
                else:
                    clustertuple = None
                candplot(canddata, cluster=clustertuple, snrs=snrs)

	#Make VOEvents from the candcollection
	make_voevent(candcollection)

    return candcollection


def canddata_feature(canddata, feature):
    """ Calculate a feature (or candloc) from a canddata instance.
    feature must be name from st.features or 'candloc'.
    """

#    TODO: update this to take feature as canddata property

    if feature == 'candloc':
        return canddata.loc
    elif feature == 'snr1':
        return canddata.snr1
    elif feature == 'snrarms':
        return canddata.snrarms
    elif feature == 'snrk':
        return canddata.snrk
    elif feature == 'cluster':
        return canddata.cluster
    elif feature == 'clustersize':
        return canddata.clustersize
    elif feature == 'specstd':
        return canddata.specstd
    elif feature == 'specskew':
        return canddata.specskew
    elif feature == 'speckur':
        return canddata.speckur
    elif feature == 'immax1':
        return canddata.immax1
    elif feature == 'l1':
        return canddata.l1
    elif feature == 'm1':
        return canddata.m1
    elif feature == 'imskew':
        return canddata.imskew
    elif feature == 'imkur':
        return canddata.imkur
    elif feature == 'tskew':
        return canddata.tskew
    elif feature == 'tkur':
        return canddata.tkur
    else:
        raise NotImplementedError("Feature {0} calculation not implemented"
                                  .format(feature))


def make_candcollection(st, **kwargs):
    """ Construct a candcollection with columns set by keywords.
    Minimal cc has a candloc (segment, int, dmind, dtind, beamnum).
    Can also provide features as keyword/value pairs.
    keyword is the name of the column (e.g., "l1", "snr")
    and the value is a list of values of equal length as candlocs.
    """

    if len(kwargs):
        remove = []
        for k, v in iteritems(kwargs):
            if (len(v) == 0) and (k != 'candloc'):
                remove.append(k)
        for k in remove:
            _ = kwargs.pop(k)

        # assert 1-to-1 mapping of input lists
        assert 'candloc' in kwargs
        assert isinstance(kwargs['candloc'], list)
        for v in itervalues(kwargs):
            assert len(v) == len(kwargs['candloc'])

        candlocs = kwargs['candloc']
        features = [kw for kw in list(kwargs.keys())]
        features.remove('candloc')
        fields = []
        types = []
        for ff in st.search_dimensions + tuple(features):
            fields.append(str(ff))
            if ff in st.search_dimensions + ('cluster', 'clustersize'):
                tt = '<i4'
            else:
                tt = '<f4'
            types.append(str(tt))

        dtype = np.dtype({'names': fields, 'formats': types})
        array = np.zeros(len(candlocs), dtype=dtype)
        for i in range(len(candlocs)):
            ff = list(candlocs[i])
            for feature in features:
                ff.append(kwargs[feature][i])

            array[i] = tuple(ff)
        candcollection = CandCollection(array=array, prefs=st.prefs,
                                        metadata=st.metadata)
    else:
        candcollection = CandCollection(prefs=st.prefs,
                                        metadata=st.metadata)

    return candcollection


def cluster_candidates(cc, downsample_xy=1, returnclusterer=False, label_unclustered=True):
    """ Perform density based clustering on candidates using HDBSCAN
    parameters used for clustering: dm, time, l,m.
    label_unclustered adds new cluster label for each unclustered candidate.
    Returns label for each row in candcollection.
    """

    cc1 = deepcopy(cc)
    if len(cc1) > 1:
        if isinstance(cc1.prefs.clustercands, tuple):
            min_cluster_size, min_samples = cc1.prefs.clustercands
        elif isinstance(cc1.prefs.clustercands, bool):
            if cc1.prefs.clustercands:
                min_cluster_size = 5
                min_samples = 3
            else:
                logger.info("Not performing clustering")
                return cc1
        else:
            logger.warning("No clustering. prefs.clustercands value not valid: {0}."
                        .format(cc1.prefs.clustercands))
            return cc1

        logger.info("Clustering parameters set to ({0},{1}) and downsampling in xy by {2}."
                    .format(min_cluster_size, min_samples, downsample_xy))

        if min_cluster_size > len(cc1):
            logger.info("Setting min_cluster_size to number of cands {0}"
                        .format(len(cc1)))
            min_cluster_size = len(cc1)
        candl = cc1.candl
        candm = cc1.candm
        npixx = cc1.state.npixx
        npixy = cc1.state.npixy
        uvres = cc1.state.uvres

        dmind = cc1.array['dmind']
        dtind = cc1.array['dtind']
        dtarr = cc1.state.dtarr
        timearr_ind = cc1.array['integration']  # time index of all the candidates

        time_ind = np.multiply(timearr_ind, np.array(dtarr).take(dtind))
        peakx_ind, peaky_ind = cc1.state.calcpix(candl, candm, npixx, npixy,
                                                 uvres)
        data = np.transpose([peakx_ind//downsample_xy, peaky_ind//downsample_xy,
                             dmind, time_ind])

        clusterer = hdbscan.HDBSCAN(metric='hamming',
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_method='eom',
                                    allow_single_cluster=True).fit(data)
        nclustered = np.max(clusterer.labels_ + 1)
        nunclustered = len(np.where(clusterer.labels_ == -1)[0])

        logger.info("Found {0} clusters and {1} unclustered candidates for "
                    "min cluster size {2}"
                    .format(nclustered, nunclustered, min_cluster_size))

        labels = clusterer.labels_.astype(np.int32)
    else:
        clusterer = None
        labels = -1*np.ones(len(cc1), dtype=np.int32)

    if -1 in labels and label_unclustered:
        unclustered = np.where(labels == -1)[0]
        logger.info("Adding {0} unclustered candidates as individual clusters"
                    .format(len(unclustered)))
        newind = max(labels)
        for cli in unclustered:
            newind += 1
            labels[cli] = newind

    # TODO: rebuild array with new col or accept broken python 2 or create cc with 'cluster' set to -1
    if 'cluster' not in cc1.array.dtype.fields:
        cc1.array = append_fields(cc1.array, 'cluster', labels, usemask=False)
    else:
        cc1.array['cluster'] = labels

    if returnclusterer:
        return cc1, clusterer
    else:
        return cc1


def calc_cluster_rank(cc):
    """ Given cluster array of candcollection, calculate rank relative
    to total count in each cluster.
    Rank ranges from 1 (highest SNR in cluster) to total count in cluster.
    """

    assert 'cluster' in cc.array.dtype.fields

    # get count in cluster and snr rank of each in its cluster
    clusters = cc.array['cluster'].astype(int)
    cl_rank = np.zeros(len(clusters), dtype=int)
    cl_count = np.zeros(len(clusters), dtype=int)

    # TODO: check on best way to find max SNR with kalman, etc
    for cluster in np.unique(clusters):
        clusterinds = np.where(cluster == clusters)[0]
        snrs = cc.array['snr1'][clusterinds]
        cl_rank[clusterinds] = np.argsort(np.argsort(snrs)[::-1])+1 
        cl_count[clusterinds] = len(clusterinds)

    return cl_rank, cl_count


def save_cands(st, candcollection=None, canddata=None):
    """ Save candidate collection or cand data to pickle file.
    Collection saved as array with metadata and preferences attached.
    Writes to location defined by state using a file lock to allow multiple
    writers.
    """

    if canddata is not None:
        if st.prefs.savecanddata:
            logger.info('Saving CandData to {0}.'.format(st.candsfile))

            try:
                with fileLock.FileLock(st.candsfile+'.lock', timeout=60):
                    with open(st.candsfile, 'ab+') as pkl:
                        pickle.dump(canddata, pkl)

            except fileLock.FileLock.FileLockException:
                segment = canddata.loc[0]
                newcandsfile = ('{0}_seg{1}.pkl'
                                .format(st.candsfile.rstrip('.pkl'), segment))
                logger.warning('Candidate file writing timeout. '
                               'Spilling to new file {0}.'.format(newcandsfile))
                with open(newcandsfile, 'ab+') as pkl:
                    pickle.dump(canddata, pkl)

        else:
            logger.info('Not saving CandData.')

    if candcollection is not None:
        if st.prefs.savecandcollection:
            logger.info('Saving {0} candidate{1} to {2}.'
                        .format(len(candcollection),
                                's'[not len(candcollection)-1:], st.candsfile))

            try:
                with fileLock.FileLock(st.candsfile+'.lock', timeout=60):
                    with open(st.candsfile, 'ab+') as pkl:
                        pickle.dump(candcollection, pkl)

            except fileLock.FileLock.FileLockException:
                segment = candcollection.segment
                newcandsfile = ('{0}_seg{1}.pkl'
                                .format(st.candsfile.rstrip('.pkl'), segment))
                logger.warning('Candidate file writing timeout. '
                               'Spilling to new file {0}.'.format(newcandsfile))
                with open(newcandsfile, 'ab+') as pkl:
                    pickle.dump(candcollection, pkl)
        else:
            logger.info('Not saving candidates.')


def iter_cands(candsfile, select='candcollection'):
    """ Iterate through (new style) candsfile and return either
    a candidatecollection or canddata.
    select defines what kind of object to return:
    - 'canddata' is heavier object with image and spectrum (used to make plots)
    - 'candcollection' is lighter object with features.
    """

    assert select.lower() in ['candcollection', 'canddata']

    try:
        with open(candsfile, 'rb') as pkl:
            while True:  # step through all possible segments
                try:
                    candobj = pickle.load(pkl)
                    if select.lower() in str(type(candobj)).lower():
                        yield candobj
                except EOFError:
                    logger.debug('Reached end of pickle.')
                    break
    except UnicodeDecodeError:
        with open(candsfile, 'rb') as pkl:
            while True:  # step through all possible segments
                try:
                    candobj = pickle.load(pkl, encoding='latin-1')
                    if select.lower() in str(type(candobj)).lower():
                        yield candobj
                except EOFError:
                    logger.debug('Reached end of pickle.')
                    break


def iter_noise(noisefile):
    """ Iterate through (new style) noisefile and return a list of tuples
    for each segment.
    """

    try:
        with open(noisefile, 'rb') as pkl:
            while True:  # step through all possible segments
                try:
                    noises = pickle.load(pkl)
                    for noise in noises:
                        yield noise

                except EOFError:
                    logger.debug('No more CandCollections.')
                    break
    except UnicodeDecodeError:
        with open(noisefile, 'rb') as pkl:
            while True:  # step through all possible segments
                try:
                    noises = pickle.load(pkl, encoding='latin-1')
                    for noise in noises:
                        yield noise

                except EOFError:
                    logger.debug('No more CandCollections.')
                    break


### bokeh summary plot
def visualize_clustering(cc, clusterer):
    
    import seaborn as sns
    from bokeh.plotting import figure, output_file, show, output_notebook
    from bokeh.layouts import row,column
    from bokeh.models import HoverTool
    from bokeh.models.sources import ColumnDataSource
    
    color_palette = sns.color_palette('deep', np.max(cc.cluster) + 1) #get a color palette with number of colors = number of clusters
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in cc.cluster]    #assigning each cluster a color, and making a list

    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    cluster_colors = list(map(mpl.colors.rgb2hex, cluster_member_colors)) #converting sns colors to hex for bokeh
    
    width = 450
    height = 350
    alpha = 0.1
    output_notebook()

    TOOLS = 'crosshair, box_zoom, reset, box_select, tap, hover, wheel_zoom'
    
    candl = cc.candl
    candm = cc.candm
    npixx = cc.state.npixx
    npixy = cc.state.npixy
    uvres = cc.state.uvres

    dmind = cc.array['dmind']
    dtind = cc.array['dtind']
    dtarr = cc.state.dtarr
    timearr_ind = cc.array['integration']  # time index of all the candidates

    time_ind = np.multiply(timearr_ind, np.array(dtarr).take(dtind))
    peakx_ind, peaky_ind = cc.state.calcpix(candl, candm, npixx, npixy, uvres)
    
    snr = cc.snrtot    
    
    data = dict(l= peakx_ind, m= peaky_ind, dm= dmind, time= time_ind, snr= snr, colors = cluster_colors)
    source=ColumnDataSource(data=data)


    p = figure(title="m vs l", x_axis_label='l', y_axis_label='m',plot_width=width, plot_height=height, tools = TOOLS)
    p.circle(x='l',y='m', size='snr', line_width = 1, color = 'colors', fill_alpha=alpha, source = source) # linewidth=0,
    #p.circle(x=df.l,y=df.m, size=5, line_width = 1, color = cluster_colors, fill_alpha=0.5) # linewidth=0,
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("m", "@m"), ("l", "@l"), ("time", "@time"), ("DM", "@dm"), ("SNR", "@snr")]

    #p.circle(x,y, size=5, line_width = 1, color = colors)#, , fill_alpha=1) # linewidth=0,
    #p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")
    p2 = figure(title="DM vs time", x_axis_label='time', y_axis_label='DM',plot_width=width, plot_height=height, tools = TOOLS)
    p2.circle(x='time',y='dm', size='snr', line_width = 1, color = 'colors', fill_alpha=alpha, source=source) # linewidth=0,
    hover = p2.select(dict(type=HoverTool))
    hover.tooltips = [("m", "@m"), ("l", "@l"), ("time", "@time"), ("DM", "@dm"), ("SNR", "@snr")]


    p3 = figure(title="DM vs l", x_axis_label='l', y_axis_label='DM',plot_width=width, plot_height=height, tools = TOOLS)
    p3.circle(x='l',y='dm', size='snr', line_width = 1, color = 'colors', fill_alpha=alpha, source=source) # linewidth=0,
    hover = p3.select(dict(type=HoverTool))
    hover.tooltips = [("m", "@m"), ("l", "@l"), ("time", "@time"), ("DM", "@dm"), ("SNR", "@snr")]


    p4 = figure(title="time vs l", x_axis_label='l', y_axis_label='time',plot_width=width, plot_height=height, tools = TOOLS)
    p4.circle(x='l',y='time', size='snr', line_width = 1, color = 'colors', fill_alpha=alpha, source=source) # linewidth=0,
    hover = p4.select(dict(type=HoverTool))
    hover.tooltips = [("m", "@m"), ("l", "@l"), ("time", "@time"), ("DM", "@dm"), ("SNR", "@snr")]


    # show the results
    (show(column(row(p, p2), row(p3, p4))))


def makesummaryplot(candsfile):
    """ Given a scan's candsfile, read all candcollections and create
    bokeh summary plot
    TODO: modify to take candcollection
    """

    time = []
    segment = []
    integration = []
    dmind = []
    dtind = []
    snr = []
    dm = []
    dt = []
    l1 = []
    m1 = []
    ccs = list(iter_cands(candsfile))
    if len(ccs):
        cc = sum(ccs)
        if not len(cc):
            return 0
    else:
        return 0

    time.append(cc.candmjd*(24*3600))
    segment.append(cc.array['segment'])
    integration.append(cc.array['integration'])
    dmind.append(cc.array['dmind'])
    dtind.append(cc.array['dtind'])
    snr.append(cc.snrtot)
    dm.append(cc.canddm)
    dt.append(cc.canddt)
    l1.append(cc.array['l1'])
    m1.append(cc.array['m1'])

    time = np.concatenate(time)
#    time = time - time.min()  # TODO: try this, or ensure nonzero time array
    segment = np.concatenate(segment)
    integration = np.concatenate(integration)
    dmind = np.concatenate(dmind)
    dtind = np.concatenate(dtind)
    snr = np.concatenate(snr)
    dm = np.concatenate(dm)
    dt = np.concatenate(dt)
    l1 = np.concatenate(l1)
    m1 = np.concatenate(m1)

    keys = ['seg{0}-i{1}-dm{2}-dt{3}'.format(segment[i], integration[i],
                                             dmind[i], dtind[i])
            for i in range(len(segment))]
    sizes = calcsize(snr)
    colors = colorsat(l1, m1)
    data = dict(snrs=snr, dm=dm, l1=l1, m1=m1, time=time, sizes=sizes,
                colors=colors, keys=keys)

    dmt = plotdmt(data, yrange=(min(cc.state.dmarr), max(cc.state.dmarr)))
    loc = plotloc(data, extent=radians(cc.state.fieldsize_deg))
    combined = Row(dmt, loc, width=950)

    htmlfile = candsfile.replace('.pkl', '.html')
    output_file(htmlfile)
    save(combined)
    logger.info("Saved summary plot {0} with {1} candidate{2}"
                .format(htmlfile, len(segment), 's'[not len(segment)-1:]))

    return len(cc)


def plotdmt(data, circleinds=[], crossinds=[], edgeinds=[],
            tools="hover,pan,box_select,wheel_zoom,reset", plot_width=450,
            plot_height=400, yrange=None):
    """ Make a light-weight dm-time figure """

    fields = ['dm', 'time', 'sizes', 'colors', 'snrs', 'keys']

    if not len(circleinds):
        circleinds = list(range(len(data['snrs'])))

    # set ranges
    inds = circleinds + crossinds + edgeinds
    dm = [data['dm'][i] for i in inds]
    if yrange is None:
        dm_min = min(min(dm), max(dm)/1.05)
        dm_max = max(max(dm), min(dm)*1.05)
    else:
        assert isinstance(yrange, tuple)
        dm_min, dm_max = yrange
    t0 = min(data['time'])
    t1 = max(data['time'])
    data['time'] = data['time'] - t0
    time_range = t1-t0
    time_min = -0.05*time_range
    time_max = 1.05*time_range

    source = ColumnDataSource(data=dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds]))
                                        for (key, value) in list(data.items())
                                        if key in fields}))
    dmt = Figure(plot_width=plot_width, plot_height=plot_height,
                 toolbar_location="left", x_axis_label='Time (s; from {0})'.format(t0),
                 y_axis_label='DM (pc/cm3)', x_range=(time_min, time_max),
                 y_range=(dm_min, dm_max),
                 output_backend='webgl', tools=tools)
    dmt.circle('time', 'dm', size='sizes', fill_color='colors',
               line_color=None, fill_alpha=0.2, source=source)

#    if crossinds:
#        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
#                                                  for (key, value) in list(data.items()) if key in fields}))
#        dmt.cross('time', 'dm', size='sizes', fill_color='colors',
#                  line_alpha=0.3, source=sourceneg)
#
#    if edgeinds:
#        sourceedge = ColumnDataSource(data=dict({(key, tuple([value[i] for i in edgeinds]))
#                                                   for (key, value) in list(data.items()) if key in fields}))
#        dmt.circle('time', 'dm', size='sizes', line_color='colors',
#                   fill_color='colors', line_alpha=0.5, fill_alpha=0.2,
#                   source=sourceedge)

    hover = dmt.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('keys', '@keys')])

    return dmt


def plotloc(data, circleinds=[], crossinds=[], edgeinds=[],
            tools="hover,pan,box_select,wheel_zoom,reset", plot_width=450,
            plot_height=400, extent=None):
    """
    Make a light-weight loc figure
    extent is half size of (square) lm plot.
    """

    fields = ['l1', 'm1', 'sizes', 'colors', 'snrs', 'keys']

    if not len(circleinds):
        circleinds = list(range(len(data['snrs'])))

    # set ranges
    inds = circleinds + crossinds + edgeinds
    l1 = [data['l1'][i] for i in inds]
    m1 = [data['m1'][i] for i in inds]
    if extent is None:
        extent = max([max(m1), -min(m1), max(l1), -min(l1)])

    source = ColumnDataSource(data=dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds]))
                                        for (key, value) in list(data.items())
                                        if key in fields}))
    loc = Figure(plot_width=plot_width, plot_height=plot_height,
                 toolbar_location="left", x_axis_label='l1 (rad)',
                 y_axis_label='m1 (rad)', x_range=(-extent, extent),
                 y_range=(-extent, extent),
                 output_backend='webgl', tools=tools)
    loc.circle('l1', 'm1', size='sizes', fill_color='colors',
               line_color=None, fill_alpha=0.2, source=source)

    hover = loc.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('keys', '@keys')])

    return loc


def calcsize(values, sizerange=(4, 70), inds=None, plaw=2):
    """ Use set of values to calculate symbol size.
    values is a list of floats for candidate significance.
    inds is an optional list of indexes to use to calculate symbol size.
    Scaling of symbol size min max set by sizerange tuple (min, max).
    plaw is powerlaw scaling of symbol size from values
    """

    if inds:
        smax = max([abs(values[i]) for i in inds])
        smin = min([abs(values[i]) for i in inds])
    else:
        smax = max([abs(val) for val in values])
        smin = min([abs(val) for val in values])

    if smax == smin:
        return [sizerange[1]]*len(values)
    else:
        return [sizerange[0] + sizerange[1] * ((abs(val) - smin)/(smax - smin))**plaw for val in values]


def colorsat(l, m):
    """ Returns color for given l,m
    Designed to look like a color wheel that is more saturated in middle.
    """

    lm = np.zeros(len(l), dtype='complex')
    lm.real = l
    lm.imag = m
    red = 0.5*(1+np.cos(np.angle(lm)))
    green = 0.5*(1+np.cos(np.angle(lm) + 2*3.14/3))
    blue = 0.5*(1+np.cos(np.angle(lm) - 2*3.14/3))
    amp = np.where(lm == 0, 256, 256*np.abs(lm)/np.abs(lm).max())
    return ["#%02x%02x%02x" % (np.floor(amp[i]*red[i]).astype(int),
            np.floor(amp[i]*green[i]).astype(int),
            np.floor(amp[i]*blue[i]).astype(int))
            for i in range(len(l))]


def calcinds(data, threshold, ignoret=None):
    """ Find indexes for data above (or below) given threshold. """

    inds = []
    for i in range(len(data['time'])):
        snr = data['snrs'][i]
        time = data['time'][i]
        if (threshold >= 0 and snr > threshold):
            if ignoret:
                incl = [t0 for (t0, t1) in ignoret if np.round(time).astype(int) in range(t0,t1)]
                logger.debug('{} {} {} {}'.format(np.round(time).astype(int), t0, t1, incl))
                if not incl:
                    inds.append(i)
            else:
                inds.append(i)
        elif threshold < 0 and snr < threshold:
            if ignoret:
                incl = [t0 for (t0, t1) in ignoret if np.round(time).astype(int) in range(t0,t1)]
                logger.debug('{} {} {} {}'.format(np.round(time).astype(int), t0, t1, incl))
                if not incl:
                    inds.append(i)
            else:
                inds.append(i)

    return inds


def candplot(canddatalist, snrs=None, cluster=None, outname=''):
    """ Takes output of search_thresh (CandData objects) to make
    candidate plots.
    Expects pipeline state, candidate location, image, and
    phased, dedispersed data (cut out in time, dual-pol).
    snrs is array for an (optional) SNR histogram plot.
    cluster allows cluster info to be passed in as (cluster_label, size).
    Written by Bridget Andersen and modified by Casey for rfpipe.
    """

    if not isinstance(canddatalist, list):
        logger.debug('Wrapping solo CandData object')
        canddatalist = [canddatalist]

    logger.info('Making {0} candidate plots.'.format(len(canddatalist)))

    for i in range(len(canddatalist)):
        canddata = canddatalist[i]
        st = canddata.state
        candloc = canddata.loc
        im = canddata.image
        data = canddata.data

        scan = st.metadata.scan
        segment, candint, dmind, dtind, beamnum = candloc

        # calc source location
#        imstd = util.madtostd(im)  # outlier resistant
        imstd = im.std()  # consistent with rfgpu
        snrim = im.max()/imstd
        l1, m1 = st.pixtolm(np.where(im == im.max()))

        logger.info('Plotting candloc {0} with SNR {1:.1f} and image/data shapes: {2}/{3}'
                    .format(str(candloc), snrim, str(im.shape), str(data.shape)))

        # either standard radec or otf phasecenter radec
        if st.otfcorrections is not None:
            ints, pt_ra_deg, pt_dec_deg = st.otfcorrections[segment][0]
            pt_ra = np.radians(pt_ra_deg)
            pt_dec = np.radians(pt_dec_deg)
        else:
            pt_ra, pt_dec = st.metadata.radec
        src_ra, src_dec = source_location(pt_ra, pt_dec, l1, m1)
        logger.info('Peak (RA, Dec): ({0}, {1})'.format(src_ra, src_dec))

        # convert l1 and m1 from radians to arcminutes
        l1arcm = np.degrees(l1)*60
        m1arcm = np.degrees(m1)*60

        # build overall plot
        fig = plt.Figure(figsize=(12.75, 8))

        # add metadata in subfigure
        ax = fig.add_subplot(2, 3, 1, facecolor='white')

        # calculate the overall dispersion delay: dd
        f1 = st.metadata.freq_orig[0]
        f2 = st.metadata.freq_orig[-1]
        dd = 4.15*st.dmarr[dmind]*(f1**(-2)-f2**(-2))

        # add annotating info
        # set spacing and location of the annotating information
        start = 1.1
        space = 0.07
        left = 0.0
        ax.text(left, start, st.fileroot, fontname='sans-serif',
                transform=ax.transAxes, fontsize='small')
        ax.text(left, start-space, 'Peak (arcmin): ('
                + str(np.round(l1arcm, 3)) + ', '
                + str(np.round(m1arcm, 3)) + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        # split the RA and Dec and display in a nice format
        ra = src_ra.split()
        dec = src_dec.split()
        ax.text(left, start-2*space, 'Peak (RA, Dec): (' + ra[0] + ':' + ra[1]
                + ':' + ra[2][0:4] + ', ' + dec[0] + ':' + dec[1] + ':'
                + dec[2][0:4] + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-3*space, 'Source: ' + str(st.metadata.source),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-4*space, 'scan: ' + str(scan),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-5*space, 'segment: ' + str(segment),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-6*space, 'integration: ' + str(candint),
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-7*space, 'DM = ' + str(st.dmarr[dmind])
                + ' (index ' + str(dmind) + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-8*space, 'dt = '
                + str(np.round(st.inttime*st.dtarr[dtind], 3)*1e3)
                + ' ms' + ' (index ' + str(dtind) + ')',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        ax.text(left, start-9*space, 'disp delay = ' + str(np.round(dd, 1))
                + ' ms',
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')
        defstr = 'SNR (im'
        snrstr = str(np.round(snrim, 1))
        if canddata.snrk is not None:
            defstr += '/k): '
            snrstr += '/' + str(np.round(canddata.snrk, 1))
        else:
            defstr += '): '
        ax.text(left, start-10*space, defstr+snrstr,
                fontname='sans-serif', transform=ax.transAxes,
                fontsize='small')

        if cluster is not None:
            label, size = cluster
            ax.text(left, start-11*space, 'Cluster label: {0}'.format(str(label)),
                    fontname='sans-serif',
                    transform=ax.transAxes, fontsize='small')
            ax.text(left, start-12*space, 'Cluster size: {0}'.format(size),
                    fontname='sans-serif', transform=ax.transAxes,
                    fontsize='small')

        # set the plot invisible so that it doesn't interfere with annotations
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')

        # plot full dynamic spectra
        left, width = 0.75, 0.2*2./3.
        bottom, height = 0.2, 0.7
        # three rectangles for each panel of the spectrum (RR, RR+LL, LL)
        rect_dynsp1 = [left, bottom, width/3., height]
        rect_dynsp2 = [left+width/3., bottom, width/3., height]
        rect_dynsp3 = [left+2.*width/3., bottom, width/3., height]
        rect_lc1 = [left, bottom-0.1, width/3., 0.1]
        rect_lc2 = [left+width/3., bottom-0.1, width/3., 0.1]
        rect_lc3 = [left+2.*width/3., bottom-0.1, width/3., 0.1]
        rect_sp = [left+width, bottom, 0.1*2./3., height]
        ax_dynsp1 = fig.add_axes(rect_dynsp1)
        # sharey so that axes line up
        ax_dynsp2 = fig.add_axes(rect_dynsp2, sharey=ax_dynsp1)
        ax_dynsp3 = fig.add_axes(rect_dynsp3, sharey=ax_dynsp1)
        # hide RR+LL and LL dynamic spectra y labels to avoid overlap
        [label.set_visible(False) for label in ax_dynsp2.get_yticklabels()]
        [label.set_visible(False) for label in ax_dynsp3.get_yticklabels()]
        ax_sp = fig.add_axes(rect_sp, sharey=ax_dynsp3)
        [label.set_visible(False) for label in ax_sp.get_yticklabels()]
        ax_lc1 = fig.add_axes(rect_lc1)
        ax_lc2 = fig.add_axes(rect_lc2, sharey=ax_lc1)
        ax_lc3 = fig.add_axes(rect_lc3, sharey=ax_lc1)
        [label.set_visible(False) for label in ax_lc2.get_yticklabels()]
        [label.set_visible(False) for label in ax_lc3.get_yticklabels()]

        # now actually plot the data
        spectra = np.swapaxes(data.real, 0, 1)
        dd1 = spectra[..., 0]
        dd2 = spectra[..., 0] + spectra[..., 1]
        dd3 = spectra[..., 1]
        colormap = 'viridis'
        logger.debug('{0}'.format(dd1.shape))
        logger.debug('{0}'.format(dd2.shape))
        logger.debug('{0}'.format(dd3.shape))
        _ = ax_dynsp1.imshow(dd1, origin='lower', interpolation='nearest',
                             aspect='auto', cmap=plt.get_cmap(colormap))
        _ = ax_dynsp2.imshow(dd2, origin='lower', interpolation='nearest',
                             aspect='auto', cmap=plt.get_cmap(colormap))
        _ = ax_dynsp3.imshow(dd3, origin='lower', interpolation='nearest',
                             aspect='auto', cmap=plt.get_cmap(colormap))
        ax_dynsp1.set_yticks(list(range(0, len(st.freq), 30)))
        ax_dynsp1.set_yticklabels(st.freq[::30].round(3))
        ax_dynsp1.set_ylabel('Freq (GHz)')
        ax_dynsp1.set_xlabel('RR')
        ax_dynsp1.xaxis.set_label_position('top')
        ax_dynsp2.set_xlabel('RR+LL')
        ax_dynsp2.xaxis.set_label_position('top')
        ax_dynsp3.set_xlabel('LL')
        ax_dynsp3.xaxis.set_label_position('top')
        # hide xlabels invisible so that they don't interefere with lc plots
        [label.set_visible(False) for label in ax_dynsp1.get_xticklabels()]
        # This one y label was getting in the way
        ax_dynsp1.get_yticklabels()[0].set_visible(False)
        # plot stokes I spectrum of the candidate pulse (assume middle bin)
        # select stokes I middle bin
        spectrum = spectra[:, canddata.integration_rel].mean(axis=1)
        ax_sp.plot(spectrum, list(range(len(spectrum))), 'k.')
        # plot 0 Jy dotted line
        ax_sp.plot(np.zeros(len(spectrum)), list(range(len(spectrum))), 'r:')
        xmin, xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(np.linspace(xmin, xmax, 3).round(2))
        ax_sp.set_xlabel('Flux (Jy)')

        # plot mean flux values for each time bin
        lc1 = dd1.mean(axis=0)
        lc2 = dd2.mean(axis=0)
        lc3 = dd3.mean(axis=0)
        lenlc = len(data)
        ax_lc1.plot(list(range(0, lenlc)), list(lc1)[:lenlc], 'k.')
        ax_lc2.plot(list(range(0, lenlc)), list(lc2)[:lenlc], 'k.')
        ax_lc3.plot(list(range(0, lenlc)), list(lc3)[:lenlc], 'k.')
        # plot 0 Jy dotted line for each plot
        ax_lc1.plot(list(range(0, lenlc)), list(np.zeros(lenlc)), 'r:')
        ax_lc2.plot(list(range(0, lenlc)), list(np.zeros(lenlc)), 'r:')
        ax_lc3.plot(list(range(0, lenlc)), list(np.zeros(lenlc)), 'r:')
        ax_lc2.set_xlabel('Integration (rel)')
        ax_lc1.set_ylabel('Flux (Jy)')
        ax_lc1.set_xticks([0, 0.5*lenlc, lenlc])
        # only show the '0' label for one of the plots to avoid messy overlap
        ax_lc1.set_xticklabels(['0', str(lenlc//2), str(lenlc)])
        ax_lc2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_lc2.set_xticklabels(['', str(lenlc//2), str(lenlc)])
        ax_lc3.set_xticks([0, 0.5*lenlc, lenlc])
        ax_lc3.set_xticklabels(['', str(lenlc//2), str(lenlc)])
        ymin, ymax = ax_lc1.get_ylim()
        ax_lc1.set_yticks(np.linspace(ymin, ymax, 3).round(2))

        # adjust the x tick marks to line up with the lc plots
        ax_dynsp1.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp3.set_xticks([0, 0.5*lenlc, lenlc])

        # plot second set of dynamic spectra
        left, width = 0.45, 0.1333
        bottom, height = 0.1, 0.4
        rect_dynsp1 = [left, bottom, width/3., height]
        rect_dynsp2 = [left+width/3., bottom, width/3., height]
        rect_dynsp3 = [left+2.*width/3., bottom, width/3., height]
        rect_sp = [left+width, bottom, 0.1*2./3., height]
        ax_dynsp1 = fig.add_axes(rect_dynsp1)
        ax_dynsp2 = fig.add_axes(rect_dynsp2, sharey=ax_dynsp1)
        ax_dynsp3 = fig.add_axes(rect_dynsp3, sharey=ax_dynsp1)
        # hide RR+LL and LL dynamic spectra y labels
        [label.set_visible(False) for label in ax_dynsp2.get_yticklabels()]
        [label.set_visible(False) for label in ax_dynsp3.get_yticklabels()]
        ax_sp = fig.add_axes(rect_sp, sharey=ax_dynsp3)
        [label.set_visible(False) for label in ax_sp.get_yticklabels()]

        # calculate the channels to average together for SNR=2
        n = int((2.*(len(spectra))**0.5/snrim)**2)
        if n == 0:  # if n==0 then don't average
            dd1avg = dd1
            dd3avg = dd3
        else:
            # otherwise, add zeros onto the data so that it's length is cleanly
            # divisible by n (makes it easier to average over)
            dd1zerotemp = np.concatenate((np.zeros((n-len(spectra) % n,
                                                    len(spectra[0])),
                                         dtype=dd1.dtype), dd1), axis=0)
            dd3zerotemp = np.concatenate((np.zeros((n-len(spectra) % n,
                                                    len(spectra[0])),
                                         dtype=dd3.dtype), dd3), axis=0)
            # make masked arrays so appended zeros do not affect average
            zeros = np.zeros((len(dd1), len(dd1[0])))
            ones = np.ones((n-len(spectra) % n, len(dd1[0])))
            masktemp = np.concatenate((ones, zeros), axis=0)
            dd1zero = np.ma.masked_array(dd1zerotemp, mask=masktemp)
            dd3zero = np.ma.masked_array(dd3zerotemp, mask=masktemp)
            # average together the data
            dd1avg = np.array([], dtype=dd1.dtype)
            for i in range(len(spectra[0])):
                temp = dd1zero[:, i].reshape(-1, n)
                tempavg = np.reshape(np.mean(temp, axis=1), (len(temp), 1))
                # repeats the mean values to create more pixels
                # (easier to properly crop when it is finally displayed)
                temprep = np.repeat(tempavg, n, axis=0)
                if i == 0:
                    dd1avg = temprep
                else:
                    dd1avg = np.concatenate((dd1avg, temprep), axis=1)
            dd3avg = np.array([], dtype=dd3.dtype)
            for i in range(len(spectra[0])):
                temp = dd3zero[:, i].reshape(-1, n)
                tempavg = np.reshape(np.mean(temp, axis=1), (len(temp), 1))
                temprep = np.repeat(tempavg, n, axis=0)
                if i == 0:
                    dd3avg = temprep
                else:
                    dd3avg = np.concatenate((dd3avg, temprep), axis=1)
        dd2avg = dd1avg + dd3avg  # add together to get averaged RR+LL spectrum
        colormap = 'viridis'
        # if n==0 then don't crop the spectra because no zeroes were appended
        if n == 0:
            dd1avgcrop = dd1avg
            dd2avgcrop = dd2avg
            dd3avgcrop = dd3avg
        else:  # otherwise, crop off the appended zeroes
            dd1avgcrop = dd1avg[len(ones):len(dd1avg), :]
            dd2avgcrop = dd2avg[len(ones):len(dd2avg), :]
            dd3avgcrop = dd3avg[len(ones):len(dd3avg), :]
        logger.debug('{0}'.format(dd1avgcrop.shape))
        logger.debug('{0}'.format(dd2avgcrop.shape))
        logger.debug('{0}'.format(dd3avgcrop.shape))
        _ = ax_dynsp1.imshow(dd1avgcrop, origin='lower',
                             interpolation='nearest', aspect='auto',
                             cmap=plt.get_cmap(colormap))
        _ = ax_dynsp2.imshow(dd2avgcrop, origin='lower',
                             interpolation='nearest', aspect='auto',
                             cmap=plt.get_cmap(colormap))
        _ = ax_dynsp3.imshow(dd3avgcrop, origin='lower',
                             interpolation='nearest', aspect='auto',
                             cmap=plt.get_cmap(colormap))
        spw_reffreq = np.sort(st.metadata.spw_reffreq)
        # TODO: need to find best chan for label even for overlapping spw
        spw_chans = [np.abs(reffreq/1e9-st.freq).argmin() for reffreq in spw_reffreq]
        ax_dynsp1.set_yticks(spw_chans)
        ax_dynsp1.set_yticklabels((spw_reffreq/1e9).round(3))
        ax_dynsp1.set_ylabel('Freq of SPW (GHz)')
        ax_dynsp1.set_xlabel('RR')
        ax_dynsp1.xaxis.set_label_position('top')
        ax_dynsp2.set_xlabel('Integration (rel)')
        ax2 = ax_dynsp2.twiny()
        ax2.set_xlabel('RR+LL')
        [label.set_visible(False) for label in ax2.get_xticklabels()]
        ax_dynsp3.set_xlabel('LL')
        ax_dynsp3.xaxis.set_label_position('top')

        # plot stokes I spectrum of the candidate pulse from middle integration
        ax_sp.plot(dd2avgcrop[:, canddata.integration_rel]/2.,
                   list(range(len(dd2avgcrop))), 'k.')
        ax_sp.plot(np.zeros(len(dd2avgcrop)), list(range(len(dd2avgcrop))),
                   'r:')
        xmin, xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(np.linspace(xmin, xmax, 3).round(2))
        ax_sp.get_xticklabels()[0].set_visible(False)
        ax_sp.set_xlabel('Flux (Jy)')

        # readjust the x tick marks on the dynamic spectra
        ax_dynsp1.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp1.set_xticklabels(['0', str(lenlc//2), str(lenlc)])
        ax_dynsp2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp2.set_xticklabels(['', str(lenlc//2), str(lenlc)])
        ax_dynsp3.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp3.set_xticklabels(['', str(lenlc//2), str(lenlc)])

        # plot the image and zoomed cutout
        ax = fig.add_subplot(2, 3, 4)
        fov = np.degrees(1./st.uvres)*60.
        _ = ax.imshow(im.transpose(), aspect='equal', origin='upper',
                      interpolation='nearest',
                      extent=[fov/2, -fov/2, -fov/2, fov/2],
                      cmap=plt.get_cmap('viridis'), vmin=0,
                      vmax=0.5*im.max())
        ax.set_xlabel('RA Offset (arcmin)')
        ax.set_ylabel('Dec Offset (arcmin)')
        # to set scale when we plot the triangles that label the location
        ax.autoscale(False)
        # add markers on the axes at measured position of the candidate
        ax.scatter(x=[l1arcm], y=[-fov/2], c='#ffff00', s=60, marker='^',
                   clip_on=False)
        ax.scatter(x=[fov/2], y=[m1arcm], c='#ffff00', s=60, marker='>',
                   clip_on=False)
        # makes it so the axis does not intersect the location triangles
        ax.set_frame_on(False)

        # add a zoomed cutout image of the candidate (set width at 5*beam)
        sbeam = np.mean(st.beamsize_deg)*60
        # figure out the location to center the zoomed image on
        xratio = len(im[0])/fov  # pix/arcmin
        yratio = len(im)/fov  # pix/arcmin
        mult = 5  # sets how many times the synthesized beam the zoomed FOV is
        xmin = max(0, int(len(im[0])//2-(m1arcm+sbeam*mult)*xratio))
        xmax = int(len(im[0])//2-(m1arcm-sbeam*mult)*xratio)
        ymin = max(0, int(len(im)//2-(l1arcm+sbeam*mult)*yratio))
        ymax = int(len(im)//2-(l1arcm-sbeam*mult)*yratio)
        left, width = 0.231, 0.15
        bottom, height = 0.465, 0.15
        rect_imcrop = [left, bottom, width, height]
        ax_imcrop = fig.add_axes(rect_imcrop)
        logger.debug('{0}'.format(im.transpose()[xmin:xmax, ymin:ymax].shape))
        logger.debug('{0} {1} {2} {3}'.format(xmin, xmax, ymin, ymax))
        _ = ax_imcrop.imshow(im.transpose()[xmin:xmax,ymin:ymax], aspect=1,
                             origin='upper', interpolation='nearest',
                             extent=[-1, 1, -1, 1],
                             cmap=plt.get_cmap('viridis'), vmin=0,
                             vmax=0.5*im.max())
        # setup the axes
        ax_imcrop.set_ylabel('Dec (arcmin)')
        ax_imcrop.set_xlabel('RA (arcmin)')
        ax_imcrop.xaxis.set_label_position('top')
        ax_imcrop.xaxis.tick_top()
        xlabels = [str(np.round(l1arcm+sbeam*mult/2, 1)), '',
                   str(np.round(l1arcm, 1)), '',
                   str(np.round(l1arcm-sbeam*mult/2, 1))]
        ylabels = [str(np.round(m1arcm-sbeam*mult/2, 1)), '',
                   str(np.round(m1arcm, 1)), '',
                   str(np.round(m1arcm+sbeam*mult/2, 1))]
        ax_imcrop.set_xticklabels(xlabels)
        ax_imcrop.set_yticklabels(ylabels)
        # change axis label loc of inset to avoid the full picture
        ax_imcrop.get_yticklabels()[0].set_verticalalignment('bottom')

        # create SNR versus N histogram for the whole observation
        # (properties for each candidate in the observation given by prop)
        if snrs is not None:
            left, width = 0.45, 0.2
            bottom, height = 0.6, 0.3
            rect_snr = [left, bottom, width, height]
            ax_snr = fig.add_axes(rect_snr)
            pos_snrs = snrs[snrs >= 0]
            neg_snrs = snrs[snrs < 0]
            if not len(neg_snrs):  # if working with subset and only pos snrs
                neg_snrs = pos_snrs
                nonegs = True
            else:
                nonegs = False
            minval = 5.5
            maxval = 8.0
            # determine the min and max values of the x axis
            if min(pos_snrs) < min(np.abs(neg_snrs)):
                minval = min(pos_snrs)
            else:
                minval = min(np.abs(neg_snrs))
            if max(pos_snrs) > max(np.abs(neg_snrs)):
                maxval = max(pos_snrs)
            else:
                maxval = max(np.abs(neg_snrs))

            # positive SNR bins are in blue
            # absolute values of negative SNR bins are taken and plotted as
            # red x's on top of positive blue bins for compactness
            n, b, patches = ax_snr.hist(pos_snrs, 50, (minval, maxval),
                                        facecolor='blue', zorder=1)
            vals, bin_edges = np.histogram(np.abs(neg_snrs), 50,
                                           (minval, maxval))
            bins = np.array([(bin_edges[i]+bin_edges[i+1])/2.
                             for i in range(len(vals))])
            vals = np.array(vals)
            if not nonegs:
                ax_snr.scatter(bins[vals > 0], vals[vals > 0], marker='x',
                               c='orangered', alpha=1.0, zorder=2)
            ax_snr.set_xlabel('SNR')
            ax_snr.set_xlim(left=minval-0.2)
            ax_snr.set_xlim(right=maxval+0.2)
            ax_snr.set_ylabel('N')
            ax_snr.set_yscale('log')
            # draw vertical line where the candidate SNR is
            ax_snr.axvline(x=snrim, linewidth=1, color='y', alpha=0.7)

        if not outname:
            outname = os.path.join(st.prefs.workdir,
                                   'cands_{0}_seg{1}-i{2}-dm{3}-dt{4}.png'
                                   .format(st.fileroot, segment, candint,
                                           dmind, dtind))

        try:
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(outname)
            logger.info('Wrote candidate plot to {0}'.format(outname))
        except ValueError:
            logger.warning('Could not write figure to {0}'.format(outname))


def source_location(pt_ra, pt_dec, l1, m1):
    """ Takes phase center and src l,m in radians to get ra,dec of source.
    Returns string ('hh mm ss', 'dd mm ss')
    """

    srcra = np.degrees(pt_ra + l1/cos(pt_dec))
    srcdec = np.degrees(pt_dec + m1)

    return deg2HMS(srcra, srcdec)


def deg2HMS(ra=None, dec=None, round=False):
    """ quick and dirty coord conversion. googled to find bdnyc.org.
    """
    RA, DEC, rs, ds = '', '', '', ''
    if dec is not None:
        if str(dec)[0] == '-':
            ds, dec = '-', abs(dec)
        deg = int(dec)
        decM = abs(int((dec-deg)*60))
        if round:
            decS = int((abs((dec-deg)*60)-decM)*60)
        else:
            decS = (abs((dec-deg)*60)-decM)*60
        DEC = '{0}{1} {2} {3}'.format(ds, deg, decM, decS)

    if ra is not None:
        if str(ra)[0] == '-':
            rs, ra = '-', abs(ra)
        raH = int(ra/15)
        raM = int(((ra/15)-raH)*60)
        if round:
            raS = int(((((ra/15)-raH)*60)-raM)*60)
        else:
            raS = ((((ra/15)-raH)*60)-raM)*60
        RA = '{0}{1} {2} {3}'.format(rs, raH, raM, raS)

    if ra is not None and dec is not None:
        return (RA, DEC)
    else:
	return RA or DEC


def make_voevent(candcollection):    
    """ Script to generate a VOEvent file from the CandCollection 
    Takes Candcollection info and writes a .xml file with relevant inforation
    VOEvent format based on Petroff et al. 2017 VOEvent Standard for Fast Radio Busrts
    See https://github.com/ebpetroff/FRB_VOEvent
    written by Justin D. Linford with input from Casey Law, Sarah Burke-Spolaor
    and Kshitij Aggarwal
    --please excuse my terrible, self-taught python habits--
    """
    from astropy.coordinates import SkyCoord  #needed to get galactic coordinates of FRB
    from astropy.time import Time #needed to get MJD to ISOTime
    import math #need to get RA & DEC into degrees
    
    #get candata separated into useful parts
    st = candcollection.state
    
    #LOOP TO STEP THROUGH ENTREES IN CANDCOLLECTION
    
    for n1 in range(len(candcollection.locs)):
    
        candloc = candcollection.locs[n1]
        #get some usefult info out of candidate location
        segment = candcollection.segment
        candint = candloc[1]
        dmind = candloc[2]
        dtind = candloc[3]
        beamnum = candloc[4]
        
        #Basic data easily accessible from CandCollection
        FRB_DM = candcollection.canddm[n1]
        #FRB_DM_err = -999 #TODO: need to figure out how to get DM nucertainty
        #DM uncertainty: From Cordes & McLaughlin 2003, FWHM in S/N vs delDM distribution should be 
        #delta-DM ~ 506 * pulse width [ms] * observing freq ^3 [Ghz] / bandwidth [MHz]  (Eq. 14)
        #by definition, FWHM = 2*sqrt(2*ln2)*sigma for a Gaussian
        #DM_err ~ 506/(2*sqrt(2 ln2)) * Width(ms) * ObsFrequency(GHz)^3 / bandwidth(MHz)
        FRB_obsmjd = candcollection.candmjd[n1]
        FRB_width = candcollection.canddt[n1]*1.0e3 #approximate pulse width in ms
        snr1 = candcollection.array['snr1'].flatten()
        FRB_SNR = snr1[n1]
        l1 = candcollection.candl[n1]
        m1 = candcollection.candm[n1]
        
        #get FRB RA & DEC location in degrees --> NOTE: Stole this from source_location
        pt_ra, pt_dec = st.metadata.radec
        srcra = np.degrees(pt_ra + l1/math.cos(pt_dec))
        srcdec = np.degrees(pt_dec + m1)
        im_pix_scale = np.degrees((st.npixx*st.uvres)**-1.0) #degrees per pixel
        srcloc_err = im_pix_scale #set source location uncertainty to the pixel scale, for now --> assumes source only fills a single pixel
        #put location into SkyCoord
        FRB_loc = SkyCoord(srcra,srcdec,frame='icrs',unit='deg')
        #FRB galactic coordinates
        FRB_gl = FRB_loc.galactic.l.deg
        FRB_gb = FRB_loc.galactic.b.deg
        
        #WHAT fields
        #observatory parameters
        beam_size = st.beamsize_deg #estimate of beam size in degrees
        #TODO: is st.beamsize)deg an estimate of the primary beam or the restoring beam?
        beam_semimaj = max(beam_size) * 3600.0 # TODO: figure out how to get this info
        beam_semimin = min(beam_size) * 3600.0 # TODO: figure out how to get this info
        beam_rot_ang = -999 # TODO: figure out how to get this info
        samp_time = np.round(st.inttime, 3)*1e3 #sampling time in ms
        band_width = np.round((st.freq.max() - st.freq.min())*1.0e3,4)#bandwidth in MHz
        num_chan = np.sum(st.metadata.spw_nchan)
        center_freq = st.metadata.spw_reffreq[int(len(st.metadata.spw_reffreq)/2.0)]/1.0e6 #should be center freq in MHz
        num_pol = int(len(st.metadata.pols_orig))
        bits_per_sample = 2 #TODO: check that this is always accurate
        gain_KJy = -999 #TODO: figure out how to get this info
        Tsys_K = -999 #TODO: figure out how to get this info
        VLA_backend = 'WIDAR' #may need to find out how to get this info from the data
        VLA_beam = beamnum #VLA only has a single beam
        
        #should now have all the necessary numbers to calculate DM uncertainty
        FRB_DM_err = (506.0/(2.0*np.sqrt(2.0*np.log(2.0)))) * FRB_width * (center_freq*1.0e-3)**3 / band_width
        
        #now compare beam size to pixel scale
        #if the 1/2 beam semi-minor axis is larger than the pixel scale, set the location uncertainty to 1/2 the semi-minor axis
        if 0.5*min(beam_size)>im_pix_scale: srcloc_err = 0.5*min(beam_size)
        
        FRB_obstime = Time(FRB_obsmjd,format='mjd',scale='utc')
        #print(FRB_obstime)
        FRB_ISOT = FRB_obstime.isot #convert time to ISOT
        #print(FRB_ISOT)
        #get the hour of the observation for FRB name
        t_pos = FRB_ISOT.find('T')
        FRB_ISOT_UTHH = 'UT'+FRB_ISOT[t_pos+1:t_pos+3]
        
        #Importance parameter
        FRB_importance = 0.8 #default to relatively high importance #TODO: look into setting this based on candidate decision trees or SNR
        if candcollection.clustersize is not None:
            if candcollection.clustersize[n1]>10. and FRB_SNR>30.: FRB_importance=1.0
            if candcollection.clustersize[n1]>10. and FRB_SNR<30. and FRB_SNR>20.: FRB_importance=0.9
            if candcollection.clustersize[n1]<5.0 and FRB_SNR<20.0: FRB_importance=0.5
        else:
            if FRB_SNR>30.: FRB_importance=0.95
            if FRB_SNR>20. and FRB_SNR<30.: FRB_importance=0.85
        
        #build FRB name
        FRB_YY = FRB_ISOT[2:4] #last 2 digits of year
        FRB_MM = FRB_ISOT[5:7] #2-digit month
        FRB_DD = FRB_ISOT[8:10] #2-digit day
    
        FRB_RADEC_str = FRB_loc.to_string('hmsdms') #convert FRB coordinates to HH:MM:SS.SSSS (+/-)DD:MM:SS.SSSS
        
        #FRB_NAME = 'FRB'+FRB_YY+FRB_MM+FRB_DD + '.J' + FRB_RAhh+FRB_RAmm+FRB_RAss + FRB_DECdd+FRB_DECmm+FRB_DECss
        FRB_NAME = 'FRB'+FRB_YY+FRB_MM+FRB_DD + FRB_ISOT_UTHH
        
        #set filename to FRB_NAME + '_detection.xml'
        outname = os.path.join(st.prefs.workdir,FRB_NAME+'_detection.xml')
        
        try:
            #write VOEvent file
            #create a text file with all the VLA fluxes to include in paper
            VOEvent_of = open(outname,'w')
            #header
            VOEvent_of.write("<?xml version='1.0' encoding='UTF-8'?>"+'\n')
            VOEvent_of.write('<voe:VOEvent xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:voe="http://www.ivoa.net/xml/VOEvent/v2.0" xsi:schemaLocation="http://www.ivoa.net/xml/VOEvent/v2.0 http://www.ivoa.net/xml/VOEvent/VOEvent-v2.0.xsd" version="2.0" role="test" ivorn="ivo://realfast.io/realfast#'+FRB_NAME+'/'+str(FRB_obsmjd)+'">'+'\n')
            #WHO
            VOEvent_of.write('\t'+'<Who>'+'\n')
            VOEvent_of.write('\t\t'+'<AuthorIVORN>ivo://realfast.io/contact</AuthorIVORN>'+'\n')
            VOEvent_of.write('\t\t'+'<Date>'+FRB_ISOT+'</Date>\n')
            VOEvent_of.write('\t\t'+'<Author><contactEmail>claw@astro.berkeley.edu</contactEmail><contactName>Casey Law</contactName></Author>\n')
            VOEvent_of.write('\t</Who>\n')
            #What
            VOEvent_of.write('\t<What>\n')
            VOEvent_of.write('\t\tParam name="AlertType" dataType="string" value="Preliminary">\n')
            VOEvent_of.write('\t\t</Param>\n')
            VOEvent_of.write('\t\t<Group name="observatory parameters">\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="beam_semi-major_axis" ucd="instr.beam;pos.errorEllipse;phys.angSize.smajAxis" unit="SS" value="'+str(beam_semimaj)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="beam_semi-minor_axis" ucd="instr.beam;pos.errorEllipse;phys.angSize.sminAxis" unit="SS" value="'+str(beam_semimin)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="beam_rotation_angle" ucd="instr.beam;pos.errorEllipse;instr.offset" unit="Degrees" value="'+str(beam_rot_ang)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="sampling_time" ucd="time.resolution" unit="ms" value="'+str(samp_time)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="bandwidth" ucd="instr.bandwidth" unit="MHz" value="'+str(band_width)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="int" name="nchan" ucd="meta.number;em.freq;em.bin" unit="None" value="'+str(num_chan)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="centre_frequency" ucd="em.freq;instr" unit="MHz" value="'+str(center_freq)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="int" name="npol" unit="None" value="'+str(num_pol)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="int" name="bits_per_sample" unit="None" value="'+str(bits_per_sample)+'"/>\n')
            #VOEvent_of.write('\t\t\t<Param dataType="float" name="gain" unit="K/Jy" value="'+str(gain_KJy)+'"/>\n')  #FOR NOW: do not report gain
            #VOEvent_of.write('\t\t\t<Param dataType="float" name="tsys" ucd="phot.antennaTemp" unit="K" value="'+str(Tsys_K)+'"/>\n')  #FOR NOW: do not report Tsys
            VOEvent_of.write('\t\t\t<Param name="backend" value="'+VLA_backend+'"/>\n')
            #VOEvent_of.write('\t\t\t<Param name="beam" value="'+str(VLA_beam)+'"/><Description>Detection beam number if backend is a multi beam receiver</Description>\n')
            VOEvent_of.write('\t\t</Group>\n')
            VOEvent_of.write('\t\t<Group name="event parameters">\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="dm" ucd="phys.dispMeasure;em.radio.'+str(int(np.floor(st.freq.min())))+'000-'+str(int(np.ceil(st.freq.max())))+'000MHz" unit="pc/cm^3" value="'+str(FRB_DM)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="dm_error" ucd="stat.error;phys.dispMeasure" unit="pc/cm^3" value="'+str(int(np.ceil(FRB_DM_err)))+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="width" ucd="time.duration;src.var.pulse" unit="ms" value="'+str(FRB_width)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="snr" ucd="stat.snr" unit="None" value="'+str(FRB_SNR)+'"/>\n')
            #VOEvent_of.write('\t\t\t<Param dataType="float" name="flux" ucd="phot.flux" unit="Jy" value="'+str(FRB_flux)+'"/>\n') #FOR NOW: do not report flux density.  We do not have good enough absolute flux density calibration
            VOEvent_of.write('\t\t\t<Param dataType="float" name="gl" ucd="pos.galactic.lon" unit="Degrees" value="'+str(FRB_gl)+'"/>\n')
            VOEvent_of.write('\t\t\t<Param dataType="float" name="gb" ucd="pos.galactic.lat" unit="Degrees" value="'+str(FRB_gb)+'"/>\n')
            VOEvent_of.write('\t\t</Group>\n')
            VOEvent_of.write('\t\t<Group name="advanced parameters">\n')
            #VOEvent_of.write('\t\t\t<Param dataType="float" name="MW_dm_limit" unit="pc/cm^3" value="34.9"/>\n')
            #VOEvent_of.write('\t\t\t\t</Param>\n')
            VOEvent_of.write('\t\t</Group>\n')
            VOEvent_of.write('\t</What>\n')
            #WhereWhen
            VOEvent_of.write('\t<WhereWhen>\n')
            VOEvent_of.write('\t\t<ObsDataLocation>\n')
            VOEvent_of.write('\t\t\t<ObservatoryLocation id="VLA">\n')
            VOEvent_of.write('\t\t\t<AstroCoordSystem id="UTC-GEOD-TOPO"/>\n')
            VOEvent_of.write('\t\t\t<AstroCoords coord_system_id="UTC-GEOD-TOPO">\n')
            VOEvent_of.write('\t\t\t<Position3D unit="deg-deg-m">\n')
            VOEvent_of.write('\t\t\t  <Value3>\n')
            VOEvent_of.write('\t\t\t    <C1>107.6184</C1>\n')
            VOEvent_of.write('\t\t\t    <C2>34.0784</C2>\n')
            VOEvent_of.write('\t\t\t    <C3>2124.456</C3>\n')
            VOEvent_of.write('\t\t\t  </Value3>\n')
            VOEvent_of.write('\t\t\t</Position3D>\n')
            VOEvent_of.write('\t\t\t</AstroCoords>\n')
            VOEvent_of.write('\t\t\t</ObservatoryLocation>\n')
            VOEvent_of.write('\t\t\t<ObservationLocation>\n')
            VOEvent_of.write('\t\t\t\t<AstroCoordSystem id="UTC-FK5-GEO"/><AstroCoords coord_system_id="UTC-FK5-GEO">\n')
            VOEvent_of.write('\t\t\t\t<Time unit="s"><TimeInstant><ISOTime>'+FRB_ISOT+'</ISOTime></TimeInstant></Time>\n')
            VOEvent_of.write('\t\t\t\t<Position2D unit="deg"><Name1>RA</Name1><Name2>Dec</Name2><Value2><C1>'+str(srcra)+'</C1><C2>'+str(srcdec)+'</C2></Value2><Error2Radius>'+str(srcloc_err)+'</Error2Radius></Position2D>\n')
            VOEvent_of.write('\t\t\t\t</AstroCoords>\n')
            VOEvent_of.write('\t\t\t</ObservationLocation>\n')
            VOEvent_of.write('\t\t</ObsDataLocation>\n')
            VOEvent_of.write('\t</WhereWhen>\n')
            #How
            VOEvent_of.write('\t<How>\n')
            VOEvent_of.write('\t\t<Description>Discovered by realfast</Description>')
            VOEvent_of.write('\t\t<Reference uri="http://realfast.io"/>')
            VOEvent_of.write('\t\t</How>\n')
            #Why
            VOEvent_of.write('\t<Why importance="'+str(FRB_importance)+'">\n')
            VOEvent_of.write('\t\t\t<Concept></Concept><Description>Detection of a new FRB by RealFast</Description>\n')
            VOEvent_of.write('\t\t<Name>'+FRB_NAME+'</Name>\n')
            VOEvent_of.write('\t</Why>\n')
            VOEvent_of.write('</voe:VOEvent>')
            
            #close file
            VOEvent_of.close()
            logger.info('Wrote VOEvent file to {0}'.format(outname))
            
        except ValueError:
            logger.warn('Could not write VOEvent file {0}'.format(outname))
