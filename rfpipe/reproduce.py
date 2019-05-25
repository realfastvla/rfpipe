from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import bytes, dict, object, range, map, input, str
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import pickle
import os.path
import numpy as np
from kalman_detector import kalman_prepare_coeffs, kalman_significance
import logging
logger = logging.getLogger(__name__)


def reproduce_candcollection(cc, data, wisdom=None, spec_std=None, sig_ts=None,
                             kalman_coeffs=None):
    """ Uses prepared data to make canddata for each cand in candcollection.
    Will look for cluster label and filter only for peak snr, if available.
    Location (e.g., integration, dm, dt) of each is used to create
    canddata for each candidate.
    Calculates features not used directly for search (as defined in
    state.prefs.calcfeatures).
    """

    from rfpipe import candidates

    # set up output cc
    st = cc.state
    cc1 = candidates.CandCollection(prefs=st.prefs, metadata=st.metadata)

    if len(cc):
        candlocs = cc.locs
        snrs = cc.snrtot

        if 'cluster' in cc.array.dtype.fields:
            clusters = cc.array['cluster'].astype(int)
            cl_rank, cl_count = candidates.calc_cluster_rank(cc)
            calcinds = np.unique(np.where(cl_rank == 1)[0]).tolist()
            logger.debug("Reproducing cands at {0} cluster peaks"
                         .format(len(calcinds)))

            # TODO: use number of clusters as test of an RFI-affected segment?
            # If threshold exceeded, could reproduce subset of all candidates.

        else:
            logger.debug("No cluster field found. Reproducing all.")
            calcinds = list(range(len(cc)))

        # reproduce canddata for each
        for i in calcinds:
            # TODO: check on best way to find max SNR with kalman, etc
            snr = snrs[i]
            candloc = candlocs[i]

            # kwargs passed to canddata object for plotting/saving
            kwargs = {}
            if 'cluster' in cc.array.dtype.fields:
                logger.info("Cluster {0}/{1} has {2} candidates and max detected SNR {3:.1f} at {4}"
                            .format(calcinds.index(i), len(calcinds)-1, cl_count[i],
                                    snr, candloc))
                # add supplementary plotting and cc info
                kwargs['cluster'] = clusters[i]
                kwargs['clustersize'] = cl_count[i]
            else:
                logger.info("Candidate {0}/{1} has detected SNR {2:.1f} at {3}"
                            .format(calcinds.index(i), len(calcinds)-1, snr, candloc))

            # reproduce candidate and get/calc features
            data_corr = pipeline_datacorrect(st, candloc, data_prep=data)

            for feature in st.searchfeatures:
                if feature in cc.array.dtype.fields:  # if already calculated
                    kwargs[feature] = cc.array[feature][i]
                else:  # if desired, but not yet calculated
                    if feature == 'snrk':
                        spec = data_corr.real.mean(axis=3).mean(axis=1)[candloc[1]]
                        significance_kalman = -kalman_significance(spec,
                                                                   spec_std,
                                                                   sig_ts=sig_ts,
                                                                   coeffs=kalman_coeffs)
                        snrk = (2*significance_kalman)**0.5
                        logger.info("Calculated snrk of {0} after detection. Adding it to CandData.".format(snrk))
                        kwargs[feature] = snrk
                    else:
                        logger.warning("Feature calculation {0} not yet supported"
                                       .format(feature))

            cd = pipeline_canddata(st, candloc, data_corr, sig_ts=sig_ts,
                                   kalman_coeffs=kalman_coeffs, **kwargs)
            if st.prefs.savecanddata:
                candidates.save_cands(st, canddata=cd)
            if st.prefs.saveplots:
                candidates.candplot(cd, snrs=snrs)  # snrs before clustering

            # regenerate cc with extra features in cd
            cc1 += candidates.cd_to_cc(cd)

    return cc1


def pipeline_dataprep(st, candloc):
    """ Prepare (read, cal, flag) data for a given state and candloc.
    """

    from rfpipe import source

    segment, candint, dmind, dtind, beamnum = candloc

    # propagate through to new candcollection
    st.prefs.segmenttimes = st._segmenttimes.tolist()

    # prep data
    data = source.read_segment(st, segment)
    flagversion = "rtpipe" if hasattr(st, "rtpipe_version") else "latest"
    data_prep = source.data_prep(st, segment, data, flagversion=flagversion)

    return data_prep


def pipeline_datacorrect(st, candloc, data_prep=None):
    """ Prepare and correct for dm and dt sampling of a given candloc
    Can optionally pass in prepared (flagged, calibrated) data, if available.
    """

    from rfpipe import util
    import rfpipe.search

    if data_prep is None:
        data_prep = pipeline_dataprep(st, candloc)

    segment, candint, dmind, dtind, beamnum = candloc
    dt = st.dtarr[dtind]
    dm = st.dmarr[dmind]

    scale = None
    if hasattr(st, "rtpipe_version"):
        scale = 4.2e-3 if st.rtpipe_version <= 1.54 else None
    delay = util.calc_delay(st.freq, st.freq.max(), dm, st.inttime,
                            scale=scale)

    data_dmdt = rfpipe.search.dedisperseresample(data_prep, delay, dt,
                                                 parallel=st.prefs.nthread > 1,
                                                 resamplefirst=st.fftmode=='cuda')

    return data_dmdt


def pipeline_canddata(st, candloc, data_dmdt=None, cpuonly=False, sig_ts=None,
                      kalman_coeffs=None, **kwargs):
    """ Generate image and phased visibility data for candloc.
    Phases to peak pixel in image of candidate.
    Can optionally pass in corrected data, if available.
    cpuonly argument not being used at present.
    """

    from rfpipe import candidates, util
    import rfpipe.search

    segment, candint, dmind, dtind, beamnum = candloc
    dt = st.dtarr[dtind]
    dm = st.dmarr[dmind]

    uvw = util.get_uvw_segment(st, segment)
    wisdom = rfpipe.search.set_wisdom(st.npixx, st.npixy)

    if data_dmdt is None:
        data_dmdt = pipeline_datacorrect(st, candloc)

    if 'snrk' in st.features:
        if data_dmdt.shape[0] > 1:
            spec_std = data_dmdt.real.mean(axis=3).mean(axis=1).std(axis=0)
        else:
            spec_std = data_dmdt[0].real.mean(axis=2).std(axis=0)

        if not np.any(spec_std):
            logger.warning("spectrum std all zeros. Not estimating coeffs.")
            kalman_coeffs = []
        else:
            sig_ts, kalman_coeffs = kalman_prepare_coeffs(spec_std)

        if not np.all(np.nan_to_num(sig_ts)):
            kalman_coeffs = []
    else:
        spec_std, sig_ts, kalman_coeffs = None, None, None

#    fftmode = 'fftw' if cpuonly else st.fftmode  # can't remember why i did this!
    image = rfpipe.search.grid_image(data_dmdt, uvw, st.npixx, st.npixy, st.uvres,
                                     'fftw', st.prefs.nthread, wisdom=wisdom,
                                     integrations=[candint])[0]

    # TODO: allow dl,dm as args and reproduce detection for other SNRs
    dl, dm = st.pixtolm(np.where(image == image.max()))
    util.phase_shift(data_dmdt, uvw, dl, dm)
    dataph = data_dmdt[max(0, candint-st.prefs.timewindow//2):candint+st.prefs.timewindow//2].mean(axis=1)
    util.phase_shift(data_dmdt, uvw, -dl, -dm)

    spec = data_dmdt.real.mean(axis=3).mean(axis=1)[candloc[1]]

    if 'snrk' in st.features:
        significance_kalman = -kalman_significance(spec, spec_std,
                                                   sig_ts=sig_ts,
                                                   coeffs=kalman_coeffs)
        snrk = (2*significance_kalman)**0.5
        logger.info("Calculated snrk of {0} after detection. Adding it to CandData.".format(snrk))
        kwargs['snrk'] = snrk

    canddata = candidates.CandData(state=st, loc=tuple(candloc), image=image,
                                   data=dataph, **kwargs)

    # output is as from searching functions
    return canddata


def pipeline_candidate(st, candloc, canddata=None):
    """ End-to-end pipeline to reproduce candidate plot and calculate features.
    Can optionally pass in canddata, if available.
    *TODO: confirm that cc returned by this has clustering and other enhanced features*
    """

    from rfpipe import candidates

    segment, candint, dmind, dtind, beamnum = candloc

    if canddata is None:
        canddata = pipeline_canddata(st, candloc)

    candcollection = candidates.cd_to_cc(canddata)

    return candcollection


def oldcands_read(candsfile, sdmscan=None):
    """ Read old-style candfile and create new-style candcollection
    Returns a list of tuples (state, dataframe) per scan.
    """

    with open(candsfile, 'rb') as pkl:
        try:
            d = pickle.load(pkl)
            ret = pickle.load(pkl)
        except UnicodeDecodeError:
            d = pickle.load(pkl, encoding='latin-1')
            ret = pickle.load(pkl, encoding='latin-1')
        if isinstance(ret, tuple):
            loc, prop = ret
        elif isinstance(ret, dict):
            loc = np.array(list(ret.keys()))
            prop = np.array(list(ret.values()))
        else:
            logger.warning("Not sure what we've got in this here cands pkl file...")

    if sdmscan is None:  # and (u'scan' in d['featureind']):
#        scanind = d['featureind'].index('scan')
        scanind = 0
        scans = np.unique(loc[:, scanind])
    elif sdmscan is not None:
        scans = [sdmscan]

    ll = []
    for scan in scans:
        try:
            st, cc = oldcands_readone(candsfile, scan)
            ll.append((st, cc))
        except AttributeError:
            pass

    return ll


def oldcands_readone(candsfile, scan=None):
    """ Reads old-style candidate files to create new state and candidate
    collection for a given scan.
    Parsing merged cands file requires sdm locally with bdf for given scan.
    If no scan provided, assumes candsfile is from single scan not merged.
    """

    from rfpipe import preferences, metadata, state, candidates

    with open(candsfile, 'rb') as pkl:
        try:
            d = pickle.load(pkl)
            ret = pickle.load(pkl)
        except UnicodeDecodeError:
            d = pickle.load(pkl, encoding='latin-1')
            ret = pickle.load(pkl, encoding='latin-1')
        if isinstance(ret, tuple):
            loc, prop = ret
        elif isinstance(ret, dict):
            loc = np.array(list(ret.keys()))
            prop = np.array(list(ret.values()))
        else:
            logger.warning("Not sure what we've got in this here cands pkl file...")

    # detect merged vs nonmerged
    if 'scan' in d['featureind']:
        locind0 = 1
    else:
        locind0 = 0

    # merged candsfiles must be called with scan arg
    if scan is None:
        assert locind0 == 0, "Set scan if candsfile has multiple scans."

    inprefs = preferences.oldstate_preferences(d, scan=scan)
    inprefs.pop('gainfile')
    inprefs.pop('workdir')
    inprefs.pop('fileroot')
    inprefs['segmenttimes'] = inprefs['segmenttimes']
    sdmfile = os.path.basename(d['filename'])

    try:
        assert scan is not None
        st = state.State(sdmfile=sdmfile, sdmscan=scan, inprefs=inprefs)
    except:
        meta = metadata.oldstate_metadata(d, scan=scan)
        st = state.State(inmeta=meta, inprefs=inprefs, showsummary=False)

    if 'rtpipe_version' in d:
        st.rtpipe_version = float(d['rtpipe_version'])  # TODO test this
        if st.rtpipe_version <= 1.54:
            logger.info('Candidates detected with rtpipe version {0}. All '
                        'versions <=1.54 used incorrect DM scaling.'
                        .format(st.rtpipe_version))

    if scan is None:
        assert locind0 == 0, "Set scan if candsfile has multiple scans."
        scan = d['scan']

    logger.info('Calculating candidate properties for scan {0}'.format(scan))

    if locind0 == 1:
        loc = loc[np.where(loc[:, 0] == scan)][:, locind0:]

    print(st.features, st.prefs.searchtype)
    fields = [str(ff) for ff in st.search_dimensions + st.features]
    types = [str(tt) for tt in len(st.search_dimensions)*['<i4'] + len(st.features)*['<f4']]
    dtype = np.dtype({'names': fields, 'formats': types})
    features = np.zeros(len(loc), dtype=dtype)
    for i in range(len(loc)):
        features[i] = tuple(list(loc[i]) + list(prop[i]))
    cc = candidates.CandCollection(features, st.prefs, st.metadata)

    return st, cc


def oldcands_convert(candsfile, scan=None):
    """ Take old style candsfile for a single scan and writes new style file.
    """

    st, cc = oldcands_readone(candsfile, scan=scan)
    with open(st.candsfile, 'wb') as pkl:
        pickle.dump(cc, pkl)
