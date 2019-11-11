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


def reproduce_candcollection(cc, data=None, wisdom=None, spec_std=None,
                             sig_ts=[], kalman_coeffs=[]):
    """ Uses candcollection to make new candcollection with required info.
    Will look for cluster label and filter only for peak snr, if available.
    Location (e.g., integration, dm, dt) of each is used to create
    canddata for each candidate, if required.
    Can calculates features not used directly for search (as defined in
    state.prefs.calcfeatures).
    """

    from rfpipe import candidates, util

    # set up output cc
    st = cc.state
    cc1 = candidates.CandCollection(prefs=st.prefs, metadata=st.metadata)

    if len(cc):
        if 'cluster' in cc.array.dtype.fields:
            clusters = cc.array['cluster'].astype(int)
            cl_rank, cl_count = candidates.calc_cluster_rank(cc)
            calcinds = np.unique(np.where(cl_rank == 1)[0]).tolist()
            logger.debug("Reproducing cands at {0} cluster peaks"
                         .format(len(calcinds)))
        else:
            logger.debug("No cluster field found. Reproducing all.")
            calcinds = list(range(len(cc)))

        # if candidates that need new feature calculations
        if not all([f in cc.array.dtype.fields for f in st.features]):
            logger.info("Generating canddata for {0} candidates"
                        .format(len(calcinds)))

            candlocs = cc.locs
            snrs = cc.snrtot
            normprob = candidates.normprob(snrs, st.ntrials)
            snrmax = snrs.max()
            logger.info('Zscore/SNR for strongest candidate: {0}/{1}'
                        .format(normprob[np.where(snrs == snrmax)[0]][0], snrmax))

            if ('snrk' in st.features and
                'snrk' not in cc.array.dtype.fields and
                (spec_std is None or not len(sig_ts) or not len(kalman_coeffs))):
                # TODO: use same kalman calc for search as reproduce?
                spec_std, sig_ts, kalman_coeffs = util.kalman_prep(data)

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
                                .format(calcinds.index(i), len(calcinds)-1, snr,
                                        candloc))

                # reproduce candidate and get/calc features
                data_corr = pipeline_datacorrect(st, candloc, data_prep=data)

                for feature in st.features:
                    if feature in cc.array.dtype.fields:  # if already calculated
                        kwargs[feature] = cc.array[feature][i]
                    else:  # if desired, but not calculated here or from canddata
                        if feature == 'snrk':
                            if 'snrk' not in cc.array.dtype.fields:
                                spec = data_corr.real.mean(axis=3).mean(axis=1)[candloc[1]]
                                significance_kalman = -kalman_significance(spec,
                                                                           spec_std,
                                                                           sig_ts=sig_ts,
                                                                           coeffs=kalman_coeffs)
                                snrk = (2*significance_kalman)**0.5
                                logger.info("Calculated snrk of {0} after detection. "
                                            "Adding it to CandData.".format(snrk))
                                kwargs[feature] = snrk

                cd = pipeline_canddata(st, candloc, data_corr, spec_std=spec_std,
                                       sig_ts=sig_ts, kalman_coeffs=kalman_coeffs, **kwargs)

                if st.prefs.saveplots:
                    candidates.candplot(cd, snrs=snrs)  # snrs before clustering

                # regenerate cc with extra features in cd
                cc1 += candidates.cd_to_cc(cd)

        # if candidates that do not need new featuers, just select peaks
        else:
            logger.info("Using clustering info to select {0} candidates"
                        .format(len(calcinds)))
            cc1.array = cc.array.take(calcinds)

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


def pipeline_canddata(st, candloc, data_dmdt=None, spec_std=None, cpuonly=False,
                      sig_ts=[], kalman_coeffs=[], **kwargs):
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

    if ('snrk' in st.features and
        'snrk' not in kwargs and
        (spec_std is None or not len(sig_ts) or not len(kalman_coeffs))):
        spec_std, sig_ts, kalman_coeffs = util.kalman_prep(data_dmdt)

#    fftmode = 'fftw' if cpuonly else st.fftmode  # can't remember why i did this!
    image = rfpipe.search.grid_image(data_dmdt, uvw, st.npixx, st.npixy, st.uvres,
                                     'fftw', st.prefs.nthread, wisdom=wisdom,
                                     integrations=[candint])[0]

    # TODO: allow dl,dm as args and reproduce detection for other SNRs
    dl, dm = st.pixtolm(np.where(image == image.max()))
    util.phase_shift(data_dmdt, uvw, dl, dm)
    dataph = data_dmdt[max(0, candint-st.prefs.timewindow//2):candint+st.prefs.timewindow//2].mean(axis=1)
    util.phase_shift(data_dmdt, uvw, -dl, -dm)

    # TODO: This probably needs to be masked to avoid averaging zeros in
    spec = data_dmdt.real.mean(axis=3).mean(axis=1)[candloc[1]]

    if 'snrk' in st.features and 'snrk' not in kwargs:
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


def refine_sdm(sdmname, dm, preffile='realfast.yml', gainpath='/home/mchammer/evladata/telcal/',
               npix_max_orig=None, search_sigma=7, ddm=100,
               refine=True, classify=True, devicenum=None, workdir=None):
    """  Given candId, look for SDM in portal, then run refinement.
    Assumes this is running on rfnode with CBE lustre.
    npix_max_orig sets the npix_max or the original detection.
    ddm sets +- of dm grid to search
    """

    from rfpipe import metadata, state, pipeline, candidates, util

    if devicenum is None:
        from distributed import get_worker
        name = get_worker().name
        assert 'fetch' in name
        devicenum = name.split('fetch')[1]

    # Searching for gainfile
    datasetId = '{0}'.format('_'.join(os.path.basename(sdmname).split('_')[1:-1]))
    # set the paths to the gainfile
    gainname = datasetId + '.GN'
    logging.info('Searching for the gainfile {0} in {1}'.format(gainname, gainpath))
    for path, dirs, files in os.walk(gainpath):
        for f in filter(lambda x: gainname in x, files):
            gainfile = os.path.join(path, gainname)
            break

    # Searching all miniSDMs
    prefs = {'saveplots': False, 'savenoise': False, 'savesols': False, 'savecandcollection': False,
             'savecanddata': True, 'gainfile': gainfile, 'sigma_image1': search_sigma, 'workdir': workdir,
             'dm_maxloss': 0.01, 'maxdm': dm+ddm, 'npix_max': None}

    band = metadata.sdmband(sdmfile=sdmname, sdmscan=1)

    try:
        st = state.State(sdmfile=sdmname, sdmscan=1, inprefs=prefs, preffile=preffile, name='NRAOdefault'+band, showsummary=False)
    except AssertionError:  # could be no bdf
        try:
            logger.warn("Could not generate state for full image search with built-in BDFs. Trying with lustre BDFs.")
            st = state.State(sdmfile=sdmname, sdmscan=1, inprefs=prefs, preffile=preffile, name='NRAOdefault'+band, bdfdir='/lustre/evla/wcbe/data/realfast', showsummary=False)
        except AssertionError:  # could be state can't be defined
            try:
                logger.warn("Could not generate state with lustre BDFs. Trying with npix_max at 2x original image size...")
                prefs['npix_max'] = 2*npix_max_orig
                st = state.State(sdmfile=sdmname, sdmscan=1, inprefs=prefs, preffile=preffile, name='NRAOdefault'+band, bdfdir='/lustre/evla/wcbe/data/realfast', showsummary=False)
            except AssertionError:  # could be state can't be defined
                logger.warn("Could not generate state with lustre BDFs. Trying with original image size...")
                prefs['npix_max'] = npix_max_orig
                st = state.State(sdmfile=sdmname, sdmscan=1, inprefs=prefs, preffile=preffile, name='NRAOdefault'+band, bdfdir='/lustre/evla/wcbe/data/realfast', showsummary=False)

    st.prefs.dmarr = [dm for dm in st.dmarr if (dm == 0 or dm > dm-ddm)]  # remove superfluous dms
    st.clearcache()
    st.summarize()
    ccs = pipeline.pipeline_scan(st, devicenum=devicenum)
    cc = sum(ccs) if len(ccs) else ccs

    # Classify the generated pickles using FETCH and generate refinement plots
    if len(cc):
        maxind = np.where(cc.snrtot == cc.snrtot.max())[0]
        assert len(maxind) == 1
        cd = cc[maxind[0]].canddata[0]
        assert isinstance(cd, candidates.CandData)

        if classify:
            frbprob = candidates.cd_to_fetch(cd, classify=True, mode='CPU')
            logging.info('FETCH FRB Probability of the candidate {0} is {1}'.format(cd.candid, frbprob))
        else:
            frbprob = None

        if refine:
            logging.info('Generating Refinement plots')
            cd_refined_plot(cd, devicenum, frbprob=frbprob)
    else:
        if prefs['npix_max'] != npix_max_orig:
            logging.info('No candidate was found in first search. Trying again with original image size.'.format(cc))
            prefs['npix_max'] = npix_max_orig
            st = state.State(sdmfile=sdmname, sdmscan=1, inprefs=prefs, preffile=preffile, name='NRAOdefault'+band, bdfdir='/lustre/evla/wcbe/data/realfast',
                             showsummary=False)
            st.prefs.dmarr = [dm for dm in st.dmarr if (dm == 0 or dm > dm-ddm)]  # remove superfluous dms
            st.clearcache()
            st.summarize()
            ccs = pipeline.pipeline_scan(st, devicenum=devicenum)
            cc = sum(ccs) if len(ccs) else ccs
            if len(cc):
                maxind = np.where(cc.snrtot == cc.snrtot.max())[0]
                assert len(maxind) == 1
                cd = cc[maxind[0]].canddata[0]
                assert isinstance(cd, candidates.CandData)

                if classify:
                    frbprob = candidates.cd_to_fetch(cd, classify=True, mode='CPU')
                    logging.info('FETCH FRB Probability of the candidate {0} is {1}'.format(cd.candid, frbprob))
                else:
                    frbprob = None

                if refine:
                    logging.info('Generating Refinement plots')
                    cd_refined_plot(cd, devicenum, frbprob=frbprob)
            else:
                logging.info('No candidate was found in search at original image size. Giving up.')


def cd_refined_plot(cd, devicenum, nsubbands=4, mode='CPU', frbprob=None):
    """ Use canddata object to create refinement plot of subbanded SNR and dm-time plot.
    """
    
    import rfpipe.search
    from rfpipe import util
    from matplotlib import gridspec
    import pylab as plt
    import matplotlib

    params = {
        'axes.labelsize' : 14,
        'font.size' : 9,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': False,
        'figure.figsize': [12, 10]
    }
    matplotlib.rcParams.update(params)
    
    segment, candint, dmind, dtind, beamnum = cd.loc
    st = cd.state
    scanid = cd.state.metadata.scanId
    width_m = st.dtarr[dtind]
    timewindow = st.prefs.timewindow
    tsamp = st.inttime*width_m
    dm = st.dmarr[dmind]
    ft_dedisp = np.flip((cd.data.real.sum(axis=2).T), axis=0)
    chan_freqs = np.flip(st.freq*1000, axis=0)  # from high to low, MHz
    nf, nt = np.shape(ft_dedisp)
    
    candloc = cd.loc
    
    logger.debug('Size of the FT array is ({0}, {1})'.format(nf, nt))

    try:
        assert nt > 0
    except AssertionError as err:
        logger.exception("Number of time bins is equal to 0")
        raise err

    try:
        assert nf > 0
    except AssertionError as err:
        logger.exception("Number of frequency bins is equal to 0")
        raise err    

    roll_to_center = nt//2 - cd.integration_rel
    ft_dedisp = np.roll(ft_dedisp, shift=roll_to_center, axis=1)

    # If timewindow is not set during search, set it equal to the number of time bins of candidate
    if nt != timewindow:
        logger.info('Setting timewindow equal to nt = {0}'.format(nt))
        timewindow = nt
    else:
        logger.info('Timewindow length is {0}'.format(timewindow))

    try:
        assert nf == len(chan_freqs)
    except AssertionError as err:
        logger.exception("Number of frequency channel in data should match the frequency list")
        raise err

    if dm is not 0:
        dm_start = 0
        dm_end = 2*dm
    else:
        dm_start = -10
        dm_end = 10

    logger.info('Generating DM-time for candid {0} in DM range {1:.2f}--{2:.2f} pc/cm3'
                .format(cd.candid, dm_start, dm_end))

    logger.info("Using gpu devicenum: {0}".format(devicenum))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devicenum)

    dmt = rfpipe.search.make_dmt(ft_dedisp, dm_start-dm, dm_end-dm, 256, chan_freqs/1000,
                          tsamp, mode=mode, devicenum=int(devicenum))
    
    delay = util.calc_delay(chan_freqs/1000, chan_freqs.max()/1000, -1*dm, tsamp)
    dispersed = rfpipe.search.dedisperse_roll(ft_dedisp, delay)
#    dispersed = disperse(ft_dedisp, -1*dm, chan_freqs/1000, tsamp)
    
    subsnrs, subts, bands = calc_subband_info(ft_dedisp, chan_freqs, nsubbands)    
    logging.info(f'Generating time series of full band')
    ts_full = ft_dedisp.sum(0)
    logging.info(f'Calculating SNR of full band')
    snr_full = calc_snr(ts_full)

    to_print = []
    logging.info(f'{scanid}')
    to_print.append(f'{scanid}\n')
    logging.info(f'candloc: {candloc}, dm: {dm:.2f}')
    to_print.append(f'candloc: {candloc}, dm: {dm:.2f}\n')
    logging.info(f'SNR of full band is: {snr_full:.2f}')
    to_print.append(f'SNR of full band is: {snr_full:.2f}\n')
    logging.info(f'Subbanded SNRs are:')    
    to_print.append(f'Subbanded SNRs are:\n')
    for i in range(nsubbands):
        logging.info(f'Band: {chan_freqs[bands[i][0]]:.2f}-{chan_freqs[bands[i][1]-1]:.2f}, SNR: {subsnrs[i]:.2f}')
        to_print.append(f'Band: {chan_freqs[bands[i][0]]:.2f}-{chan_freqs[bands[i][1]-1]:.2f}, SNR: {subsnrs[i]:.2f}\n')
    if frbprob is not None:
        logging.info(f'frbprob: {frbprob}')
        to_print.append(f'frbprob: {frbprob}\n')

    str_print = ''.join(to_print)

    ts = np.arange(timewindow)*tsamp
        
    gs = gridspec.GridSpec(4, 3, width_ratios=[4, 0.1, 2], height_ratios=[1, 1, 1, 1], wspace=0.02, hspace=0.15)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[2, 0])
    ax4 = plt.subplot(gs[3, 0])
    ax11 = plt.subplot(gs[0, 1])
    ax22 = plt.subplot(gs[1, 1])
    ax33 = plt.subplot(gs[2, 1])
    ax44 = plt.subplot(gs[3, 1])
    ax5 = plt.subplot(gs[:, 2])

    x_loc = 0.1
    y_loc = 0.5

    for i in range(nsubbands):
        ax1.plot(ts, subts[i] - subts[i].mean(), label = f'Band: {chan_freqs[bands[i][0]]:.0f}-{chan_freqs[bands[i][1]-1]:.0f}')
    ax1.plot(ts, subts.sum(0) - subts.sum(0).mean(), 'k.', label = 'Full Band')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3, fancybox=True, shadow=True)
    ax1.set_ylabel('Flux (Arb. units)')
    ax1.set_xlim(np.min(ts), np.max(ts))
    ax11.text(x_loc, y_loc, 'Time Series', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax11.axis('off')

    ax2.imshow(ft_dedisp, aspect='auto', extent=[ts[0], ts[-1], np.min(chan_freqs), np.max(chan_freqs)])
    ax2.set_ylabel('Freq')
    ax22.text(x_loc, y_loc, 'Dedispersed FT', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax22.axis('off')

    ax3.imshow(dispersed, aspect='auto', extent=[ts[0], ts[-1], np.min(chan_freqs), np.max(chan_freqs)])
    ax3.set_ylabel('Freq')
    ax33.text(x_loc, y_loc, 'Original dispersed FT', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax33.axis('off')

    ax4.imshow(dmt, aspect='auto', extent=[ts[0], ts[-1], dm+1*dm, dm-dm])
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('DM')
    ax44.text(x_loc, y_loc, 'DM-Time', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax44.axis('off')

    ax5.text(0.02, 0.8, str_print, fontsize=14, ha='left', va='top', wrap=True)
    ax5.axis('off')
    segment, candint, dmind, dtind, beamnum = candloc
    #plt.tight_layout()
    plt.savefig(os.path.join(cd.state.prefs.workdir, 'cands_{0}_refined.png'.format(cd.state.metadata.scanId)), bbox_inches='tight')


def calc_subband_info(ft, chan_freqs, nsubbands=4):
    """ Use freq-time array to calculate subbands and detect in each subband.
    """
    
    nf, nt = ft.shape

    subbandsize = nf//nsubbands
    bandstarts = np.arange(1,nf,subbandsize) - 1
    subsnrs = np.zeros(nsubbands)
    subts = np.zeros((nsubbands, ft.shape[1]))
    bands = []
    for i in range(nsubbands):
        bandstart = i*subbandsize
        if i == nsubbands-1:
            bandend = nf-1
        else:
            bandend = (i+1)*subbandsize

        bands.append([bandstart, bandend])
        logging.info(f'Generating time series of band: {chan_freqs[bands[i][0]]:.0f}-{chan_freqs[bands[i][1]-1]:.0f}')
        subts[i, :] = ft[bandstart: bandend,:].sum(0)
        logging.info(f'Calculating SNR of band: {chan_freqs[bands[i][0]]:.0f}-{chan_freqs[bands[i][1]-1]:.0f}')
        subsnrs[i] = calc_snr(subts[i, :])
    return subsnrs, subts, bands
    

def calc_snr(ts):
    """ Use time series to calculate SNR of peak.
    """
    from rfpipe import util
    
    std =  util.madtostd(ts)
    if std == 0:
        logging.warning('Standard Deviation of time series is 0. SNR not defined.')
        snr = np.nan
        return snr

    noise_mask = (np.median(ts) - 3*std < ts) & (ts < np.median(ts) + 3*std)
    if noise_mask.sum() == len(ts):
        logging.warning('Time series is just noise, SNR = 0.')
        snr = 0
    else:
        mean_ts = np.mean(ts[noise_mask])
        std = util.madtostd(ts[noise_mask]-mean_ts)
        if std == 0:
            logging.warning('Noise Standard Deviation is 0. SNR not defined.')
        snr = np.max(ts[~noise_mask]-mean_ts)/std
    return snr


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
