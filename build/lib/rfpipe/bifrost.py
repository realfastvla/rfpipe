#from __future__ import print_function, division, absolute_import, unicode_literals
#from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems
from io import open

import logging
logger = logging.getLogger(__name__)

import sys, time, glob
from datetime import datetime
from copy import deepcopy
import numpy as np
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.dtype import name_nbit2numpy
import bifrost.blocks as blocks
import bifrost.views as views
import sdmpy

from rfpipe.util import meantsub, dedisperse, resample_image
from rfpipe.source import data_prep

#resample_image(data, dt, uvw, freqs, npixx, npixy, uvres, threshold, wisdom)

class SdmFileRead(sdmpy.sdm.SDM):
    """ File-like object for science data model (sdm) files
    Based on realfastvla/sdmpy, which has context manager in SDM class.

    Args:
        filename (str): Name of sdm file to open
        scan_id (int): 1-based index for scan in sdm file

    """

    def __init__(self, filename, scan_id):
        super(SdmFileRead, self).__init__(filename)
        self.scan_id = scan_id
        self.sdm = sdmpy.SDM(filename)
        self.scan = self.sdm.scan(self.scan_id)
        
        self.n_baselines = int(len(self.scan.baselines))
        self.n_chans     = int(self.scan.numchans[0])  # **hack: need to use chan per spw properly
        self.n_pol       = int(self.scan.bdf.spws[0].npol('cross'))  # **hack: need to use chan per spw properly
        self.n_spw       = int(len(self.scan.spws))  # **hack: need to use chan per spw properly
        self.shape       = (self.n_baselines, self.n_chans*self.n_spw, self.n_pol)
        self.integration_id = 0

    def read(self):
        try:
            data = self.scan.bdf.get_integration(self.integration_id).get_data().reshape(self.shape)
            self.integration_id += 1
        except IndexError:
            return np.array([])

        return data


class SdmReadBlock(bfp.SourceBlock):
    """ Block for reading binary data from sdm file and streaming it into a bifrost pipeline

    Args:
        filenames (list): A string of filename to open
        scan_id (int): 1-based index of scan to read
        gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
    """

    def __init__(self, filename, scan_id, gulp_nframe=1, *args, **kwargs):
        super(SdmReadBlock, self).__init__([filename], gulp_nframe, *args, **kwargs)  # **hack: filename is string, but list expected here
        self.filename = filename
        self.scan_id = scan_id
        self.gulp_nframe = gulp_nframe

    def create_reader(self, filename):
        print("Loading sdm file {0}, scan {1}".format(filename, self.scan_id))
        return SdmFileRead(filename, self.scan_id)

    def on_sequence(self, ireader, filename):

        shape = ireader.shape
        n_baselines, n_chans, n_pol = shape

        ohdr = {'name': self.filename,
                '_tensor': {
                        'dtype':  'cf32',
                        'shape':  [-1, n_baselines, n_chans, n_pol],
                        'labels': ['time', 'baseline', 'chan', 'pol']
                        },
                'gulp_nframe': self.gulp_nframe,
                'itershape': shape
                }

        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()

        print("Integration {0}: shape {1}".format(reader.integration_id, indata.shape))

        if indata.shape[0] != 0:
            ospans[0].data[0] = indata
            return [1]
        else:
            return [0]


class GainCalBlock(bfp.TransformBlock):
    def __init__(self, iring, gainfile, *args, **kwargs):
        super(GainCalBlock, self).__init__(iring, *args, **kwargs)
        self.gainfile = gainfile

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)

        shape = ohdr['itershape']
        gr = np.random.normal(size=shape)
        gi = np.random.normal(size=shape)
        self.gain = np.zeros(dtype='complex64', shape=shape)
        self.gain.real = gr
        self.gain.imag = gi
        print('Parsed *fake* gainfile {0}'.format(self.gainfile))

        return ohdr

    def on_data(self, ispan, ospan):
        idata = ispan.data
        ospan.data[...] = idata * self.gain

        return ispan.nframe


class PrintMeanBlock(bfp.SinkBlock):
    def __init__(self, iring, *args, **kwargs):
        super(PrintMeanBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0

    def on_sequence(self, iseq):
        print("Starting sequence processing at {0}".format(datetime.now()))
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        print("Time {0}: {1}".format(now, np.mean(ispan.data)))
        self.n_iter += 1


if __name__ == "__main__":

    # Setup pipeline
    assert len(sys.argv) == 3, 'Need sdm filename and scan_id'
    filename   = str(sys.argv[1])
    scan_id   = str(sys.argv[2])

    b_read      = SdmReadBlock(filename, scan_id)
    b_cal       = GainCalBlock(b_read, 'fakecal.GN')
    b_scrunched = views.split_axis(
                    b_cal, axis='time',
                    n=20, label='window')
    b_pr = PrintMeanBlock(b_scrunched)

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print(pipeline.dot_graph())
    pipeline.run()
	
