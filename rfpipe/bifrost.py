from __future__ import print_function, division, absolute_import

from functools import partial

from bifrost.block import *
from rfpipe.search import meantsub, dedisperse, resample_image
from rfpipe.source import dataprep
from rfpipe.calibration import apply_telcal

def array_print(data):
    print(data.shape)
    
#resample_image(data, dt, uvw, freqs, npixx, npixy, uvres, threshold, wisdom)

def run(st):

    def generate_data():
        for i in range(st.nsegments):
            yield dataprep(st, i)   # IS THIS THE ITERATION I WANT? OR MULTIPLE (SMALLER) KINDS OK?
    segment = 0  # NEED TO PASS SEGMENT IN AS HEADER INFO?
    
    blocks = [
        (NumpySourceBlock(generate_data), {'out_1': 'raw_ring'}),
        (NumpyBlock(partial(apply_telcal, st, segment)), {'in_1' : 'raw_ring', 'out_1' : 'cal_ring'}),
        (NumpyBlock(partial(apply_telcal, st, segment)), {'in_1' : 'cal_ring', 'out_1' : 'cal2_ring'}),
#        (NumpyBlock(meantsub), {'in_1' : 'cal_ring', 'out_1' : 'sub_ring'}),   # IN/OUT not working with numba?
        (NumpyBlock(array_print, outputs=0), {'in_1': 'cal2_ring'})]

    Pipeline(blocks).main()
