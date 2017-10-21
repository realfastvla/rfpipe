from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import str, bytes, dict, object, range, map, input
from future.utils import itervalues, viewitems, iteritems, listvalues, listitems

import sdmpy
import os
import progressbar

bar = progressbar.ProgressBar()


def writesdm(sdmname, newsdmname, i0, i1, newstartTime):
    """ Going from sdm to new sdm for given integration range (i0, i1).
    Just playing for now...
    """

    # get stuff
    sdm = sdmpy.SDM(sdmname)
    bdf = sdm.scan(scannum).bdf

    # modify stuff
    bdf.sdmDataHeader.startTime = newstartTime # mjd nanoseconds?

    # write stuff
    sdm.write(newsdmname)
    os.mkdir(os.path.join(newsdmname, 'ASDMBinary'))

    newbdf = sdmpy.bdf.BDFWriter(os.path.join(sdmname, 'ASDMBinary', 'newbdf'), bdf=bdf)
    newbdf.write_header()
    for i in bar(range(i0, i1)):
        integration = bdf.get_integration(i)
        newbdf.write_integration(integration)
    newbdf.close()
