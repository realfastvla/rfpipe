from __future__ import absolute_import

__all__ = ['state', 'preferences', 'metadata', 'source', 'search', 'pipeline',
           'reproduce', 'candidates']  # 'flagging', 'calibration', 'util'

from rfpipe import *
from rfpipe.version import __version__
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')