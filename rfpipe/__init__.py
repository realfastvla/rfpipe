from __future__ import absolute_import

__all__ = ['state', 'preferences', 'metadata', 'source', 'pipeline', 'reproduce']
          # 'candidates', 'flagging', 'calibration', 'util', 'search'  # these are slow to import

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rfpipe import *
from rfpipe.version import __version__
