"""
The DESDM single-CCD image masking module.

"""
import os

__author__ = "Felipe Menanteau, Alex Drlica-Wagner, Eli Rykoff"
__version__ = '3.0.0'
__revision__= '0'
version = __version__

from . import immasklib
from .immasklib import cmdline
from .immasklib import elapsed_time
