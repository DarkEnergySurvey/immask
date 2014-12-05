"""
The DESDM single-CCD image masking module.

$Id: __init__.py 22391 2014-05-29 22:17:54Z kadrlica $
$Rev:: 30138                            $:  # Revision of last commit.
$LastChangedBy:: kadrlica               $:  # Author of last commit.
$LastChangedDate:: 2014-05-29 17:17:54 #$:  # Date of last commit.
"""
svnrev = "$Rev:$"
svnurl = "$HeadURL$"

__author__  = "Felipe Menanteau, Alex Drlica-Wagner, Eli Rykoff"
__version__ = svnurl.strip('$').split()[-1].split(os.sep)[-4]
__revision__= svnrev.strip('$').split()[-1]
version = __version__

from . import immasklib
from .immasklib import cmdline
from .immasklib import elapsed_time
