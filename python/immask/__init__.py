"""
The DESDM single-CCD image masking module.

$Id: __init__.py 22391 2014-05-29 22:17:54Z kadrlica $
$Rev::                                  $:  # Revision of last commit.
$LastChangedBy:: kadrlica               $:  # Author of last commit.
$LastChangedDate:: 2014-05-29 17:17:54 #$:  # Date of last commit.
"""

__author__  = "Felipe Menanteau, Alex Drlica-Wagner, Eli Rykoff"
__version__ = 'trunk'
__revision__= "$Rev$".strip('$').split()[-1]
version = __version__

from . import immasklib
from .immasklib import cmdline
from .immasklib import elapsed_time

#from .immasklib import DESIMA

# Other ways... not really used
# CR masking routine only
#from cr_masking import *

# To load the standalone streak masking routines at startup --- not required
#from . import streak_masking
#from .streak_masking import STREAK
#from .streak_masking import cmdline
#print "# Module %s v%s is loaded" % (__name__,__version__)

