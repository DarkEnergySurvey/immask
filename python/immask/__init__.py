__author__  = "Felipe Menanteau, Alex Drlica-Wagner, Eli Rykoff"
__version__ = 'trunk'
version = __version__

"""
The DESDM single-CCD image masking module.

$Id$
$Rev::                                  $:  # Revision of last commit.
$LastChangedBy::                        $:  # Author of last commit.
$LastChangedDate::                      $:  # Date of last commit.
"""

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

