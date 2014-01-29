__author__  = "Felipe Menanteau"
__version__ = '0.1'
version = __version__
from . import immasklib
from .immasklib import DESIMA
from .immasklib import elapsed_time
from .immasklib import cmdline

# CR masking routine only
#from cr_masking import *
# Streak masking routines
#from . import streak_masking
#from .streak_masking import STREAK
#from .streak_masking import cmdline

print "#\tModule %s by %s, version %s is loaded" % (__name__,__author__,__version__)

