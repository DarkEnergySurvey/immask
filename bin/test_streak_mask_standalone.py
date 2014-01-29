#!/usr/bin/env python

import time
import os,sys
from immask import streak_masking

# Get the start time
t0 = time.time()

args      = streak_masking.cmdline()
streakobj = streak_masking.STREAK(**args.__dict__) # Pass it as a dictionary, as required by fuctions/classes
streakobj.mask_streaks(**args.__dict__)
# Write it out
streakobj.write(compress=args.compress)
print >>sys.stderr,"# Time:%s" % streak_masking.elapsed_time(t0)

