#!/usr/bin/env python

import time
import os
import sys
import immask

# Get the start time
t0 = time.time()
   
args   = immask.cmdline()
desobj = immask.DESIMA(args.fileName,args.outName)

# CR Rejection
desobj.CRs(**args.__dict__)
desobj.mask_streaks(**args.__dict__)
desobj.write(compress=args.compress)
print >>sys.stderr,"# Time:%s" % immask.elapsed_time(t0)


