#!/usr/bin/env python

import time
import os
import sys
import immask

# Get the start time
t0 = time.time()
   
parser,args   = immask.cmdline()
desobj = immask.DESIMA(args.fileName,args.outdir)

# CR Rejection
desobj.CRs(vars(args))
desobj.mask_streaks(vars(args))
desobj.write(compress=args.compress)
print >>sys.stderr,"# Time:%s" % immask.elapsed_time(t0)


