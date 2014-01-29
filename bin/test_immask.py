#!/usr/bin/env python

import time
import os,sys
import immask

# Get the start time
t0 = time.time()
   
args   = immask.cmdline()
desobj = immask.DESIMA(args.fileName,args.outdir)

# CR Rejection
#desobj.CRs(FWHM     = args.fwhm,
#           dilateCR = args.dilateCR,
#           interpCR = args.interpCR,
#           minSigma = args.minSigma,
#           min_DN   = args.min_DN,
#           nGrowCR  = args.nGrowCR)

streak_args = {
    'bkgfile'    :'DECam_00145879_21_bkg.fits.fz',
    'bin_factor' : 8,
    }
streakobj = immask.STREAK(desobj)
streakobj.mask_streaks(**streak_args)#,writeobjs=True,writemask=False,draw=False) 
#streakobj.draw()
#desobj.OUT_MSK = msk
#desobj.OUT_WGT = wgt
# Write it out
desobj.write(compress=args.compress)
print >>sys.stderr,"# Time:%s" % immask.elapsed_time(t0)


sys.exit()




