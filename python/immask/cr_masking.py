#!/usr/bin/env python

import math
import os,sys
import shutil
import numpy as np
import fitsio 
import time
import copy

# LSST modules
import lsst.daf.base        as dafBase
import lsst.pex.config      as pexConfig
import lsst.afw.image       as afwImage
import lsst.afw.math        as afwMath
import lsst.meas.algorithms as measAlg
# If we want to interpolate and/or dilate masks we need this
import lsst.ip.isr          as ip_isr

"""
   Set of Cosmic Ray rejection routines using the LSST framework
   Felipe Menanteau, NCSA (felipe@illinois.edu)
"""


class DESMaskCRs:

   """
   A Class to handle DECam fits/fits.fz files and mask cosmic rays
   using fitsio top open/read/close files and the lsst framework to
   find and mask the cosmic rays.

   Felipe Menanteau, NCSA, University of Illinois.
   """

   def __init__ (self,fileName,outdir):

      self.fileName  = extract_filename(fileName)
      self.outdir    = outdir
      
      # Make sure the output directory exits
      if not os.path.exists(outdir):
         print "# Will create output directory  %s" % outdir
         os.mkdir(outdir)
         
      # Gets SCI, MSK, WGT and created VAR
      self.read_HDUs()

      # Make copies (shallow) of SCI, MSK, WGT and created VAR (OUT_*)
      self.copy_ndarrays() 


   def CRs(self,**keys):
      
      self.FWHM      = keys.get('FWHM',None)
      self.minSigma  = keys.get('minSigma')
      self.min_DN    = keys.get('min_DN')
      self.interpCR  = keys.get('interpCR')
      self.dilateCR  = keys.get('dilateCR')
      self.nGrowCR   = keys.get('nGrowCR')

      self.make_BAD_mask() # True by default, unless we decide not to.
      self.make_lsst_image()
      self.find_CRs()
      self.fix_pixels_CR()
      self.update_hdr_CR()


   def set_DECamMaskPlaneDict(self):
      
      """
      Dictionary to translate from DECam mask plane to LSST definition
      To see all mask planes:
            print msk.printMaskPlanes()
      """
      self.DECamMaskPlaneDict = dict(
         BAD       = 0,  # set in bpm (hot/dead pixel/column) (was BPM)
         SAT       = 1,  # saturated pixel (was SATURATE)
         INTRP     = 2,  # interpolated pixel (was INTERP)
         LOW       = 3,  # too little signal- i.e. poor read
         CRAY      = 4,  # cosmic ray pixel N.b. not CR --- we'll set that ourselves later
                         # CR is the name used in the LSST framework, and will supersede CRAY!
         DETECTED  = 5,  # bright star pixel (was STAR)
         TRAIL     = 6,  # bleed trail pixel
         EDGEBLEED = 7,  # edge bleed pixel
         SSXTALK   = 8,  # pixel potentially effected by xtalk from super-saturated source
         EDGE      = 9,  # pixel flagged to exclude CCD glowing edges
         STREAK    = 10, # streak 
         FIX       = 11, # a bad pixel that was fixed
         )

   def read_HDUs(self):

      """
      Read in the HDU as ndarrays and headers as dictionaries with fitsio for a DESDM/DECam image
      """

      # Get the fitsio element -- we'll modify this in place
      print "# Reading in extensions and headers for %s" % self.fileName
      self.ifits = fitsio.FITS(self.fileName,'r')
      sci_hdu, msk_hdu, wgt_hdu = get_hdu_numbers(self.ifits)
      # Read in the Science, Mask and Weight Images array with fitsio, as we'll
      # need to write them out using fitsio once we are done with them
      self.SCI = self.ifits[sci_hdu].read()
      self.MSK = self.ifits[msk_hdu].read()
      self.WGT = self.ifits[wgt_hdu].read()
      # Now let's read the headers
      self.h_sci = self.ifits[sci_hdu].read_header()
      self.h_msk = self.ifits[msk_hdu].read_header()
      self.h_wgt = self.ifits[wgt_hdu].read_header()     

      # Get the image size to set the allowed fraction of image to be already masked
      (self.ny,self.nx) = self.SCI.shape

      # Pass them up
      self.sci_hdu = sci_hdu
      self.msk_hdu = msk_hdu
      self.wgt_hdu = wgt_hdu

      # A handy handle
      self.headers = (self.h_sci,self.h_msk,self.h_wgt)
      print "# Done reading HDU "

   def make_BAD_mask(self):

      """
      Create a specific boolean masks to temporaryly deal with BPM
      interpolation Mask for the BAD pixels mask, this won't be
      requited if all of the BPM are masked by imdetrend in the
      future. It assumes the following bit from the maskbit plane:

          BADbit    = 1
          INTERPbit = 4
          EDGEbit   = 512
      """
      
      print "# Creating BAD/BPM pixel mask for interpolation"
      # Mask for BAD=1 pixels that have been already interpolated
      # (INTERP=4), and keep the original values for later.  It is
      # important to note that we modify only the MSK ndarray, which
      # is the one used by the LSST framework. The original,
      # unmodified values are keept in the copy for output OUT_MSK.
      masked_bad        = (self.MSK & 1) > 0
      masked_bad_interp = ((self.MSK & 4) > 0) & masked_bad
      
      # Create a boolean mask of the glowing edges, we don't need to
      # keep the original values for later, as they are kept in the OUT_MSK copy.
      masked_edge = (self.MSK & 512) > 0
      
      # Temporaly change values of 513 to 512 in order to avoid
      # interpolating over glowing edges of ccds also labeled as BAD (512+1=513)
      self.MSK[masked_edge] = 512

      # Temporaryly change the values of BAD and INTERP pixels from 5
      # to 4 for the MSK ndarray as we do not want to interpolate twice.
      self.MSK[masked_bad_interp] = 4

   def copy_ndarrays(self):

      """
      Make shallow copies (shallow is enough) of the SCI, MSK and WGT
      ndarrays using python copy function to preserve the original
      information of the fits files. We need to do this before they
      are modified in place by the the LSST framework functions.
      """
      print "# Making shallow copies of SCI, MSK and WGT ndarrays"
      self.OUT_SCI = copy.copy(self.SCI)
      self.OUT_MSK = copy.copy(self.MSK)
      self.OUT_WGT = copy.copy(self.WGT)

   def make_lsst_image(self):

      """
      Create and LSST-like image from the SCI, MSK and WGT
      ndarrays. We pass them into the LSST framework structure to call
      the CR finder.
      """

      print "# Making LSST image structure from SCI, MSK and VAR"
      # 0 Set up the Mask Plane dictionay for DECam
      self.set_DECamMaskPlaneDict()
      # 1 - Science
      self.sci = afwImage.ImageF(self.SCI)
      # 2- The Mask plane Image and rewrite mask planes into LSST convention
      self.msk = afwImage.MaskU(self.MSK)
      self.msk.conformMaskPlanes(self.DECamMaskPlaneDict) 

      # 3 - The variance Image
      self.WGT_fixed = np.where(self.WGT<=0, self.WGT.max()/1e6, self.WGT) # Fix values < 0
      self.VAR = 1/self.WGT_fixed
      self.var = afwImage.ImageF(self.VAR)
      # Into numpy-arrays to handle some numpy fast operations
      self.scia = self.sci.getArray()
      self.mska = self.msk.getArray()
      self.vara = self.var.getArray()

      # **************************************************************
      # This part is required to rerun cosmic ray masking on existing images,
      # i.e. ccd images already with cosmic ray on the weight/mask images.
      # We need to set the 0s in the inverse variance for the existing
      # cosmic rays (CRAY) to the median. Notice that CRAY is the "fake" name for
      # the existing cosmic rays mask plane. The plane with the newly
      # detected one is called "CR" in the LSST frame work and that is
      # the one we should use when handlying the cosmic rays detected
      # with the LSST function.
      # *** Only need for images with existing CRAY plane ****
      CRAY     = self.msk.getPlaneBitMask("CRAY") # Gets the value for CR (i.e. 256)
      cr_prev  = np.where(np.bitwise_and(self.mska, CRAY))
      NCR_prev = len(cr_prev[0])
      if NCR_prev > 0:
         print "# CRs previously found in image -- will be fixed before CR"
         median_rep            = np.median(self.vara)
         print "# Fixing weight map with median=%.1f for those %d CR-pixels" % (NCR_prev, median_rep)
         self.vara[cr_prev]    = median_rep # The LSST handle
         self.OUT_WGT[cr_prev] = median_rep # The original array
         print "# Fixing mask plane for those CR-pixels too " 
         self.mska[cr_prev]    = self.mska[cr_prev]    - 16 # The LSST handle
         self.OUT_MSK[cr_prev] = self.OUT_MSK[cr_prev] - 16 # The original array
      # ************************************************************
      # Make an LSST masked image (science, mask, and weight) 
      self.mi = afwImage.makeMaskedImage(self.sci, self.msk, self.var)

   def find_CRs(self,**keys):

      """
      Find the Cosmic Rays on a Science Image using lsst.meas.algorithms.findCosmicRays
      """
      
      # Estimate the PSF of the science image.
      if not self.FWHM:
         print "# Will attempt to get FWHM from the image header"
         self.FWHM  = get_FWHM(self.ifits,self.sci_hdu)
         xsize = int(self.FWHM*9)
         psf   = measAlg.DoubleGaussianPsf(xsize, xsize, self.FWHM/(2*math.sqrt(2*math.log(2))))

      # Interpolate bad pixels before finding CR to avoid false detections
      fwhm = 2*math.sqrt(2*math.log(psf.computeShape().getDeterminantRadius()))
      print "# Interpolating BPM/BAD pix mask" 
      ip_isr.isr.interpolateFromMask(self.mi, fwhm, growFootprints=0, maskName = 'BAD')

      # simple background estimation to use on findCosmicRays
      background = afwMath.makeStatistics(self.mi.getImage(), afwMath.MEANCLIP).getValue()
      print "# Setting up background at: %.1f [counts]" %  background

      # Tweak some of the configuration and call FindCosmicRays
      # More info on lsst/meas/algorithms/findCosmicRaysConfig.py
      crConfig             = measAlg.FindCosmicRaysConfig()
      crConfig.minSigma    = self.minSigma
      crConfig.min_DN      = self.min_DN
      crConfig.nCrPixelMax = int(self.nx*self.ny/3) # 1e6 will not work, needs an integer
      if self.interpCR:
         crConfig.keepCRs  = False # Do interpolate
         print "# Will erase detected CRs -- interpolate CRs on SCI image"
      else:
         crConfig.keepCRs  = True # Do not interpolate -- THIS DOESN'T WORK!!!!!
         print "# Will keep detected CRs -- no interpolation on SCI image"

      # Now, we do the magic
      self.crs = measAlg.findCosmicRays(self.mi, psf, background, pexConfig.makePolicy(crConfig))

      # Dilate interpolation working on the mi element
      if self.dilateCR and self.interpCR:
         print "# Dilating CR pix mask by %s pixel(s):" % self.nGrowCR
         ip_isr.isr.interpolateFromMask(self.mi, fwhm, growFootprints=self.nGrowCR, maskName = 'CR')

   def fix_pixels_CR(self):

      """
      Fix newly interpolated bits and CRs found in the mask and weight
      images. This need to be done after the CR have been detected.
      """

      # Extract the CR detected and INTRP pixels
      CRbit     = self.msk.getPlaneBitMask("CR")    # Gets the value for CR    (2^3=8)
      INTERPbit = self.msk.getPlaneBitMask("INTRP") # Gets the value for INTRP (2^2=4)

      # Create a boolean selection CR-mask and INTRP from the requested bits
      masked_interp = (self.mska & INTERPbit) > 0 
      masked_cr     = (self.mska & CRbit) > 0
      NCR           = len(masked_cr[masked_cr==True])
      print "# Detected:   %d CRs in %s " % (len(self.crs),self.fileName)
      print "# Containing: %d CR-pixels" % NCR

      # 1. The Science 
      # Put back the CRs in case we don't want to interp
      # This is a WORK AROUND FOR crConfig.keepCRs option for now
      if not self.interpCR:
         self.scia[masked_cr] = self.OUT_SCI[masked_cr] 
      self.OUT_SCI = self.scia.copy() # This is the output now

      # 2. The Mask
      self.OUT_MSK[masked_cr]         = 16 | self.OUT_MSK[masked_cr]  
      if self.interpCR:
         self.OUT_MSK[masked_interp]  = 4  | self.OUT_MSK[masked_interp]

      # 3. The Weight
      self.OUT_WGT[masked_cr]       = 0 # PLEASE REVISE
      self.OUT_WGT[masked_interp]   = 0 # PLEASE REVISE

   def update_hdr_CR(self):
      """
      Add records to the header of the fits files
      """
      timenow   = time.asctime() # Time stamp for new fits files
      rec1 = {'name':'DESNCRAY', 'value':len(self.crs),'comment':"Number of Cosmic Rays masked"}
      rec2 = {'name':'DESDMCR',  'value':timenow,      'comment':"Date DESDM CRs Masked on image"}
      for h in self.headers:
         h.add_record(rec1)
         h.add_record(rec2)


   def get_outName(self):

      """
      Set up the output name based on the input filename
      """
      # Get extension and basename
      baseName = os.path.basename(self.fileName)
      extName  = os.path.splitext(baseName)[1]

      # Check wether this is fits.fz or fits files and get the outputname
      if extName == '.fz':
         baseNameOut = "%s"      % os.path.splitext(baseName)[0]
      elif extName == '.fits':
         baseNameOut = "%s.fits" % os.path.splitext(baseName)[0]
      else:
         sys.exit("ERROR: Unknown file type extension for %s" % self.fileName)

      # Decide if we want to compress the output, default is not to.
      if self.compress:
         outName = "%s.fz" % os.path.join(self.outdir,baseNameOut)
      else:
         outName = "%s"    % os.path.join(self.outdir,baseNameOut)
         
      self.outName = outName

         
   def write(self,**keys):
      """
      Use fitsio to write the output file compressed or not
      """
      
      # Decide if compress, that will define the fileName, compression type and tile_dims
      self.compress  = keys.get('compress',None)

      # Define type of compresion and tile_dims
      if self.compress:
         self.compress  = 'RICE'
         self.tile_dims = [1,2048]
      else:
         self.compress  = None
         self.tile_dims = None

      # Get the outputname self.outName
      self.get_outName()

      # Write the output file, one HDU at a time
      ofits = fitsio.FITS(self.outName,'rw',clobber=True)
      # Science -- use scia -- ndarray representation
      ofits.write(self.OUT_SCI,header=self.h_sci,compress=self.compress,tile_dims=self.tile_dims)
      # The Mask
      ofits.write(self.OUT_MSK,header=self.h_msk,compress=self.compress,tile_dims=self.tile_dims)
      # The Weight
      ofits.write(self.OUT_WGT,header=self.h_wgt,compress=self.compress,tile_dims=self.tile_dims)
      # Close the file
      ofits.close()
      print >>sys.stderr,"# Wrote: %s" % self.outName

def get_hdu_numbers(FITS):
   """
   Simple function to figure the HDU extensions for DESDM fits files
   in:  a fitsio.FITS object
   out: (sci_ext, msk_ext, wgt_ext) extension numbers 
   """

   sci_ext = None
   msk_ext = None
   wgt_ext = None

   # Loop trough each HDU on the fits file
   for i in range(len(FITS)):
      h = FITS[i].read_header()       # Get the header
      if ('DES_EXT' in h.keys()) :
         extname = h['DES_EXT'].strip()
         if (extname == 'IMAGE') :
            sci_ext = i
         elif (extname == 'MASK') :
            msk_ext = i
         elif (extname == 'WEIGHT'):
            wgt_ext = i

   if (sci_ext is None or msk_ext is None or wgt_ext is None):
      sys.exit("Cannot find IMAGE, MASK, WEIGHT extensions via DES_EXT keyword in header")
    
   return sci_ext,msk_ext,wgt_ext

def get_FWHM(FITS,sci_hdu):

   """ Get the FHWM of the image in pixels """

   header = FITS[sci_hdu].read_header()
   # Read in the pixelscale
   try:
      pixscale = header['PIXSCAL1']
   except:
      pixscale = 0.27
      print "# WARNING: Could not read PIXSCAL1 keyword from Science HDU"
      print "# WARNING: Will default to %s" % pixscale

   # Read in the FWHM
   try:
      fwhm        = header['FWHM']
      fwhm_arcsec = fwhm*pixscale
   except:
      fwhm = 5.5
      fwhm_arcsec = fwhm*pixscale
      print "# WARNING: Could not read FWHM keyword from Science HDU"
      print "# WARNING: Will use FWHM=%s pixels instead" % fwhm

   # Into arcseconds
   if fwhm_arcsec > 1.2:
      print "# WARNING: Header FWHM value: %.2f[arcsec] / %.1f[pixels] is too high -- should not be trusted" % (fwhm_arcsec,fwhm)
      
   return fwhm

def mk2Dimage_from_footprint(footprint,dimensions,output='ndarray',value=1.0):

   '''Create a 2D-image of the from a footprint element'''

   image2D    = afwImage.ImageF(dimensions) 
   image2D[:] = 0x0
   nspan = 0
   # Loop over 
   for cr in footprint:
       for s in cr.getSpans():
           nspan += 1
           x0, x1, y = s.getX0(), s.getX1(), s.getY()
           image2D[x0:x1+1, y] = value

   # Make it a numpy array
   if output == 'ndarray' or output == 'numpy':
      print "# Will return Numpy Array"
      return image2D.getArray()           
   else:
      return image2D

   return


# Safe extraction of filename
def extract_filename(filename):
    if filename[0] == "!":
        filename=filename[1:]
    filename = os.path.expandvars(filename)
    filename = os.path.expanduser(filename)
    return filename


# Format time
def elapsed_time(t1,verb=False):
    import time
    t2    = time.time()
    stime = "%dm %2.2fs" % ( int( (t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    if verb:
        print >>sys.stderr,"Elapsed time: %s" % stime
    return stime


def cmdline():

    import argparse
    parser = argparse.ArgumentParser(description="Performs CR masking of DECam images using the LSST python framework")

    # The positional arguments
    parser.add_argument("fileName", help="Fits file to process")
    parser.add_argument("outdir",   help="Path to output files [will preserve same name]")

    # The optional arguments
    parser.add_argument("--interpCR", action="store_true", default=True,
                        help="Interpolate CR in Science image [default=True]")

    parser.add_argument("--noInterpCR", action="store_true", default=False,
                        help="Do not Interpolate CR in Science image [default=False]")

    parser.add_argument("--dilateCR", action="store_true", default=False,
                        help="Dilate CR mask by 1 pixel [default=False]")

    parser.add_argument("--nGrowCR", type=int, action="store", default=1,
                        help="Dilate CR mask by nGrowCR pixels [default=False]")

    parser.add_argument("--compress", action="store_true", default=False,
                        help="RICE/fpack compress output file [default=False]")

    #parser.add_argument("--tile_dims", action="store", type=int, default=None, nargs=2,
    #                    help="ZTILE1,ZTILE2 dimensions for compression of output file [default=None]")

    parser.add_argument("--fwhm", type=float, action="store", default=None,
                        help="Set a FWHM [pixels] value that overides the header FWHM value in image")

    parser.add_argument("--minSigma", type=float, action="store", default=5.0,
                        help="CRs must be > this many sky-sig above sky")

    parser.add_argument("--min_DN", type=int, action="store", default=150,
                        help="CRs must have > this many DN (== electrons/gain) in initial detection")

    args = parser.parse_args()

    if args.noInterpCR:
        args.interpCR = False

    print "# Will run:"
    print "# %s " % parser.prog
    for key in vars(args):
        print "# \t--%-10s\t%s" % (key,vars(args)[key])

    return args

if __name__ == "__main__":

   # Get the start time
   t0 = time.time()
   
   args = cmdline()

   desobj = DESMaskCRs(args.fileName,args.outdir)
   desobj.CRs(FWHM     = args.fwhm,
              dilateCR = args.dilateCR,
              interpCR = args.interpCR,
              minSigma = args.minSigma,
              min_DN   = args.min_DN,
              nGrowCR  = args.nGrowCR)

   # Write it out
   desobj.write(compress=args.compress)
   print >>sys.stderr,"# Time:%s" % elapsed_time(t0)


   
   
