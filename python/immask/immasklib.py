#!/usr/bin/env python

"""
Suite of fuctions and class for the DESDM immask python module.

$Id$
$Rev::                                  $:  # Revision of last commit.
$LastChangedBy::                        $:  # Author of last commit.
$LastChangedDate::                      $:  # Date of last commit.

@authors: Felipe Menanteau    <felipe@illinois.edu>
@authors: Eli Rykoff          <rykoff@slac.stanford.edu>
@authors: Alex Drlica-Wagner  <kadrlica@fnal.gov>
"""

import os
import sys
import fitsio 
import time
import copy
import argparse
from collections import OrderedDict

import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import fmin
from scipy.spatial  import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.path

from despyutils import wcsutil
from pyhough.pyhough_lib import Hough


#####################
### Image objects ###
#####################

class DESImage(object):
    """
    A Class to handle DECam fits/fits.fz files. Uses fitsio to 
    open/read/close files. 
    """

    # Should not require outname and outdir
    def __init__(self, filename, outname, **kwargs):
        self.filename  = self.extract_filename(filename)
        self.outname   = self.extract_filename(outname)

        # Decide whether output should be compressed
        self.compress  = kwargs.get('compress',False)

        # Gets SCI, MSK, WGT and created VAR
        self.read()
        # Make shallow copies of HDUs and created VAR (OUT_*)
        self.copy_ndarrays() 

    def __repr__(self):
        return repr(self.ifits)

    def __str__(self):
        return str(self.ifits)

    # This could be moved to a utils module
    @staticmethod
    def extract_filename(filename):
        """ Safe extraction of filename """
        if filename[0] == "!": filename=filename[1:]
        filename = os.path.expandvars(filename)
        filename = os.path.expanduser(filename)
        return filename

    def get_hdu_numbers(self):
        """
        Simple function to figure the HDU extensions for DESDM fits files
        in:  a fitsio.FITS object
        out: (sci_ext, msk_ext, wgt_ext) extension numbers 
        """
        FITS = self.ifits

        sci_ext = None
        msk_ext = None
        wgt_ext = None
        # Loop trough each HDU on the fits file
        for i in range(len(FITS)):
            h = FITS[i].read_header()       # Get the header
            if ('DES_EXT' in h.keys()) :
                extname = h['DES_EXT'].strip()
                if   (extname == 'IMAGE') : sci_ext = i
                elif (extname == 'MASK')  : msk_ext = i
                elif (extname == 'WEIGHT'): wgt_ext = i
         
        if (sci_ext is None or msk_ext is None or wgt_ext is None):
            raise ValueError("Cannot find IMAGE, MASK, WEIGHT extensions via DES_EXT keyword")
         
        return sci_ext,msk_ext,wgt_ext

    def get_FWHM(self):
       """ Get the FHWM of the image in pixels. """
       FITS = self.ifits
       sci_hdu = self.sci_hdu
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

    def read(self):
        """
        Read image HDUs as ndarrays and headers as dictionaries 
        using fitsio.
        """
        # Get the fitsio element -- we'll modify this in place
        print "# Reading in extensions and headers for %s" % self.filename
        self.ifits = fitsio.FITS(self.filename,'r')
        sci_hdu, msk_hdu, wgt_hdu = self.get_hdu_numbers()
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
        print "# Done reading HDU "
  
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
        # A handy handle
        self.headers = (self.h_sci,self.h_msk,self.h_wgt)
        # Let's make a handle for the DESDM object
        self.DESDMImage = (self.OUT_SCI,self.OUT_MSK,self.OUT_WGT)

    def check_output(self):
        """
        Make sure that fpack files have the .fz extension
        """
        basename = os.path.basename(self.outname)
        extname  = os.path.splitext(basename)[1]
      
        if self.compress and extname == '.fits':
            raise IOError ("--compress specified with '.fits' outfile")

        if not self.compress and extname == '.fz':
            raise IOError ("--compress not specified with '.fz' outfile")

    def write(self):
        """
        Use fitsio to write the output file compressed or not
        """

        # Define type of compresion and tile_dims
        if self.compress:
            self.compress  = 'RICE'
            self.tile_dims = [1,2048]
        else:
            self.compress  = None
            self.tile_dims = None
         
        # Check the output name is consistent with compression
        self.check_output()
        # Write the output file, one HDU at a time
        ofits = fitsio.FITS(self.outname,'rw',clobber=True)
        # Science -- use scia -- ndarray representation
        ofits.write(self.OUT_SCI,header=self.h_sci,compress=self.compress,tile_dims=self.tile_dims)
        # The Mask
        ofits.write(self.OUT_MSK,header=self.h_msk,compress=self.compress,tile_dims=self.tile_dims)
        # The Weight
        ofits.write(self.OUT_WGT,header=self.h_wgt,compress=self.compress,tile_dims=self.tile_dims)
        # Close the file
        ofits.close()
        print >>sys.stderr,"# Wrote: %s" % self.outname

######################
### Masker objects ###
######################

class BaseMasker(object):
    """
    Base class for defining image maskers. The basic tenents of
    a masker are as follows:
    - By default, masking is performed in object creation
    - The constructor arguments are listed in an ordered dict
      which is also converted into a argparse object
    """

    defaults = OrderedDict([])

    def __init__(self, image, outdir, **kwargs):
        """ Create and run the masker """
        self.image = image
        self.outdir = outdir

        if not os.path.exists(self.outdir):
            print "# Creating output directory  %s" % self.outdir
            os.mkdir(self.outdir)

        self._parse(**kwargs)
        self.run()

    def run(self):
        """ Run the masking routines """
        pass

    def write(self):
        """ Write the masker QA files """
        pass
        
    def _parse(self, **kwargs):
        for key in np.intersect1d(self.defaults.keys(),kwargs.keys()):
            self.__dict__[key] = kwargs[key]

    @classmethod
    def argparser(cls, title, **kwargs):
        """ 
        Fill ArgumentParser with class defaults as optional arguments.
        """
        if 'add_help' not in kwargs: kwargs['add_help']=False

        parser = argparse.ArgumentParser(**kwargs)
        group = parser.add_argument_group(title)
        args = OrderedDict(cls.defaults)
        for key,value in args.items():
            group.add_argument('--%s'%key,dest=key,**value)
        return parser

class CosmicMasker(BaseMasker):

    ### Command line arguments for cosmic-ray masking
    defaults = OrderedDict([
        ['interpCR'  , dict(default=True, action="store_true", help="Interpolate CR in science image.")],
        ['noInterpCR', dict(action="store_true", help="Do not Interpolate CR in Science image.")],
        ['dilateCR'  , dict(action="store_true", help="Dilate CR mask by 1 pixel.")],
        ['nGrowCR'   , dict(default=1, type=int, help="Dilate CR mask by nGrowCR pixels.")],
        ['fwhm'      , dict(default=None, type=float, help="Set a FWHM [pixels] value that overrides the header FWHM value in image")],
        ['minSigma'  , dict(default=5, type=float, help="CRs must be > this many sky-sig above sky")],
        ['min_DN'    , dict(default=150, type=int, help="CRs must have > this many DN (== electrons/gain) in initial detection")],
        ])

    def run(self):
        """
        Top-level function for CR masking
        """

        # Make the individual calls
        self.make_BAD_mask() # True by default, unless we decide not to.
        self.make_lsst_image()
        self.find_CRs()
        self.fix_pixels_CR()
        self.update_header()

  
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
        masked_bad        = (self.image.MSK & 1) > 0
        masked_bad_interp = ((self.image.MSK & 4) > 0) & masked_bad
         
        # Create a boolean mask of the glowing edges, we don't need to
        # keep the original values for later, as they are kept in the OUT_MSK copy.
        masked_edge = (self.image.MSK & 512) > 0
         
        # Temporaly change values of 513 to 512 in order to avoid
        # interpolating over glowing edges of ccds also labeled as BAD (512+1=513)
        self.image.MSK[masked_edge] = 512
         
        # Temporaryly change the values of BAD and INTERP pixels from 5
        # to 4 for the MSK ndarray as we do not want to interpolate twice.
        self.image.MSK[masked_bad_interp] = 4
  
  
    def make_lsst_image(self):
        import lsst.afw.image  as afwImage

        """
        Create and LSST-like image from the SCI, MSK and WGT
        ndarrays. We pass them into the LSST framework structure to call
        the CR finder.
        """
  
        print "# Making LSST image structure from SCI, MSK and VAR"
        # 0 Set up the Mask Plane dictionay for DECam
        self.set_DECamMaskPlaneDict()
        # 1 - Science
        self.sci = afwImage.ImageF(self.image.SCI)
        # 2- The Mask plane Image and rewrite mask planes into LSST convention
        self.msk = afwImage.MaskU(self.image.MSK)
        self.msk.conformMaskPlanes(self.DECamMaskPlaneDict) 
         
        # 3 - The variance Image
        self.WGT_fixed = np.where(self.image.WGT<=0, self.image.WGT.max()/1e6, self.image.WGT) # Fix values < 0
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
            print "# Fixing weight map with median=%.1f for those %d CR-pixels"%(NCR_prev, median_rep)
            self.vara[cr_prev]    = median_rep # The LSST handle
            self.image.OUT_WGT[cr_prev] = median_rep # The original array
            print "# Fixing mask plane for those CR-pixels too " 
            self.mska[cr_prev]    = self.mska[cr_prev]    - 16 # The LSST handle
            self.image.OUT_MSK[cr_prev] = self.image.OUT_MSK[cr_prev] - 16 # The original array
        # ************************************************************
        # Make an LSST masked image (science, mask, and weight) 
        self.mi = afwImage.makeMaskedImage(self.sci, self.msk, self.var)
  
    def find_CRs(self,**kwargs):
        # Load the the LSST modules here to avoid problems elsewhere in case they are not present
        import lsst.meas.algorithms as measAlg
        import lsst.ip.isr          as ip_isr
        import lsst.afw.math        as afwMath
        import lsst.pex.config      as pexConfig
         
        """
        Find the Cosmic Rays on a Science Image using lsst.meas.algorithms.findCosmicRays
        """
         
        # Estimate the PSF of the science image.
        if not self.fwhm:
            print "# Will attempt to get FWHM from the image header"
            self.fwhm  = self.image.get_FWHM()
            xsize = int(self.fwhm*9)
            psf   = measAlg.DoubleGaussianPsf(xsize, xsize, self.fwhm/(2*np.sqrt(2*np.log(2))))
         
        # Interpolate bad pixels before finding CR to avoid false detections
        fwhm = 2*np.sqrt(2*np.log(psf.computeShape().getDeterminantRadius()))
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
        crConfig.nCrPixelMax = int(self.image.nx*self.image.ny/3) # 1e6 will not work, needs an integer
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
        print "# Detected:   %d CRs in %s " % (len(self.crs),self.image.filename)
        print "# Containing: %d CR-pixels" % NCR
         
        # 1. The Science 
        # Put back the CRs in case we don't want to interp
        # This is a WORK AROUND FOR crConfig.keepCRs option for now
        if not self.interpCR:
            self.scia[masked_cr] = self.image.OUT_SCI[masked_cr] 
        self.image.OUT_SCI = self.scia.copy() # This is the output now
         
        # 2. The Mask
        self.image.OUT_MSK[masked_cr] = 16 | self.image.OUT_MSK[masked_cr]  
        if self.interpCR:
            self.image.OUT_MSK[masked_interp] = 4 | self.image.OUT_MSK[masked_interp]
         
        # 3. The Weight
        self.image.OUT_WGT[masked_cr]     = 0 # PLEASE REVISE
        self.image.OUT_WGT[masked_interp] = 0 # PLEASE REVISE
  
    def update_header(self):
        """
        Add records to the header of the fits files
        """
        timenow   = time.asctime() # Time stamp for new fits files
        rec1 = {'name':'DESNCRAY', 'value':len(self.crs),'comment':"Number of cosmic rays masked"}
        rec2 = {'name':'DESDMCR',  'value':timenow,      'comment':"Date DESDM CRs masked on image"}
        for h in self.image.headers:
            h.add_record(rec1)
            h.add_record(rec2)

    
class StreakMasker(BaseMasker):
    """
    Masking satellite, UFO, etc. streaks.
    """

    defaults = OrderedDict([
        ['bkgfile'     , dict(required=True,help="Input background FITS file (fz/fits)")],
        ['draw'        , dict(action='store_true',help="Use matplotlib to draw diagnostic plots.")],
        ['template_dir', dict(default="/dev/null",help="Directory containing Hough template.")],
        # Image pre-processing
        ['bin_factor'   , dict(default=8, type=int, help="Binning factor to beat down sky noise")],
        # For detection, merging, and characterization
        ['nsig_sky'     , dict(default=1.5,type=float,help="Threshold for sky noise") ],
        ['nsig_detect'  , dict(default=14.,type=float,help="Threshold for Hough peak detection") ],
        ['nsig_merge'   , dict(default=8.,type=float,help="Threshold for Hough peak merging ")],
        ['nsig_mask'    , dict(default=8.,type=float,help="Threshold for Hough peak characterization")],
        # Quality cuts
        ['max_width'    , dict(default=150.,type=float,help="Maximum width in (pix)")],
        ['max_angle'    , dict(default=15., type=float,help="Maximum angular extent (deg)")],
        # For clipping the ends of streak
        ['clip'         , dict(action="store_true",help="Clip underpopulated ends of the streak.")],
        ['nsig_clip'    , dict(default=2.,  type=float,help="Clip chunks that are more than 'nsig' underdense.")],
        ['clip_angle'   , dict(default=45., type=float,help="Clip streaks close to the given abs. angle (deg).")],
        ['clip_range'   , dict(default=10., type=float,help="Allowed clip angle range (deg); >90 to accept all angles")],
        # For masking
        ['mask_factor'  , dict(default=1.5, type=float, help="Factor to increase streak width for masking")],
        ['maskbits'     , dict(default=1023,type=int,help="Ignore these mask bits ")],
        ['setbit'       , dict(default=1024,type=int,help="New streak mask bit")],
        ['maxmask'      , dict(default=1000,type=int,help="Maximum number of streaks to mask [NOT IMPLEMENTED]")],
        # Streak objects
        ['write_streaks', dict(action="store_true", help="Write out streak objects")],
        ])

    def run(self):
        """
        The primary function for masking streaks in images.
        Mask streaks from airplanes, satellites, UFOs, etc.
        by using a Hough transform. 
        @authors: Eli Rykoff            <rykoff@slac.stanford.edu>
        @authors: Alex Drlica-Wagner    <kadrlica@fnal.gov>
        --
        Made into a callable python class and stand-alone
        @updates: Felipe Menanteau <felipe@illinois.edu> 
        --
        """
        ####################################################

        # Read in the background image nd-array
        self.BKG    = self.read_bkg_image(self.bkgfile)
        #self.BKG    = DESImage(self.bkgfile).BKG
         
        # Make a background substracted image
        self.subIm = self.image.OUT_SCI - self.BKG
        # Create a boolean selection mask from the requested bits
        self.masked = ((self.image.OUT_MSK & self.maskbits ) > 0)
         
        # Re-bin the image (if requested)
        if self.bin_factor > 1:
            print "# Binning image by a factor of %s..."% self.bin_factor
            self.subIm  = self.bin_pixels(self.subIm,self.bin_factor)
            self.masked = np.ceil(self.bin_pixels(self.masked,self.bin_factor)).astype(bool)
           
        # Calculate sky noise
        print "# Measuring sky noise..."
        usepix = np.where(~self.masked)
        self.sky_bkg, self.sky_err, self.sky_skew = self.mmm(self.subIm[usepix])
         
        # allow objects to be connected diagonally, create the structure
        self.structure = ndimage.generate_binary_structure(2,2)
         
        # The thresholded image to pass to the Hough transform
        self.searchIm = ( (self.subIm > self.nsig_sky*self.sky_err) & ~self.masked )
         
        # Should do a maxarea cut here...
        print "# Performing Hough transform now..."
        self.hough, self.theta, self.rho = Hough(self.searchIm).transform()
         
        # Read in the Hough template or create one
        self.get_template()
         
        # Normalize the transform in terms of sigma
        norm  = self.fit_template(self.hough,self.template)
        mean  = norm*self.template
        sigma = np.sqrt(norm*self.template)
        self.trans = np.zeros(self.hough.shape)
        idx = np.nonzero(sigma)
        self.trans[idx] = (self.hough[idx] - mean[idx]) / sigma[idx]
        #self.trans = np.nan_to_num( (self.hough - mean) / sigma )
        print "# Maximum Hough value: %s" % self.trans.max()
         
        # Perform detection using structure and label
        self.detect_label,self.detect_nlabel = ndimage.label(self.trans > self.nsig_detect, structure=self.structure)
        if self.detect_nlabel:
            hist,edges,rev = self.idl_histogram(self.detect_label,bins=np.arange(self.detect_nlabel+2))
            maxarea = int(0.1*self.detect_label.size) # Get rid of background object
            good, = np.where((hist > 1) & (hist < maxarea))
            #good, = np.where( (hist > 1) )
            print "# Detected %i streaks above a threshold of %s"%(len(good),self.nsig_detect)
        else:
            print "# No streaks detected."
            rev,good = [],[]
         
        # Detection statistics
        print "# Calculating detection statistics..."
        self.detect_objs = self.streak_statistics(self.trans, self.rho, self.theta, good, rev)
        self.detect_objs['BINNING'][:] = self.bin_factor
         
        # Make some preliminary quality cuts
        cut = np.zeros(len(self.detect_objs),dtype=self.detect_objs['CUT'].dtype)
        # Perpendicular objects that look like a readout error...
        cut |= (np.abs(self.detect_objs['SLOPE']) > 150) & (self.detect_objs['WIDTH'] <= 3)
        # Some additional cuts that could be useful -- future
        #cut |= ... # Central bias jump (x-dir)
        #cut |= ... # Central bias jump (y-dir)
        self.detect_objs['CUT'][np.nonzero(cut)] = 1
        print "# Found %i lines passing quality cuts" % (self.detect_objs['CUT']==0).sum()
         
        # Merge nearby lines
        merge_label,merge_nlabel = ndimage.label(self.trans > self.nsig_merge,structure=self.structure)
        if merge_nlabel:
            hist,edges,rev = self.idl_histogram(merge_label,bins=np.arange(merge_nlabel+2))
            good = []
            for obj in self.detect_objs[self.detect_objs['CUT']==0]:
                lab = merge_label[obj['MAX_Y0'],obj['MAX_X0']]
                good.append(lab)
            unique = np.unique(good)
            nmerged = len(good) - len(unique)
            print "# Merging %i streaks above a threshold of %s"%(nmerged,self.nsig_merge)
        else:
            print "# No objects found for merging."
            rev,unique = [],[]
         
        print "# Calculating merging statistics..."
        merge_objs = self.streak_statistics(self.trans, self.rho, self.theta, unique, rev)
        merge_objs['BINNING'][:] = self.bin_factor
         
        # More quality cuts
        cut = np.zeros(len(merge_objs),dtype=merge_objs['CUT'].dtype)
        # Objects that are too wide...
        cut |= (merge_objs['WIDTH'] > self.max_width/float(self.bin_factor))
        # Objects that span too large an opening angle -- to curvy
        delta_theta = np.abs(self.theta[merge_objs['XMAX']] - self.theta[merge_objs['XMIN']])
        cut |= delta_theta > np.radians(self.max_angle)
         
        merge_objs['CUT'][np.nonzero(cut)] = 1
        print "# Found %i streaks passing quality cuts"%((merge_objs['CUT']==0).sum())
         
        # Masking threshold
        self.mask_label,mask_nlabel = ndimage.label(self.trans > self.nsig_mask,structure=self.structure)
        if mask_nlabel:
            hist,edges,rev = self.idl_histogram(self.mask_label,bins=np.arange(mask_nlabel+2))
            good = []
            
            for obj in merge_objs[merge_objs['CUT']==0]:
                lab = self.mask_label[obj['MAX_Y0'],obj['MAX_X0']]
                good.append(lab)
          
            unique = np.unique(good)
            nmerged = len(good) - len(unique)
            if nmerged: 
                print "# Merging %i additional streaks"%(nmerged)
        else:
           print "# No objects found for masking."
           rev,unique = [],[]
         
        print "# Calculating masking statistics..."
        self.mask_objs = self.streak_statistics(self.trans, self.rho, self.theta, unique, rev)
        self.mask_objs['BINNING'][:] = self.bin_factor
         
        # Create the mask from the streaks
        print "# Masking %i streaks"%(len(self.mask_objs))
        streak_mask = np.zeros(self.image.OUT_MSK.shape,dtype=self.image.OUT_MSK.dtype)
        wcs = wcsutil.WCS(self.image.h_sci)
        for i,obj in enumerate(self.mask_objs):
            slope  = obj['SLOPE']
            inter1 = obj['INTER1']
            inter2 = obj['INTER2']
            # Print intercept in original image coordinates
            print "#  %i   NSIG=%g, SLOPE=%g, INTER1=%g, INTER2=%g"%(i,obj['MAX'],slope,inter1*self.bin_factor,inter2*self.bin_factor)
          
            tmp_mask = np.zeros(self.searchIm.shape,dtype=self.image.OUT_MSK.dtype)
            tmp_mask = self.mask_between(tmp_mask,slope,inter1,inter2,self.mask_factor)
          
            clip = np.abs(np.abs(np.degrees(np.arctan(slope)))-self.clip_angle) < self.clip_range
            if self.clip and clip:
                print "# Examining streak for clipping..."
                tmp_mask = self.clip_mask(self.searchIm,self.masked,tmp_mask,slope,self.nsig_clip)
          
            # Move to original resolution
            tmp_mask = tmp_mask.repeat(self.bin_factor,axis=0).repeat(self.bin_factor,axis=1)
            # Smooth binning artifacts
            vertices = self.create_rectangle(tmp_mask,slope)
            obj['CORNERS'][:] = vertices
          
            #vertices_wcs = np.vstack(self.image_to_wcs(vertices[:,0],vertices[:,1])).T 
            vertices_wcs = np.vstack(wcs.image2sky(vertices[:,0],vertices[:,1])).T
            obj['CORNERS_WCS'][:] = vertices_wcs
          
            # Move to the full resolution mask
            streak_mask |= self.mask_polygon(streak_mask,vertices)
            
            #streak_mask |= mask_between(streak_mask,slope,inter1*self.bin_factor,inter2*self.bin_factor,self.mask_factor)
         
        ypix,xpix = np.nonzero(streak_mask)
         
        # set the streak bit
        self.image.OUT_MSK[ypix,xpix] = self.image.OUT_MSK[ypix,xpix] | self.setbit 
        # and zero out the weight
        self.image.OUT_WGT[ypix,xpix] = 0
         
        # Draw Plots
        if self.draw:
            self.draw_plots()
        
        # Write streak objects
        if self.write_streaks:
            self.write_streak_objects() 

    def write_streak_objects(self):
        basename = os.path.basename(self.image.outname)
        outbase = basename.split('.fit')[0]+'_streaks.fits'
        outname = os.path.join(self.outdir,outbase)

        #logger.info("Writing objects: %s" % (objsfile))
        print "# Writing streak objects: %s" % (outname)
        fitsio.write(outname,self.mask_objs,clobber=True)
  
    def write_streak_mask(self):
        basename = os.path.basename(self.image.outname)
        outbase = basename.split('.fit')[0]+'_mask.fits'
        outname = os.path.join(self.image.outdir,outbase)

        print "# Writing streak mask: %s" % (outname)
        
        maskonly = np.zeros(self.image.OUT_MSK.shape,dtype=self.image.OUT_MSK.dtype)
        test = np.where((self.image.OUT_MSK & self.setbit) > 0)
        maskonly[test[0],test[1]] = self.setbit
   
        # Write mask as float32 so that "ds9 -mask" reads properly
        header = copy.copy(self.image.h_msk)
        header['BZERO'] = 0 
        fitsio.write(outname,maskonly.astype('f4'),header=header,clobber=True)
  
    def draw_plots(self):
        """
        Draw the streak finder dignostic plot with matplotlib
        """
        import pylab as plt
         
        # Get the drawbase name
        basename      = os.path.basename(self.image.filename)
        outbase       = os.path.splitext(os.path.splitext(basename)[0])[0] # double split for '.fits.fz' files
        drawbase = os.path.join(self.outdir,outbase)
         
        # The sky noise
        pngfile = "%s_sky.png" % drawbase
        print "# Drawing sky noise: %s " % pngfile
        fig, axes = plt.subplots(1, 3)
        sigma     = [1,2,3]
        for ax,sig in zip(axes,sigma):
            label = ndimage.label(((self.subIm>sig*self.sky_err) & ~self.masked),structure=self.structure)
            ax.imshow(label[0]>0,origin='bottom'); 
            ax.set_title(r"Sky Noise ($%s\sigma$)"%sig); 
        plt.savefig(pngfile)
         
        # Draw the input mask
        pngfile = "%s_bits.png" % drawbase
        print "# Drawing input mask: %s " % pngfile
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(self.subIm > self.nsig_sky*self.sky_err,origin='bottom')
        axes[0].set_title("Threshold Image")
        axes[1].imshow(self.masked,origin='bottom')
        axes[1].set_title("Input Mask")
        axes[2].imshow(self.searchIm,origin='bottom')
        axes[2].set_title(r"Search Image")
        plt.savefig(pngfile)
         
        # Draw the lines and masks
        pngfile = "%s_mask.png" % drawbase
        print "# Drawing streaks and masks: %s" % pngfile
        ymax,xmax = self.searchIm.shape; x = np.arange(xmax)
        fig, axes = plt.subplots(1, 3)
        titles = ["Sky Noise","Streaks","Mask"]
        for ax,t in zip(axes,titles):
            ax.imshow(self.searchIm,origin='bottom'); ax.set_title(t)
        plt.sca(axes[1])
        for obj in self.mask_objs:
            y1 = obj['SLOPE']*x+obj['INTER1']; plt.plot(x,y1,'--w')
            y2 = obj['SLOPE']*x+obj['INTER2']; plt.plot(x,y2,'--w')
        if (self.image.OUT_MSK & self.setbit).sum():
            plt.sca(axes[2]) 
            bin_mask = self.bin_pixels(self.image.OUT_MSK & self.setbit,self.bin_factor).astype(bool)
            bin_mask = np.ma.array(bin_mask,mask=(bin_mask==0))
            cmap = matplotlib.cm.binary; cmap.set_bad('k',0)
            plt.imshow(bin_mask,origin='bottom',alpha=0.5,cmap=cmap)
        for ax in axes: ax.set_xlim(0,xmax); ax.set_ylim(0,ymax)
        plt.savefig(pngfile)
         
        #  Draw Hough transform at various thresholds
        pngfile = "%s_hough.png" % drawbase
        print "# Drawing Hough transform: %s" % pngfile
        fig, axes = plt.subplots(2,2,figsize=(12,6),sharey='all',sharex='all')
        axes = axes.flatten()
        extent = [self.theta.min(),self.theta.max(),self.rho.min(),self.rho.max()]
        titles = ["Hough","Normalized",
                  r'Detect $(%g \sigma)$'% self.nsig_detect,
                  r"Mask $(%g \sigma)$"  % self.nsig_mask]
        data   = [self.hough,self.trans,self.detect_label,self.mask_label]
        cmap = matplotlib.cm.jet; cmap.set_bad('w',0)
        for ax,d,t in zip(axes,data,titles):
            #d = np.ma.masked_array(d,mask=~(d>0))
            ax.imshow(d,extent=extent,aspect='auto',cmap=cmap,vmin=0)
            ax.set_title(t)
        axes[0].set_ylabel(r'Distance (pix)'); 
        axes[2].set_ylabel(r'Distance (pix)'); 
        axes[2].set_xlabel(r'Angle (rad)')
        axes[3].set_xlabel(r'Angle (rad)')
        plt.savefig(pngfile)

    def get_template(self):
        """
        Read in the Hough template from file of binned image shape
        """
        template_file = "hough_template_x%i_y%i.npy" % self.searchIm.shape[::-1]
        filename = os.path.join(self.template_dir,template_file)
        if os.path.exists(filename):
            print "# Reading template from %s" % filename
            self.template = read_template(filename)[0]
        else:
            print "# No template Found --> calculating Hough normalization..."
            template_image = (np.ones(self.searchIm.shape,dtype=bool) & ~self.masked)
            self.template = Hough(template_image).transform()[0]

    # This is the start of routines that are more or less
    # loosely bound to the StreakMasker class. The first
    # set of methods are unlikely to be used outside of 
    # streak masking and could easily be conceived of
    # as regular methods of the class (though were not
    # originally written that way).

    @staticmethod
    def streak_statistics(array, rho, theta, good, rev):
        """
        Calculate characteristics of the streaks from the "island"
        defining the streak in Hough space.
     
        Parameters:
        array : Hough transform array
        theta : Theta values for Hough array (size = array.shape[1])
        rho   : Rho values for Hough array (size = array.shape[0]
        good  : Array containg labels for each "island" in Hough array
        rev   : Reverse index array containing [label,indices]
        Returns:
        objs  : recarray of object characteristics
        """
        ngood = len(good)
        objs = np.recarray((ngood,),
                           dtype=[('LABEL','i4'),
                                  ('NPIX','f4'),
                                  ('MAX','f4'),
                                  ('MAX_X0','i4'),
                                  ('MAX_Y0','i4'),
                                  ('XMIN','i4'),
                                  ('XMAX','i4'),
                                  ('YMIN','i4'),
                                  ('YMAX','i4'),
                                  ('CEN_X0','f4'),
                                  ('CEN_Y0','f4'),
                                  ('CEN_Y1','f4'),
                                  ('CEN_Y2','f4'),
                                  ('SLOPE','f4'),
                                  ('INTER1','f4'),
                                  ('INTER2','f4'),
                                  ('WIDTH','f4'),
                                  ('BINNING','i4'),
                                  ('CORNERS','f4',(4,2)),
                                  ('CORNERS_WCS','f8',(4,2)),
                                  ('CUT','i2'),])
        objs['CUT'][:] = 0
        objs['BINNING'][:] = 0
        objs['CORNERS'][:] = 0
        objs['CORNERS_WCS'][:] = 0
     
        shape = array.shape
        ncol = shape[1]
        for i in range(0,ngood):
            #logger.debug("i=%i",i)
            # This code could use some cleanup...
            i1a=rev[rev[good[i]]:rev[good[i]+1]]
            xpix = i1a % ncol
            ypix = i1a / ncol
            pix  = zip(xpix,ypix)
            npix = len(xpix)
            objs[i]['LABEL'] = good[i]
            objs[i]['NPIX'] = npix
            #logger.debug("LABEL=%i"%objs[i]['LABEL'])
            #logger.debug("NPIX=%i"%objs[i]['NPIX'])
     
            # This would be slow with more candidate lines...
            island = np.zeros(shape)
            island[ypix,xpix] = array[ypix,xpix]
     
            # At the location of the maximum value
            idx = island.argmax()
            ymax = idx // ncol
            xmax = idx % ncol
            objs[i]['MAX']   = island[ymax,xmax]
            objs[i]['MAX_X0'] = xmax
            objs[i]['MAX_Y0'] = ymax
            #logger.debug("MAX=%i"%objs[i]['MAX'])
            #logger.debug("MAX_X0=%i"%objs[i]['MAX_X0'])
            #logger.debug("MAX_Y0=%i"%objs[i]['MAX_Y0'])
     
            # Full extent of the island
            objs[i]['XMIN'] = xpix.min()
            objs[i]['XMAX'] = xpix.max()
            #logger.debug("XMIN=%i, XMAX=%i"%(objs[i]['XMIN'],objs[i]['XMAX']))
            objs[i]['YMIN'] = ypix.min()
            objs[i]['YMAX'] = ypix.max()
            #logger.debug("YMIN=%i, YMAX=%i"%(objs[i]['YMIN'],objs[i]['YMAX']))
     
            # Find pixel closest to centroid (careful of x-y confusion)
            centroid = ndimage.center_of_mass(island)
            tree = cKDTree(pix)
            dist, idx = tree.query([centroid[1],centroid[0]])
            xcent,ycent = pix[idx]
            objs[i]['CEN_X0'] = xcent
            objs[i]['CEN_Y0'] = ycent
            #logger.debug("CEN_X0=%i"%objs[i]['CEN_X0'])
            #logger.debug("CEN_Y0=%i"%objs[i]['CEN_Y0'])
            extent = np.nonzero(island[:,xcent])[0]
            ycent1,ycent2 = extent.min(),extent.max()+1
            objs[i]['CEN_Y1'] = ycent1
            objs[i]['CEN_Y2'] = ycent2
            #logger.debug("CEN_Y1=%i"%objs[i]['CEN_Y1'])
            #logger.debug("CEN_Y2=%i"%objs[i]['CEN_Y2'])
     
            # Calculate the slope and intercept
            theta0 = theta[xcent]
            rho1,rho2 = rho[ycent1],rho[ycent2]
            slope = np.nan_to_num( -np.cos(theta0)/np.sin(theta0) )
            inter1 = np.nan_to_num(rho1/np.sin(theta0))
            inter2 = np.nan_to_num(rho2/np.sin(theta0))
            objs[i]['SLOPE']  = slope
            objs[i]['INTER1'] = inter1
            objs[i]['INTER2'] = inter2
            #logger.debug("SLOPE=%.3g"%objs[i]['SLOPE'])
            #logger.debug("INTER1=%.3g"%objs[i]['INTER1'])
            #logger.debug("INTER2=%.3g"%objs[i]['INTER2'])
     
            objs[i]['WIDTH']  = np.abs(rho1 - rho2)
            #logger.debug("WIDTH=%.3g\n"%objs[i]['WIDTH'] )
        return objs

    @staticmethod
    def create_rectangle(mask, slope, pad=0):
        """
        Create a rectangle at the given angle that bounds the
        masked region. The complication here is that the mask
        can be clipped and thus does not need to extend to 
        the edge of the image. A padding can be added to make
        sure that all the original masked pixels continue to
        be masked after the various rotations.
     
        Parameters:
        mask     : The input pixel mask
        slope    : The slope of the streak
        pad      : The padding (in pixels) to apply to vertices
        Returns:
        vertices : Array of rectangle vertices
        """
        ypix, xpix = np.nonzero(mask)
     
        # Rotate the masked streak to horizontal (hence the minus sign)
        alpha = np.arctan(slope)
        xpixp = np.cos(-alpha) * xpix - np.sin(-alpha) * ypix
        ypixp = np.sin(-alpha) * xpix + np.cos(-alpha) * ypix
     
        # Find the extrimum of the rotated x and y coordinates
        xminp,xmaxp = xpixp.min(),xpixp.max()
        yminp,ymaxp = ypixp.min(),ypixp.max()
     
        # Define the rotated vertices
        xvertp = np.array([xminp,xminp,xmaxp+pad,xmaxp+pad])
        yvertp = np.array([yminp,ymaxp+pad,ymaxp+pad,yminp])
     
        # Rotate back to image coordinates
        xvert = np.cos(alpha) * xvertp - np.sin(alpha) * yvertp
        yvert = np.sin(alpha) * xvertp + np.cos(alpha) * yvertp
        vertices = np.array( zip(xvert,yvert))
     
        return vertices
     
    @staticmethod
    def create_polygon(mask, pad=0.5):
        """
        The polygon created does not necessarily bound all of the 
        pixels in the original mask and should instead be thought 
        of as an approximate transformation from a mask to a polygon.
        This function is complicated by the fact that the mask can 
        intersect the edge of the CCD making it non-quadralateral.
        Vertices are named assuming the origin is located in the
        lower-left corner. 
     
        Parameters:
        mask     : The input pixel mask
        pad      : The padding (in pixels) to apply to vertices
        Returns:
        vertices : Array of polygon vertices
        """
        vertices = []
        ypix,xpix = np.nonzero(mask)
        xmin = xpix.min(); xmax = xpix.max()
        ymin = ypix.min(); ymax = ypix.max()
     
        # Start on the lower left
        xmin_ypix = ypix[xpix == xmin]
        xmin_ymin = xmin_ypix.min(); xmin_ymax = xmin_ypix.max()
        left_bottom = (xmin-pad,xmin_ymin-pad)
        left_top    = (xmin-pad,xmin_ymax+pad)
        for vertex in [left_bottom,left_top]:
            if vertex not in vertices: vertices.append(vertex)
     
        # Move to the top
        ymax_xpix = xpix[ypix == ymax]
        ymax_xmin = ymax_xpix.min(); ymax_xmax = ymax_xpix.max()
        top_left  = (ymax_xmin-pad,ymax+pad)
        top_right = (ymax_xmax+pad,ymax+pad)
        for vertex in [top_left,top_right]:
            if vertex not in vertices: vertices.append(vertex)
            
        # Now to the right
        xmax_ypix = ypix[xpix == xmax]
        xmax_ymin = xmax_ypix.min(); xmax_ymax = xmax_ypix.max()
        right_top    = (xmax+pad,xmax_ymax+pad)
        right_bottom = (xmax+pad,xmax_ymin-pad)
        for vertex in [right_top,right_bottom]:
            if vertex not in vertices: vertices.append(vertex)
     
        # Finally to the bottom
        ymin_xpix = xpix[ypix == ymin]
        ymin_xmin = ymin_xpix.min(); ymin_xmax = ymin_xpix.max()
        bottom_right  = (ymin_xmax+pad,ymin-pad)
        bottom_left   = (ymin_xmin-pad,ymin-pad)
        for vertex in [bottom_right, bottom_left]:
            if vertex not in vertices: vertices.append(vertex)
     
        return np.array(vertices)
     
    @staticmethod
    def mask_polygon(mask,vertices):
        """
        Set the mask bit for pixels within the polygon
        defined by it's vertices.
        
        Parameters:
        mask     : The input pixel mask
        vertices : The polygon vertices
        Returns:
        out_mask : The output pixel mask with new mask bit set
        """
        out_mask = np.zeros(mask.shape,dtype=mask.dtype)
        nypix, nxpix = out_mask.shape
        yy,xx = np.indices((nypix,nxpix))
        points = np.vstack( (xx.flatten(),yy.flatten()) ).T
        path = matplotlib.path.Path(vertices)
        inside = path.contains_points(points).reshape(nypix,nxpix)
        ypix, xpix = np.nonzero(inside)
        out_mask[ypix,xpix] = True
        return out_mask

    @staticmethod
    def mask_between(mask,slope,inter1,inter2,factor=1.0):
        """
        Mask the pixels lying between two parallel lines. 
        Increase the width of the masked region by a factor.
     
        Parameters:
        mask     : The input mask array
        slope    : Slope of the streak
        inter1   : First edge y-intercept (can be top or bottom)
        inter1   : Second edge y-intercept (can be top or bottom)
        factor   : Factor to expand mask width
        Returns:
        out_mask : Mask array with pixels between the lines masked
        """
        out_mask = np.zeros(mask.shape,dtype=int)
        delta = np.abs(inter1 - inter2)*(factor - 1)/2.
        yy,xx = np.indices(mask.shape)
        xmin = 0; xmax = mask.shape[1] - 1 
        ymin = 0; ymax = mask.shape[0] - 1
        if (inter1 > inter2):
            upper = inter1; lower = inter2
        else:
            upper = inter2; lower = inter1
        #logger.debug("Mask: %s + %s > b > %s - %s"%(upper,delta,lower,delta))
        #print "# Mask: %s + %s > b > %s - %s"%(upper,delta,lower,delta)
        select = (yy < slope * xx + upper + delta) & \
            (yy > slope * xx +lower - delta)
     
        ypix,xpix = np.nonzero(select)
        out_mask[ypix,xpix] = True
        return out_mask

    @staticmethod
    def clip_mask(image,masked,mask,slope,nsig_clip=4):
        """
        Divide the mask into chunks. Loop through these
        chunks and identify those with fewer counts than
        expected (assuming uniform distribution). Starting 
        at the ends of the streak, remove underpopulated 
        chunks until a populated chunk is reached (thus 
        clipping the chunk from the sides).
     
        Parameters:
        image     : The input search image
        masked    : Pre-masked pixels in the search image
        mask      : Input streak mask
        slope     : Best-fit slope of the streak
        Returns:
        out_mask  : Clipped version of input streak mask
        """
     
        im = np.ma.array(image,mask=masked)
        ypix,xpix = np.nonzero(mask)
     
        # Expected number of counts per pixel in the streak reagion.
        # Should always be <= 1
        ncounts = im[ypix,xpix].sum()
        npix = im[ypix,xpix].count()
        density = float(ncounts)/npix
     
        # Rotate the streak to horizontal (hence the minus sign)
        alpha = -np.arctan(slope)
        xpixp = np.cos(alpha) * xpix - np.sin(alpha) * ypix
        xmin,xmax = xpixp.min(), xpixp.max()
        length = xmax - xmin
     
        # Divide the mask into equal-sized chunks each
        # expected to contain 'chunk_size' counts
        chunk_size = 100
        nchunks = ncounts/chunk_size
        bins = np.linspace(xmin,xmax,nchunks+1,endpoint=True)
        hist,edges,rev = self.idl_histogram(xpixp, bins=bins)
     
        # Clip all chunks that are low by 'nsig_clip'
        # Loop through and identify chunks for clipping
        keep = np.ones(len(hist),dtype=bool)
        tmp_mask = np.zeros(mask.shape)
        for i,n in enumerate(hist):
            idx = rev[rev[i]:rev[i+1]]
            chunk = im[ypix[idx],xpix[idx]]
            nobs = chunk.sum()
            nexp = chunk.count()*density
            sigma = np.sqrt(nexp)
     
            #logger.info("Chunk %i: nobs=%g, nexp=%g, sigma=%g"%(i,nobs,nexp,sigma))
            nsigma = (nexp - nobs)/sigma
            print "#  Chunk %i: nexp=%g, nobs=%g, nsig=%g"%(i,nexp,nobs,nsigma)
            #if nobs < (nexp - nsig_clip * sigma): 
            if nsigma > nsig_clip: 
               #logger.debug("   Sparse chunk %i: nsig > %g"%(i,nsig_clip))
               keep[i] = False
            tmp_mask[ypix[idx],xpix[idx]] = i+1
     
        # Only clip chunks on the ends of the streak.
        # Add one chunk to the end as a buffer.
        first = 0
        if keep.argmax() > 0 :
            first = keep.argmax() - 1
        last = len(keep)
        if keep[::-1].argmax() > 0:
            last = len(keep) - (keep[::-1].argmax() - 1)
     
        out_mask = np.zeros(mask.shape)
        for i,n in enumerate(hist):
            if not (first <= i < last): 
                #logger.debug("   Clipping chunk %i"%i)
                print "# Clipping chunk %i"%i
                continue
            idx = rev[rev[i]:rev[i+1]]
            out_mask[ypix[idx],xpix[idx]] = True
     
        return out_mask


    # Below are some static methods that are logically connected to 
    # the streak masking, but could easily be split off if there 
    # were another module using the hough transform or the SExtractor
    # background image

    @staticmethod
    def read_template(filename):
        """
        Read Hough transform template from a file.
     
        Parameters:
        infile    : File to read from.
        Returns:
        transform : Tuple containing (template,theta,rho)
        """
        data = np.load(filename)
        template = data['template'][0]
        theta    = data['theta'][0]
        rho      = data['rho'][0]
        return template, theta, rho

    @staticmethod
    def fit_template(data,template,debug=False):
        """
        Fits the normalization of a template array to the data array.
        Parameters:
        data    : 2D numpy array data
        template : 2D numpy array template
        Returns:
        norm     : Best-fit normalization of the template
        """
        flatdata = data.flatten()
        flattemp = template.flatten()
        idx = np.nonzero((flatdata!=0) & (flattemp!=0))
        flatdata = flatdata[idx]
        flattemp = flattemp[idx]
        fn = lambda x: ((flatdata - x*flattemp)**2).sum()
        return fmin(fn,x0=flatdata.mean()/flattemp.mean(),disp=debug)[0]

    @staticmethod
    def write_template(shape):
        """
        Write Hough transform to a numpy file described by the
        input image shape.
       
        Parameters:
        shape    : Shape of the input image array, i.e., (4096,2048)
        Returns:
        filename : Name of the template file created
        """
        template,theta,rho = Hough(np.ones(shape)).transform()
        data = np.recarray(1, dtype=[('template','u8',template.shape),
                                     ('theta','f8',theta.size),
                                     ('rho','f8',rho.size)])
        data['template'] = template
        data['theta'] = theta
        data['rho'] = rho
        filename = "hough_template_x%i_y%i.npy" % shape[::-1]
        np.save(filename,data)
        return filename


    @staticmethod
    def mmm(sky_vector, mxiter=30):
     
        """
        Robust sky fitting from IDL astronomy library.
        Used to estimate sky noise.
     
        Parameters:
        sky_vector : flat vector of sky-background level
        mxiter     : maximum iterations of the fitter
        Returns:
        sky_fit    : tuple of (skymod, sigma, skew)
        """
     
        #mxiter = 30
        minsky = 20
        nsky=sky_vector.size
     
        integer = 0  # fix this later
     
        if (nsky <= minsky) :
            raise Exception("Input vector must contain at least %s elements"%minsky)
            return np.array(-1.0),np.array(-1.0),np.array(-1.0)
     
        nlast=nsky-1  # hmmm
        sky = np.sort(sky_vector)
     
        skymid = 0.5*sky[(nsky-1)/2] + 0.5*sky[nsky/2]  # median
     
        cut1 = np.min([skymid-sky[0],sky[nsky-1]-skymid])
        cut2 = skymid + cut1
        cut1 = skymid - cut1
     
        #good=where((sky <= cut2) and (sky >= cut1))
        good=np.where(np.logical_and(sky <= cut2, sky >= cut1))[0]
        if good.size == 0 :
            raise Exception("No good sky found after cuts.")
            return np.array(-1),np.array(-1),np.array(-1)
     
        delta = sky[good] - skymid
        sum = delta.sum()
        sumsq = np.square(delta).sum()
     
        maximm = good.max()
        minimm = good.min() - 1
     
        skymed = 0.5*sky[(minimm+maximm+1)/2] + 0.5*sky[(minimm+maximm)/2+1]
        skymn = sum/(maximm-minimm)
        sigma = np.sqrt(sumsq/(maximm-minimm)-np.square(skymn))
        skymn = skymn+skymid  # add back median
     
        if (skymed < skymn) :
            skymod = 3.*skymed - 2.*skymn
        else :
            skymod = skymn
     
        # rejection and computation loop
        niter = 0
        clamp = 1
        old = 0
        redo = True
        while (redo) :
            niter=niter+1
            if (niter > mxiter) :
                raise Exception("Too many iterations")
                return np.array(-1.0),np.array(-1.0),np.array(-1.0)
     
            if ((maximm-minimm) < minsky) :
                raise Exception("Too few valid sky elements")
                return np.array(-1.0),np.array(-1.0),np.array(-1.0)
     
            r = np.log10((maximm-minimm).astype(np.float32))
            # What is this...?
            r = max([2.,(-0.1042*r+1.1695)*r + 0.8895])
     
            cut = r*sigma + 0.5*abs(skymn-skymod)
            # integer data?
            if (integer) :
                cut=cut.clip(min=1.5)
            cut1 = skymod - cut
            cut2 = skymod + cut
     
            redo = False
            newmin=minimm
            tst_min = sky[newmin+1] >= cut1
            done = (newmin == -1) and tst_min
            if (not done) :
                done = (sky[newmin.clip(min=0)] < cut1) and tst_min
            if (not done) :
                istep = 1 - 2*tst_min.astype(np.int32)
                while (not done) :
                    newmin = newmin + istep
                    done = (newmin == -1) or (newmin == nlast)
                    if (not done) :
                        done = (sky[newmin] <= cut1) and (sky[newmin+1] >= cut1)
                if tst_min :
                    delta = sky[newmin+1:minimm] - skymid
                else :
                    delta = sky[minimm+1:newmin] - skymid
                sum = sum - istep*delta.sum()
                sumsq = sumsq - istep*np.square(delta).sum()
                redo = True
                minimm=newmin
     
            newmax = maximm
            tst_max = sky[maximm] <= cut2
            done = (maximm == nlast) and tst_max
            if (not done) :
                mp1=maximm+1
                done = (tst_max) and (sky[mp1.clip(max=nlast)] >= cut2)
            if (not done) :
                istep = -1 + 2*tst_max.astype(np.int32)
                while (not done):
                    newmax = newmax + istep
                    done = (newmax == nlast) or (newmax == -1)
                    if (not done) :
                        done = (sky[newmax] <= cut2) and (sky[newmax+1] >= cut2)
                if (tst_max) :
                    delta = sky[maximm+1:newmax] - skymid
                else :
                    delta = sky[newmax+1:maximm] - skymid
                sum = sum + istep*delta.sum()
                sumsq = sumsq + istep*np.square(delta).sum()
                redo = True
                maximm=newmax
     
            nsky=maximm-minimm
            if (nsky < minsky) :
                raise Exception("Outlier rejection rejected too many sky elements")
                return np.array(-1.0),np.array(-1.0),np.array(-1.0)
     
            skymn = sum/nsky
            tmp=sumsq/nsky-np.square(skymn)
            sigma = np.sqrt(tmp.clip(min=0))
            skymn = skymn + skymid
     
            center = (minimm + 1 + maximm)/2.
            side = np.round(0.2*(maximm-minimm))/2. + 0.25
            j = np.round(center-side)
            k = np.round(center+side)
     
            # here is a place for readnoise correction...
            skymed = sky[j:k+1].sum()/(k-j+1)
     
            if (skymed < skymn) :
                dmod = 3.*skymed-2.*skymn-skymod
            else :
                dmod = skymn - skymod
     
            if dmod*old < 0 :
                clamp = 0.5*clamp
            skymod = skymod + clamp*dmod
            old=dmod
     
        skew = (skymn-skymod).astype(np.float32)/max([1.,sigma])
        nsky=maximm-minimm
     
        return skymod,sigma,skew

    # These are more general helper methods without any strong
    # association to StreakMasker. They could easily be
    # moved to some utility module.

    @staticmethod
    def read_bkg_image(bkgfile):
        """
        Simple function to read in the SExtractor background image.
        """
        # This is not very stable. It would be better to subclass the
        # DESImage class to take care of this...

        print "# Reading background file:%s" % bkgfile
     
        # Make sure the image exists
        if not os.path.exists(bkgfile):
            raise IOError ("File: %s does not exists... bye" % bkgfile)
     
        image_ext = None
        FITS = fitsio.FITS(bkgfile)
        for i in range(len(FITS)):
            h = FITS[i].read_header()
            if ('DES_EXT' in h.keys()) :
                extname = h['DES_EXT'].strip()
                if (extname == 'IMAGE') :
                    image_ext = i
                    
        if (image_ext is None):
            raise ValueError("Cannot find IMAGE extension via DES_EXT in %s" %(infile))
     
        bkg = FITS[image_ext].read()
        FITS.close()
        return bkg
     

    @staticmethod
    def idl_histogram(data,bins=None):
        """
        Bins data using numpy.histogram and calculates the
        reverse indices for the entries like IDL.
        
        Parameters:
        data  : data to pass to numpy.histogram
        bins  : bins to pass to numpy.histogram
        Returns:
        hist  : bin content output by numpy.histogram
        edges : edges output from numpy.histogram
        rev   : reverse indices of entries in each bin
     
        Using Reverse Indices:
            h,e,rev = histogram(data, bins=bins)
            for i in range(h.size):
                if rev[i] != rev[i+1]:
                    # data points were found in this bin, get their indices
                    indices = rev[ rev[i]:rev[i+1] ]
                    # do calculations with data[indices] ...
        """
        hist, edges = np.histogram(data, bins=bins)
        digi = np.digitize(data.flat,bins=np.unique(data)).argsort()
        rev = np.hstack( (len(edges), len(edges) + np.cumsum(hist), digi) )
        return hist,edges,rev

    @staticmethod
    def mean_filter(data,factor=4):
        y,x = data.shape
        binned = data.reshape(y//factor, factor, x//factor, factor)
        binned = binned.mean(axis=3).mean(axis=1)
        return binned

    @staticmethod
    def bin_pixels(data,factor=4):
        return StreakMasker.mean_filter(data,factor)

    @staticmethod
    def percentile_filter(data,factor=4,q=50):
        # Untested...
        y,x = data.shape
        binned = data.reshape(y//factor, factor, x//factor, factor)
        binned = np.percentile(np.percentile(binned,q=q,axis=3),q=q,axis=1)
        return binned

    @staticmethod
    def median_filter(data,factor=4):
        return percentile_filter(data,factor,q=50)
     
###########################
### Command-line Parser ###
###########################

def cmdline():
    """
    Command line parser and subcommand distribution.

    returns:
      parser : argparse.ArgumentParser object
    """

    formatter = argparse.RawDescriptionHelpFormatter
    description = "Command-line tool for DECam image masking.\nType 'immask <subcommand> --help' for help on a specific subcommand.\nAll subcommands take filename and outname arguments.\nTo run all masking algorithms, use 'all' subcommand."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=formatter)

    formatter = argparse.ArgumentDefaultsHelpFormatter
    general = argparse.ArgumentParser(formatter_class=formatter,add_help=False)
    general.add_argument("filename", help="FITS/FZ file to process.")
    general.add_argument("outname", help="Name of output FITS/FZ file.")
    general.add_argument('-v','--verbose', action="count", help="Output verbosity")
    general.add_argument('--compress', action="store_true", help="RICE/fpack compress output")
    general.add_argument('--outdir',default="immask_out", help="Path to QA output files")

    subparsers=parser.add_subparsers(dest='command',title='Available subcommands')

    # Cosmic-ray masking subcommmand
    title = 'cosmics'
    description = "Mask cosmic rays using the LSST python framework."
    cosmics = CosmicMasker.argparser(title,add_help=False)
    subparser = subparsers.add_parser(title,description=description,
                                      parents=[general,cosmics],
                                      formatter_class=formatter,
                                      help=description)
    subparser.set_defaults(func=run_cosmics)

    # Streak masking subcommand
    title = 'streaks'
    description = "Mask satellites, UFOs, etc. using Hough transform."
    streaks = StreakMasker.argparser(title,add_help=False)
    subparser = subparsers.add_parser(title,description=description,
                                      parents=[general,streaks],
                                      formatter_class=formatter,
                                      help=description)
    subparser.set_defaults(func=run_streaks)

    # Add star masks here...
    
    # Add bleed trails here...

    # Run all masking subcommands
    title = 'all'
    description = "Mask all image defects."
    subparser = subparsers.add_parser(title,description=description,
                                      parents=[general,cosmics,streaks],
                                      formatter_class=formatter,
                                      help=description)
    subparser.set_defaults(func=run_all)
    
    return parser

# ADW: This should be cleaned to reduce repetion
def run_cosmics(args):
    print '# Run %s...'%args.command
    kwargs = vars(args)
    image = DESImage(**kwargs)
    cosmics = CosmicMasker(image, **kwargs)
    image.write()

def run_streaks(args):
    print '# Run %s...'%args.command
    kwargs = vars(args)
    image = DESImage(**kwargs)
    streaks = StreakMasker(image, **kwargs)
    image.write()

def run_all(args):
    print '# Run %s...'%args.command
    kwargs = vars(args)
    image = DESImage(**kwargs)
    cosmics = CosmicMasker(image, **kwargs)
    streaks = StreakMasker(image, **kwargs)
    # In the future...
    #stars = StarMasker(image, **kwargs)
    #bleeds = BleedMasker(image, **kwargs)
    image.write()

# Format time
def elapsed_time(t1,verb=False):
    import time
    t2    = time.time()
    stime = "%dm %2.2fs" % ( int( (t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    if verb:
        print >>sys.stderr,"Elapsed time: %s" % stime
    return stime
     
if __name__ == "__main__":

    # Get the start time
    t0 = time.time()

    parser = cmdline()
    args = parser.parse_args()
    args.func(args)

    print >>sys.stderr,"# Time:%s" % elapsed_time(t0)
