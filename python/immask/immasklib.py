#!/usr/bin/env python
"""
$Id$
$Rev::                                  $:  # Revision of last commit.
$LastChangedBy::                        $:  # Author of last commit.
$LastChangedDate::                      $:  # Date of last commit.

Suite of fuctions and class for the DESDM immask python module.

TODO:
 - Iterate through finding streaks (remove each streak before finding next)?
 - Avoid filled corner cases by checking the fraction of area between streak 
   and corner that is above sky threshold.
 - Require the fill_fraction to be uniformly distributed
 - Check for ellipticity of the streak

@authors: Alex Drlica-Wagner  <kadrlica@fnal.gov>
@authors: Felipe Menanteau    <felipe@illinois.edu>
@authors: Eli Rykoff          <rykoff@slac.stanford.edu>
"""

import os
import sys
import fitsio 
import time
import copy
import argparse
from collections import OrderedDict as odict
import logging

import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import fmin
from scipy.spatial  import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.path

from despyastro import wcsutil
from pyhough.pyhough_lib import Hough
import despyfits.maskbits as MASKBITS
from despyfits.DESImage import DESImage
import despyfits.DESImage

######################
### Python Logging ###
######################
class ImmaskFormatter(logging.Formatter):
    """
    Class for overloading log formatting based on level.
    """
    FORMATS = {'DEFAULT'       : "IMMASK: %(message)s",
               logging.DEBUG   : "IMMASK::DEBUG: %(message)s",
               logging.WARNING : "IMMASK::WARNING: %(message)s",
               logging.ERROR   : "IMMASK::ERROR: %(message)s"}

    def format(self, record):
        self._fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)

def create_logger(level=logging.NOTSET):
    """
    Configure the logger with a custom formatter and set level.

    Parameters:
      level  : Level of the logger [int]
    Returns:
      logger
    """
    logger  = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ImmaskFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

#################################
### Generic Utility Functions ###
#################################

def int2uint(array):
    """ Utility function to convert from signed to unsigned integer.
    This function uses the fact that the unique character codes for
    unsigned integers are upper case of their signed equivalents.
    http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    """
    # Could add a few sanity checks here...
    char = array.dtype.char
    return array.view(char.upper())

def int2bit(value):
    """ Utility function to convert from integer to binary bit. """
    return int(np.log(value)/np.log(2))

    
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

    defaults = odict([])

    def __init__(self, image, outdir, **kwargs):
        """ Create and run the masker """
        self.image = image
        self.outdir = outdir

        if not os.path.exists(self.outdir):
            logging.info("Creating output directory  %s" % self.outdir)
            os.mkdir(self.outdir)

        self._parse(**kwargs)
        self.run()

    def run(self):
        """ Run the masking routines """
        pass

    def write(self):
        """ Write the masker QA files """
        pass

    def update_header(self):
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
        args = odict(cls.defaults)
        for key,value in args.items():
            group.add_argument('--%s'%key,dest=key,**value)
        return parser

class CosmicMasker(BaseMasker):

    ### Command line arguments for cosmic-ray masking
    defaults = odict([
        ['interpCR'  , dict(default=False, action="store_true", 
                            help="Interpolate CR in science image.")],
        ['nGrowCR'   , dict(default=None, type=int, 
                            help="Dilate CR mask by nGrowCR pixels.")],
        ['updateWeightCR', dict(default=False, action="store_true", 
                            help="Update the Weight Map for CR.")],
        ['fwhm'      , dict(default=None, type=float, 
                            help="Set FWHM [pixels] value that overrides the header")],
        ['minSigma'  , dict(default=6.0, type=float, 
                            help="CRs must be at least this many sigma above sky")],
        ['min_DN'    , dict(default=600., type=float, 
                            help="The sum of (bkg subtracted) counts in CRs must be greater than this")],
        ['fractionCR', dict(default=5., type=float, 
                            help="Fraction (in percent) of the image that can be flagged as CRs")],
        ])

    # Dictionary to transform mask from DES bits to LSST bits
    LSSTMaskPlaneDict = dict(
        # These bits get mapped to existing LSST definitions
        SAT       = int2bit(MASKBITS.BADPIX_SATURATE), # Used by findCosmicRays
        INTRP     = int2bit(MASKBITS.BADPIX_INTERP),   # Used by findCosmicRays
        CR        = int2bit(MASKBITS.BADPIX_CRAY),     # Used by findCosmicRays
        DETECTED  = int2bit(MASKBITS.BADPIX_STAR),     
        EDGE      = int2bit(MASKBITS.BADPIX_EDGE),     
        SUSPECT   = int2bit(MASKBITS.BADPIX_SUSPECT),  
        # These bit definitions are added
        BPM       = int2bit(MASKBITS.BADPIX_BPM),      # Used for interpolation
        BADAMP    = int2bit(MASKBITS.BADPIX_BADAMP),   # Used for interpolation
        # These bits are added but not used (could be removed)
        TRAIL     = int2bit(MASKBITS.BADPIX_TRAIL),    
        EDGEBLEED = int2bit(MASKBITS.BADPIX_EDGEBLEED),
        SSXTALK   = int2bit(MASKBITS.BADPIX_SSXTALK),  
        STREAK    = int2bit(MASKBITS.BADPIX_STREAK),   
        )

    def run(self):
        """
        Top-level function for CR masking
        """
        logging.info("Starting cosmic ray finder")
        # ADW: Should we put a catch statment here so that
        # CosmicMasker can't be run twice?
        
        # Make the individual calls
        self.make_lsst_image()
        self.find_cosmics()
        self.fix_cosmics()
        self.update_header()

    @staticmethod
    def get_FWHM(image):
       """ 
       Get the FHWM of the image in pixels. 
       ADW: This would be better in DESImage or somewhere more general
       """
       header = image.header
       # Read in the pixelscale
       try:
           pixscale = header['PIXSCAL1']
       except:
           pixscale = 0.263
           logging.warning("Could not read PIXSCAL1 keyword from science HDU\nWill default to %s" % pixscale)
     
       # Read in the FWHM
       try:
           fwhm        = header['FWHM']
           fwhm_arcsec = fwhm*pixscale
       except:
           fwhm = 5.5
           fwhm_arcsec = fwhm*pixscale
           logging.warning("Could not read FWHM keyword from science HDU\nWill use FWHM=%s pixels instead" % fwhm)
     
       # Make sure that FWHM is not too large
       if fwhm_arcsec > 1.2:
           logging.warning("Header FWHM value: %.2f[arcsec] / %.1f[pixels] is too high -- should not be trusted" % (fwhm_arcsec,fwhm))
 
       # Make sure that FWHM is not too small either
       if fwhm_arcsec < 2*pixscale:
           logging.warning("Header FWHM value: %.2f[arcsec] / %.1f[pixels] is too small -- forcing it to 2 pix" % (fwhm_arcsec,fwhm))
           fwhm = 2.0
 
       return fwhm

    def make_lsst_mask(self):
        """ Create a bit mask in the LSST framework. """
        import lsst.afw.image  as afwImage
        logging.info("Creating LSST mask plane.")

        # Create an LSST MaskU object from the DES mask
        MSK = afwImage.MaskU(int2uint(copy.copy(self.image.mask)))
        # Remap the DES bit definitions into the LSST system (this changes the bits)
        MSK.conformMaskPlanes(self.LSSTMaskPlaneDict)
        # The numpy array itself
        mska = MSK.getArray()

        # Compress several DECam mask bits into the LSST `BAD` bit. 
        # The LSST algorithm looks for `BAD` pixels by name
        # and ignores them (along with `INTRP` and `SAT`)
        BAD = np.sum([ MASKBITS.BADPIX_BPM,
                       MASKBITS.BADPIX_SATURATE,
                       MASKBITS.BADPIX_INTERP,
                       MASKBITS.BADPIX_BADAMP,
                       MASKBITS.BADPIX_EDGEBLEED,
                       ])
        mska[(self.image.mask & BAD) > 0] |= MSK.getPlaneBitMask('BAD')

        for k,v in odict(MSK.getMaskPlaneDict()).items():
            logging.debug("Plane '%s' -> %i"%(k,v))
     
        # Return the LSST MaskU object
        return MSK

    def make_lsst_image(self):
        """
        Create an LSST MaskedImage from the DES SCI, MSK, and WGT arrays.
        This is the object passed to the LSST findCosmicRays algorithm.
        """
        import lsst.afw.image  as afwImage
        logging.info("Making LSST MaskedImage object from SCI, MSK, and VAR")

        self.SCI = copy.copy(self.image.data)
        self.MSK = copy.copy(self.image.mask)
        self.WGT = copy.copy(self.image.weight)

        # Could be done with DESImage.get_variance (but doesn't fix bad values)
        self.WGT_fixed = np.where(self.WGT<=0, self.WGT.max()/1e6, self.WGT) # Fix values < 0
        self.VAR = 1/self.WGT_fixed

        # 1 - Create the science image
        self.sci = afwImage.ImageF(self.SCI)
        # 2 - Create the mask plane (must be unsigned int)
        self.msk = self.make_lsst_mask()
        # 3 - The variance image 
        self.var = afwImage.ImageF(self.VAR)
        # Into numpy-arrays to handle some numpy fast operations
        self.scia = self.sci.getArray()
        self.mska = self.msk.getArray()
        self.vara = self.var.getArray()
         
        # Make an LSST masked image (science, mask, and weight) 
        self.mi = afwImage.makeMaskedImage(self.sci, self.msk, self.var)
        return self.mi

    def find_cosmics(self,**kwargs):
        """
        Find the cosmic rays in the science image using
        lsst.meas.algorithms.findCosmicRays
        """
        # Load the the LSST modules here to avoid problems elsewhere in case they are not present
        import lsst.meas.algorithms as measAlg
        import lsst.ip.isr          as ipIsr
        import lsst.afw.math        as afwMath
        import lsst.pex.config      as pexConfig
         
        # Estimate the PSF of the science image -- we do this by default
        if not self.fwhm:
            logging.info("Attempting to get FWHM from the image header")
            self.fwhm  = self.get_FWHM(self.image)

        xsize = int(self.fwhm*9)
        sigma = self.fwhm/(2*np.sqrt(2*np.log(2)))
        psf = measAlg.DoubleGaussianPsf(xsize, xsize, sigma)
        psf_radius = psf.computeShape().getDeterminantRadius() 

        # Avoid too small psf_radius...
        if psf_radius < 2:
            logging.warning("psf_radius = %.2f is too small"%psf_radius)
            psf_radius = 2 # pixels(?)
            logging.warning("Setting psf_radius = %.2f"%psf_radius)

        # Recover FWHM from psf value
        fwhm  = 2*np.sqrt(2*np.log(2))*psf_radius # FM: This is the right expression -- corrected 
        logging.info("Using FWHM = %.2f for interpolation"%fwhm)

        # Interpolate some bad pixels before finding CR to avoid false detections
        maskName = 'BPM'
        logging.info("Interpolating '%s' pixel mask"%maskName)
        ipIsr.interpolateFromMask(self.mi, fwhm, growFootprints=0, maskName=maskName)
         
        # Simple background estimation to use on findCosmicRays.
        # Ignore pixels with the 'BAD' bit set.
        sctrl = afwMath.StatisticsControl()
        #sctrl.setAndMask(self.msk.getPlaneBitMask("BAD"))
        sctrl.setAndMask(self.msk.getPlaneBitMask(["BPM","SAT","BADAMP"]))
        background = afwMath.makeStatistics(self.mi, afwMath.MEANCLIP, sctrl).getValue()
        logging.info("Setting background at: %.1f [counts]"%background)

        # Tweak some of the configuration and call FindCosmicRays
        # More info in lsst/meas/algorithms/findCosmicRaysConfig
        crConfig             = measAlg.FindCosmicRaysConfig()
        crConfig.minSigma    = self.minSigma
        crConfig.min_DN      = self.min_DN

        # There are serveral undocumented crConfig parameters.
        # Need to read source code in CR.cc 

        # cond3_fac: Related to the number of std between the peak 
        # and the surrounding pixels (L750 in CR.cc). 
        # Lower numbers tend to help with long CR trails.
        crConfig.cond3_fac   = 1.5
        # cond3_fac2: ???
        #crConfig.cond3_fac2  = 0.6 # default
 
        nx,ny = self.image.data.shape
        crConfig.nCrPixelMax = int(nx*ny*(self.fractionCR/100.))

        # The LSST algorithm does not respect the 'keepCRs' flag
        # We need to explicitly remove interpolated pixels later on.
        if self.interpCR:
            # Interpolate the image plane
            crConfig.keepCRs  = False 
            logging.info("Will interpolate CRs on the output image")
        else:
            # Do not interpolate -- THIS DOESN'T WORK!!!!!
            crConfig.keepCRs  = True 
            logging.info("Will *not* interpolate CRs on the output image")

        # Now, we feed the lsst black box
        try:
            tLSST = time.time()
            self.crs = measAlg.findCosmicRays(self.mi, psf, background, pexConfig.makePolicy(crConfig))
            logging.info("Ran measAlg.findCosmicRays in: %s"%(elapsed_time(tLSST,verb=False)))
            self.NCRs = len(self.crs)
        except Exception as e:
            logging.warning("Masking CR failed for %s" % self.image.sourcefile)
            logging.warning("With message: \n%s" % str(e))
            self.crs  = False
            self.NCRs = -1
            return

        # Dilate the CR mask by nGrowCR pixels
        # WARNING: This also dilates any pre-existing pixels with the CR mask bit set
        if self.nGrowCR:
            logging.info("Dilating CR pixel mask by %s pixel(s)"%self.nGrowCR)
            defectList = ipIsr.getDefectListFromMask(self.mi,maskName='CR',growFootprints=self.nGrowCR)
            ipIsr.maskPixelsFromDefectList(self.mi,defectList,maskName='CR')
            if self.interpCR:
                logging.info("Interpolating dilated CR pixels")
                ipIsr.interpolateDefectList(self.mi,defectList,fwhm,fallbackValue=None)

    def fix_cosmics(self):
        """
        Set pixels in the output image (DESImage).
        This needs to be run after the CRs are detected (ie, after `find_cosmics`).
          IMAGE: If requested ('--interpCRs', the image plane will be interpolated
          MASK:  The mask plane will have the BADPIX_CRAY bit set.
          WEIGHT:If requested ('--updateWeightCR'), the weight plane will be set to zero.
        """
        # Extract the CR detected and INTRP pixels
        CRbit     = self.msk.getPlaneBitMask("CR")    # Gets the value for LSST CR
        INTERPbit = self.msk.getPlaneBitMask("INTRP") # Gets the value for LSST INTRP
         
        # Create a boolean selection CR-mask and INTRP from the requested bits
        masked_interp = (self.mska & INTERPbit) > 0 
        masked_cr     = (self.mska & CRbit) > 0
        NCRpix        = len(masked_cr[masked_cr==True])
        if self.crs:
            logging.info("Detected %d CRs containing %d pixels"%(self.NCRs,NCRpix))
        else:
            logging.info("No CRs detected due to failure.")
            return

        # 1. The Image
        # The LSST crConfig.keepCRs option is broken and always interpolates
        # This controls when the interpolated values make it back to the DESImage
        if self.interpCR:
            logging.info("Interpolating CR pixels in output image plane")
            self.image.data[masked_cr] = self.scia[masked_cr]
        # 2. The Mask
        logging.info("Setting CRAY bit in output mask plane")
        self.image.mask[masked_cr] |= MASKBITS.BADPIX_CRAY
        if self.interpCR:
            logging.info("Setting INTERP bit in output mask plane")
            self.image.mask[masked_interp] |= MASKBITS.BADPIX_INTERP
         
        # 3. The Weight
        if self.updateWeightCR:
            logging.info("Setting zero weight in output weight plane")
            # Set to zero weight CR-detected and interpolated pixels
            self.image.weight[masked_cr]         = 0 
  
    def update_header(self):
        """
        Add records to the header of the fits files
        """
        rec = dict(name='DESNCRAY', value=self.NCRs,
                   comment="Number of masked cosmic rays")
        self.image.header.add_record(rec)

    
class StreakMasker(BaseMasker):
    """
    Masking satellite, UFO, etc. streaks.
    """

    defaults = odict([
        ['bkgfile'     , dict(default=None,
                              help="Input background FITS file (fz/fits)")],
        ['draw'        , dict(action='store_true',
                              help="Draw diagnostic plots.")],
        ['draw_basename', dict(default=False,
                               help="File basename diagnostic plots.")],
        ['template_dir', dict(default="/dev/null",
                              help="Directory containing Hough template.")],
        ['galfile'     , dict(default=None,
                               help="Galaxy masking file")],
        # Image pre-processing
        ['bin_factor'   , dict(default=8, type=int, 
                               help="Binning factor to beat down sky noise")],
        # For detection, merging, and characterization
        ['nsig_sky'     , dict(default=1.5,type=float,
                               help="Threshold for sky noise") ],
        ['nsig_detect'  , dict(default=14.,type=float,
                               help="Threshold for Hough peak detection") ],
        ['nsig_merge'   , dict(default=8.,type=float,
                               help="Threshold for Hough peak merging ")],
        ['nsig_mask'    , dict(default=8.,type=float,
                               help="Threshold for Hough peak characterization")],
        # Quality cuts
        ['max_width'    , dict(default=150.,type=float,
                               help="Maximum streak width in (pix)")],
        ['max_angle'    , dict(default=15., type=float,
                               help="Maximum streak angular extent (deg)")],
        ['min_fill',      dict(default=0.25, type=float,
                               help="Minimum fraction of streak above threshold")],
        # For clipping partial streaks and diffraction spikes
        ['clip'         , dict(action="store_true",
                               help="Clip diffraction spikes (DEPRECATED).")],
        ['nsig_clip'    , dict(default=2.,  type=float,
                               help="Clip chunks that are more than 'nsig' underdense.")],
        ['clip_angle'   , dict(default=45., type=float,
                               help="Clip streaks close to the given abs. angle (deg).")],
        ['clip_range'   , dict(default=10., type=float,
                               help="Allowed clip angle range (deg); >90 to accept all angles")],
        # For masking
        ['mask_factor'  , dict(default=1.5, type=float, 
                               help="Factor to increase streak width for masking")],
        ['maskbits'     , dict(default=MASKBITS.BADPIX_STREAK-1,type=int,
                               help="Ignore these mask bits ")],
        ['setbit'       , dict(default=MASKBITS.BADPIX_STREAK,type=int,
                               help="New streak mask bit value")],
        ['maxmask'      , dict(default=1000,type=int,
                               help="Maximum number of streaks to mask [NOT IMPLEMENTED]")],
        # Update Weight maps
        ['updateWeightStreaks', dict(default=False, action="store_true", 
                                     help="Update the Weight Map for streaks.")],
        # Streak objects (probably don't need both)
        ['write_streaks', dict(action="store_true",
                               help="Write out streak objects")],
        ['streaksfile', dict(default=False,
                             help="Output streak objects FITS file")],
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

        tSTREAKS = time.time()
        logging.info("Starting streak finder")

        # ADW: Should we put a catch statment here so that
        # StreakMasker can't be run twice?

        # Read in the background image nd-array
        if self.bkgfile is not None:
            self.BKG    = self.read_bkg_image(self.bkgfile)
            #self.BKG    = DESImage(self.bkgfile).BKG
         
            # Make a background subtracted image
            self.subIm = self.image.data - self.BKG
        else:
            self.BKG   = None
            self.subIm = self.image.data

        # Create a boolean selection mask from the requested bits
        self.masked = ((self.image.mask & self.maskbits ) > 0)
        # Mask large bright galaxies
        if self.galfile:
            self.masked |= self.mask_galaxies(self.galfile)

        # Dilate bright star mask
        if (self.maskbits & MASKBITS.BADPIX_STAR) > 0:
            niter = 25
            struct = np.ones((3,3),dtype=bool)
            # Pixels with *only* star bit set
            starmask = (self.image.mask == MASKBITS.BADPIX_STAR)
            self.masked |= ndimage.binary_dilation(starmask,struct,iterations=niter)
         
        # Re-bin the image (if requested)
        if self.bin_factor > 1:
            logging.info("Binning image by a factor of %s..."% self.bin_factor)
            self.subIm  = self.bin_pixels(self.subIm,self.bin_factor)
            self.masked = np.ceil(self.bin_pixels(self.masked,self.bin_factor)).astype(bool)

        # Skip image if it is too heavily masked
        maskfrac = float(self.masked.sum())/self.masked.size
        if maskfrac > 0.85:
            logging.warn("Image is too heavily masked; skipping...")
            self.subIm.fill(1)
            self.masked.fill(False)

        # Calculate sky noise
        logging.info("Measuring sky noise...")
        usepix = np.where(~self.masked)
        try:
            self.sky_bkg, self.sky_err, self.sky_skew = self.mmm(self.subIm[usepix])
        except ValueError as e:
            logger.warn(str(e))
            self.sky_bkg, self.sky_err, self.sky_skew = 0.0, 0.0, 0.0

        # Added to deal with offsets in pca sky subtracted images
        # Might be good for SExtractor background as well...
        # if self.bkgfile is None:
        logging.info("Adjusting sky zero-level: %.2f"%self.sky_bkg)
        self.subIm -= self.sky_bkg
        
        # allow objects to be connected diagonally, create the structure
        self.structure = ndimage.generate_binary_structure(2,2)
         
        # The thresholded image to pass to the Hough transform
        self.searchIm = ((self.subIm > self.nsig_sky*self.sky_err) & ~self.masked)

        # Should do a maxarea cut here...
        logging.info("Performing Hough transform now...")
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
        logging.info("Maximum Hough value: %s" % self.trans.max())
         
        # Perform detection using structure and label
        self.detect_label,self.detect_nlabel = ndimage.label(self.trans > self.nsig_detect, structure=self.structure)
        if self.detect_nlabel:
            hist,edges,rev = self.idl_histogram(self.detect_label,bins=np.arange(self.detect_nlabel+2))
            maxarea = int(0.1*self.detect_label.size) # Get rid of background object
            good, = np.where((hist > 1) & (hist < maxarea))
            #good, = np.where( (hist > 1) )
            logging.info("Detected %i streaks above a threshold of %s"%(len(good),self.nsig_detect))
        else:
            logging.info("No streaks detected.")
            rev,good = [],[]
         
        # Detection statistics
        logging.info("Calculating detection statistics...")
        self.detect_objs = self.streak_statistics(self.trans, self.rho, self.theta, good, rev)
        self.detect_objs['BINNING'][:] = self.bin_factor
         
        logging.info("Performing quality cuts...")
        # Make some preliminary quality cuts
        cut = np.zeros(len(self.detect_objs),dtype=self.detect_objs['CUT'].dtype)
        # Perpendicular objects that look like a readout error...
        perp_cut = (np.abs(self.detect_objs['SLOPE']) > 150) & (self.detect_objs['WIDTH']*self.bin_factor <= 24)
        logging.info("\tCutting %i objects with: SLOPE > %s"%(perp_cut.sum(),150))
        cut |= perp_cut
        # Some additional cuts that could be useful...
        #cut |= ... # Central bias jump (x-dir)
        #cut |= ... # Central bias jump (y-dir)
        #cut |= ... # Edge brightness
        self.detect_objs['CUT'][np.nonzero(cut)] = 1
        logging.info("Found %i lines passing quality cuts" % (self.detect_objs['CUT']==0).sum())
         
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
            logging.info("Merging %i streaks above a threshold of %s"%(nmerged,self.nsig_merge))
        else:
            logging.info("No objects found for merging.")
            rev,unique = [],[]
         
        logging.info("Calculating merging statistics...")
        merge_objs = self.streak_statistics(self.trans, self.rho, self.theta, unique, rev)
        merge_objs['BINNING'][:] = self.bin_factor
         
        for i,obj in enumerate(merge_objs):
            slope  = obj['SLOPE']
            inter1 = obj['INTER1']
            inter2 = obj['INTER2']

            # Don't widen the mask for fill_frac cut
            tmp_mask=np.zeros(self.searchIm.shape,dtype=self.image.mask.dtype)
            tmp_mask=self.mask_between(tmp_mask,slope,inter1,inter2)
            fill_frac = self.fill_fraction(self.searchIm,self.masked,tmp_mask)
            obj['FILL_FRAC'] = fill_frac

        logging.info("Performing quality cuts...")
        # More quality cuts
        cut = np.zeros(len(merge_objs),dtype=merge_objs['CUT'].dtype)

        # Objects that are too wide...
        width_cut = (merge_objs['WIDTH'] > self.max_width/float(self.bin_factor))
        logging.info("\tCutting %i objects with: WIDTH > %s"%(width_cut.sum(),self.max_width))
        cut |= width_cut

        # Objects that span too large an opening angle -- too curvy
        theta_cut = merge_objs['DELTA_THETA'] > self.max_angle
        logging.info("\tCutting %i objects with: THETA > %s"%(theta_cut.sum(),self.max_angle))
        cut |= theta_cut

        # Objects that do not fill the mask
        fill_cut = merge_objs['FILL_FRAC'] < self.min_fill
        logging.info("\tCutting %i objects with: FILL_FRAC < %s"%(fill_cut.sum(),self.min_fill))
        cut |= fill_cut

        merge_objs['CUT'][np.nonzero(cut)] = 1
        logging.info("Found %i streaks passing quality cuts"%((merge_objs['CUT']==0).sum()))

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
                logging.info("Merging %i additional streaks"%(nmerged))
        else:
            logging.info("No objects found for masking.")
            rev,unique = [],[]
         
        logging.info("Calculating masking statistics...")
        self.mask_objs = self.streak_statistics(self.trans, self.rho, self.theta, unique, rev)
        self.mask_objs['BINNING'][:] = self.bin_factor
         
        # Create the mask from the streaks
        logging.info("Masking %i streaks"%(len(self.mask_objs)))
        streak_mask = np.zeros(self.image.mask.shape,dtype=self.image.mask.dtype)
        wcs = wcsutil.WCS(self.image.header)
        for i,obj in enumerate(self.mask_objs):
            slope  = obj['SLOPE']
            inter1 = obj['INTER1']
            inter2 = obj['INTER2']
            # Print intercept in original image coordinates
            logging.info("  %i   NSIG=%g, SLOPE=%g, INTER1=%g, INTER2=%g"%(i,obj['MAX'],slope,inter1*self.bin_factor,inter2*self.bin_factor))
          
            tmp_mask = np.zeros(self.searchIm.shape,dtype=self.image.mask.dtype)
            tmp_mask = self.mask_between(tmp_mask,slope,inter1,inter2,self.mask_factor)
          
            # ADW 2015/08/20: This was never really tested and should be deprecated
            clip = np.abs(np.abs(np.degrees(np.arctan(slope)))-self.clip_angle) < self.clip_range
            if self.clip and clip:
                logging.info("Examining for diffraction spike clipping...")
                tmp_mask = self.clip_mask(self.searchIm,self.masked,tmp_mask,slope,self.nsig_clip)
                
            fill_fraction = self.fill_fraction(self.searchIm,self.masked,tmp_mask)
            obj['FILL_FRAC'] = fill_fraction
          
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
        self.image.mask[ypix,xpix] = self.image.mask[ypix,xpix] | self.setbit 
        # and zero out the weight
        if self.updateWeightStreaks:
            logging.info("Setting Streak pixels in map weight to zero")
            self.image.weight[ypix,xpix] = 0
        # and update the header
        self.update_header()

        logging.info("Ran streaks in %s."%elapsed_time(tSTREAKS,verb=False))

        # Draw Plots
        if self.draw:
            self.draw_plots()

        # Write streak objects
        if self.write_streaks:
            self.write_streak_objects() 
        
    def write_streak_objects(self):
        if self.streaksfile:
            outname = self.streaksfile
        else:
            basename = os.path.basename(self.image.outname)
            outbase = basename.split('.fit')[0]+'_streaks.fits'
            outname = os.path.join(self.outdir,outbase)

        logging.info("Writing streak objects: %s" % (outname))
        fitsio.write(outname,self.mask_objs,clobber=True)
  
    def write_streak_mask(self):
        basename = os.path.basename(self.image.sourcefile)
        outbase = basename.split('.fit')[0]+'_mask.fits'

        outname = os.path.join(self.outdir,outbase)

        logging.info("Writing streak mask: %s" % (outname))
        
        maskonly = np.zeros(self.image.mask.shape,dtype=self.image.mask.dtype)
        test = np.where((self.image.mask & self.setbit) > 0)
        maskonly[test[0],test[1]] = self.setbit
   
        # Write mask as float32 so that "ds9 -mask" reads properly
        header = copy.copy(self.image.mask_hdr)
        header['BZERO'] = 0 
        #fitsio.write(outname,maskonly.astype('f4'),header=header,clobber=True)
        fitsio.write(outname,uint2int(maskonly),header=header,clobber=True)
  
    def draw_plots(self):
        """
        Draw the streak finder dignostic plot with matplotlib
        """
        import pylab as plt
        dpi = None # Can shrink disck i/o

        # Get the drawbase name -- from the output name so it is unique
        if self.draw_basename:
            basename = self.draw_basename
        else:
            basename = os.path.basename(self.image.outname)
        outbase  = os.path.splitext(os.path.splitext(basename)[0])[0] # double split for '.fits.fz' files
        drawbase = os.path.join(self.outdir,outbase)
         
        # The sky noise
        pngfile = "%s_sky.png" % drawbase
        logging.info("Drawing sky noise: %s " % pngfile)
        fig, axes = plt.subplots(1, 3, dpi=dpi)
        sigma     = [1,2,3]
        for ax,sig in zip(axes,sigma):
            label = ndimage.label(((self.subIm>sig*self.sky_err) & ~self.masked),structure=self.structure)
            ax.imshow(label[0]>0,origin='bottom'); 
            ax.set_title(r"Sky Noise ($%s\sigma$)"%sig); 
        plt.savefig(pngfile,dpi=dpi)
         
        # Draw the input mask
        pngfile = "%s_bits.png" % drawbase
        logging.info("Drawing input mask: %s " % pngfile)
        fig, axes = plt.subplots(1, 3, dpi=dpi)
        axes[0].imshow(self.subIm > self.nsig_sky*self.sky_err,origin='bottom')
        axes[0].set_title("Threshold Image")
        axes[1].imshow(self.masked,origin='bottom')
        axes[1].set_title("Input Mask")
        axes[2].imshow(self.searchIm,origin='bottom')
        axes[2].set_title(r"Search Image")
        plt.savefig(pngfile,dpi=dpi)
         
        # Draw the lines and masks
        pngfile = "%s_mask.png" % drawbase
        logging.info("Drawing streaks and masks: %s" % pngfile)
        ymax,xmax = self.searchIm.shape; x = np.arange(xmax)
        fig, axes = plt.subplots(1, 3, dpi=dpi)
        titles = ["Sky Noise","Streaks","Mask"]
        for ax,t in zip(axes,titles):
            ax.imshow(self.searchIm,origin='bottom'); ax.set_title(t)
        plt.sca(axes[1])
        for obj in self.mask_objs:
            y1 = obj['SLOPE']*x+obj['INTER1']; plt.plot(x,y1,'--w')
            y2 = obj['SLOPE']*x+obj['INTER2']; plt.plot(x,y2,'--w')
        if (self.image.mask & self.setbit).sum():
            plt.sca(axes[2]) 
            bin_mask = self.bin_pixels(self.image.mask & self.setbit,self.bin_factor).astype(bool)
            bin_mask = np.ma.array(bin_mask,mask=(bin_mask==0))
            cmap = matplotlib.cm.binary; cmap.set_bad('k',0)
            plt.imshow(bin_mask,origin='bottom',alpha=0.5,cmap=cmap)
        for ax in axes: ax.set_xlim(0,xmax); ax.set_ylim(0,ymax)
        plt.savefig(pngfile,dpi=dpi)
         
        #  Draw Hough transform at various thresholds
        pngfile = "%s_hough.png" % drawbase
        logging.info("Drawing Hough transform: %s" % pngfile)
        fig, axes = plt.subplots(2,2,figsize=(12,6),sharey='all',sharex='all',dpi=dpi)
        axes = axes.flatten()

        titles = ["Hough","Normalized",
                  r'Detect $(%g \sigma)$'%self.nsig_detect,
                  r"Mask $(%g \sigma)$"  %self.nsig_mask
              ]
        data   = [self.hough,self.trans,self.detect_label,self.mask_label]
        cmap = matplotlib.cm.jet; cmap.set_bad('w',0)
        extent = [self.theta.min(),self.theta.max(),self.rho.min(),self.rho.max()]
        kwargs = dict(extent=extent,aspect='auto',cmap=cmap,vmin=0)
        kw = [dict(kwargs),
              dict(kwargs,vmax=35),
              dict(kwargs),
              dict(kwargs),
              ]
        for ax,d,t,k in zip(axes,data,titles,kw):
            ax.imshow(np.ma.masked_array(d,mask=d==0),**k)
            ax.set_title(t)
        axes[0].set_ylabel(r'Distance (pix)'); 
        axes[2].set_ylabel(r'Distance (pix)'); 
        axes[2].set_xlabel(r'Angle (rad)')
        axes[3].set_xlabel(r'Angle (rad)')
        plt.savefig(pngfile,dpi=dpi)

    def get_template(self):
        """
        Read in the Hough template from file of binned image shape
        """
        template_file = "hough_template_x%i_y%i.npy" % self.searchIm.shape[::-1]
        filename = os.path.join(self.template_dir,template_file)
        if os.path.exists(filename):
            logging.info("Reading template from %s" % filename)
            self.template = read_template(filename)[0]
        else:
            logging.info("No template found --> calculating Hough normalization...")
            template_image = (np.ones(self.searchIm.shape,dtype=bool) & ~self.masked)
            self.template = Hough(template_image).transform()[0]

            # Increase template values for corner pixels
            corrections = odict([
                ('edge',    [self.mask_edges,256/self.bin_factor,1.0]),
                ('corner',  [self.mask_corners,256/self.bin_factor,5.0]),
                ('triangle',[self.mask_triangle_corners,320/self.bin_factor,8.0]),
                ])

            #import pylab as plt
            for n in ['corner']:
                fn,npix,weight = corrections[n]
                mask = fn(template_image.astype(bool),npix)
                temp = Hough(template_image*mask).transform()[0]
                #plt.figure();plt.imshow(self.template+weight*temp);plt.colorbar();plt.savefig('tmp_%s_temp.png'%n)

            self.template += weight * temp
            

    def update_header(self):
        """
        Add records to the header of the fits files
        """
        rec = dict(name='DESNSTRK', value=len(self.mask_objs),
                   comment="Number of streaks masked")
        self.image.header.add_record(rec)

    def mask_galaxies(self,filename,bit=MASKBITS.BADPIX_STAR,factor=1.7):
        """
        Mask the regions around bright galaxies from the Hyper-LEDA
        database.

        Parameters:
        filename : name of input csv file
        bit      : bit value to use for masking
        factor   : factor by which to expand stated diameter
        Returns:
        out_mask : The output pixel mask with new mask bit set
        """
        from matplotlib.path import Path
        from matplotlib.patches import Ellipse

        logging.info('Bright galaxy mask: %s'%filename)
        # Read the bright galaxy file
        names = ['name','ra','dec','logd25','logr25','pa','bt']
        data = np.genfromtxt(filename,delimiter=',',names=names,dtype=None)
        # Convert nan position angles to zero
        # Might also want to check that ellipticity is small?
        data['pa'][np.isnan(data['pa'])] = 0.
        out_mask = np.zeros(self.image.mask.shape,dtype=self.image.mask.dtype)

        ra,dec = data['ra'],data['dec']
        # Galaxy radius (from log10(D/0.1 arcmin) to deg)
        size = factor*(10**data['logd25']/2. * 0.1)/60. 
        delta_ra,delta_dec = size/np.cos(np.radians(dec)), size
        ramin,ramax = ra-delta_ra,ra+delta_ra
        decmin,decmax = dec-delta_dec,dec+delta_dec

        ccd_ramin=self.image.header['RACMIN']
        ccd_ramax=self.image.header['RACMAX']
        ccd_decmin=self.image.header['DECCMIN']
        ccd_decmax=self.image.header['DECCMAX']

        # Tying to be robust to RA = 0 boundary...
        if self.image.header['CROSSRA0'].startswith('Y'):
            logging.debug("CROSSRA0 = Y")
            ramax -= 360.
            ccd_ramax -= 360.

        sel  = (ramax > ccd_ramin)
        sel &= (ramin < ccd_ramax)
        sel &= (decmax > ccd_decmin)
        sel &= (decmin < ccd_decmax)

        if not np.sum(sel): return out_mask

        wcs = wcsutil.WCS(self.image.header)

        for d in data[sel]:
            logging.info("Masking galaxy: %s"%d['name'])
            xpix,ypix = wcs.sky2image(d['ra'],d['dec'])
            
            xpix = xpix.astype(int)
            ypix = ypix.astype(int)

            # major axis (diameter in pix)
            dmajor = int(factor*10**d['logd25'] * (0.1 * 60 / 0.263))
            # minor axis (diameter in pix)
            dminor = int(dmajor/10**d['logr25'])
            # position angle (degrees)
            theta = 90. - d['pa']

            #http://stackoverflow.com/a/47980627/4075
            ell = Ellipse((xpix,ypix),width=dminor,height=dmajor,angle=theta)
            path = Path(ell.get_verts())

            x = np.arange(-dmajor/2,dmajor/2)
            xx,yy = np.meshgrid(x,x)
            xidx = xpix + xx.flat
            yidx = ypix + yy.flat
            points = np.vstack([xidx,yidx]).T
            mask = path.contains_points(points)

            ymax, xmax = out_mask.shape
            mask &= ((xidx >= 0) & (xidx < xmax)) 
            mask &= ((yidx >= 0) & (yidx < ymax))
            logging.debug("Masking %i pixels"%(mask.sum()))
            
            out_mask[yidx[mask],xidx[mask]] = bit

        logging.info("Masked %i pixels"%(out_mask>0).sum())
        return out_mask

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
                                  ('DELTA_THETA','f4'),
                                  ('DELTA_RHO','f4'),
                                  ('SLOPE','f4'),
                                  ('INTER1','f4'),
                                  ('INTER2','f4'),
                                  ('WIDTH','f4'),
                                  ('BINNING','i4'),
                                  ('FILL_FRAC','f4'),
                                  ('CORNERS','f4',(4,2)),
                                  ('CORNERS_WCS','f8',(4,2)),
                                  ('CUT','i2'),])

        ZEROS = ['CUT','BINNING','CORNERS','CORNERS_WCS','FILL_FRAC']
        for col in ZEROS:
            objs[col][:] = 0
        
        shape = array.shape
        ncol = shape[1]

        for i in range(0,ngood):
            logging.debug("--- i=%i",i)
            # This code could use some cleanup...
            i1a=rev[rev[good[i]]:rev[good[i]+1]]
            xpix = i1a % ncol
            ypix = i1a / ncol
            pix  = zip(xpix,ypix)
            npix = len(xpix)
            objs[i]['LABEL'] = good[i]
            objs[i]['NPIX'] = npix
            logging.debug("LABEL\t= %i"%objs[i]['LABEL'])
            logging.debug("NPIX\t= %i"%objs[i]['NPIX'])
     
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
            logging.debug("MAX\t= %i"%objs[i]['MAX'])
            logging.debug("MAX_X0, MAX_Y0\t= %i, %i"%(objs[i]['MAX_X0'],objs[i]['MAX_Y0']))
     
            # Full extent of the island
            objs[i]['XMIN'] = xpix.min()
            objs[i]['XMAX'] = xpix.max()
            logging.debug("XMIN, XMAX\t= %i, %i"%(objs[i]['XMIN'],objs[i]['XMAX']))
            objs[i]['DELTA_THETA'] = np.degrees(theta[objs[i]['XMAX']]-theta[objs[i]['XMIN']])
            logging.debug("DELTA_THETA\t= %.0f"%(objs[i]['DELTA_THETA']))

            objs[i]['YMIN'] = ypix.min()
            objs[i]['YMAX'] = ypix.max()
            logging.debug("YMIN, YMAX\t= %i, %i"%(objs[i]['YMIN'],objs[i]['YMAX']))
            objs[i]['DELTA_RHO'] = rho[objs[i]['YMAX']]-theta[objs[i]['YMIN']]
            logging.debug("DELTA_RHO\t= %.0f"%(objs[i]['DELTA_RHO']))
     
            # Find pixel closest to centroid (careful of x-y confusion)
            centroid = ndimage.center_of_mass(island)
            tree = cKDTree(pix)
            dist, idx = tree.query([centroid[1],centroid[0]])
            xcent,ycent = pix[idx]
            objs[i]['CEN_X0'] = xcent
            objs[i]['CEN_Y0'] = ycent
            logging.debug("CEN_X0, CEN_Y0\t= %i, %i"%(objs[i]['CEN_X0'],objs[i]['CEN_Y0']))
            extent = np.nonzero(island[:,xcent])[0]
            ycent1,ycent2 = extent.min(),extent.max()+1
            objs[i]['CEN_Y1'] = ycent1
            objs[i]['CEN_Y2'] = ycent2
            logging.debug("CEN_Y1, CEN_Y2\t= %i, %i"%(objs[i]['CEN_Y1'],objs[i]['CEN_Y2']))
     
            # Calculate the slope and intercept
            theta0 = theta[xcent]
            rho1,rho2 = rho[ycent1],rho[ycent2]
            slope = np.nan_to_num( -np.cos(theta0)/np.sin(theta0) )
            inter1 = np.nan_to_num(rho1/np.sin(theta0))
            inter2 = np.nan_to_num(rho2/np.sin(theta0))
            objs[i]['SLOPE']  = slope
            objs[i]['INTER1'] = inter1
            objs[i]['INTER2'] = inter2
            logging.debug("SLOPE\t= %.3g"%objs[i]['SLOPE'])
            logging.debug("INTER1\t= %.3g"%objs[i]['INTER1'])
            logging.debug("INTER2\t= %.3g"%objs[i]['INTER2'])
     
            objs[i]['WIDTH']  = np.abs(rho1 - rho2)
            logging.debug("WIDTH\t= %.3g"%objs[i]['WIDTH'] )
            
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
        logging.debug("Mask: %s + %s > b > %s - %s"%(upper,delta,lower,delta))
        select = (yy < slope * xx + upper + delta) & \
            (yy > slope * xx +lower - delta)
     
        ypix,xpix = np.nonzero(select)
        out_mask[ypix,xpix] = True
        return out_mask

    @staticmethod
    def mask_edges(mask, npix=32):
        out_mask = np.zeros(mask.shape,dtype=mask.dtype)
        out_mask[npix:-npix,npix:-npix] = True
        return out_mask

    @staticmethod
    def mask_corners(mask, npix=32):
        out_mask = np.ones(mask.shape,dtype=mask.dtype)
        out_mask[npix:-npix,:] = False
        out_mask[:,npix:-npix] = False
        return out_mask

    @staticmethod
    def mask_triangle_corners(mask, npix=32):
        out_mask = np.zeros(mask.shape,dtype=mask.dtype)
        n,m = mask.shape
        idx = np.triu_indices(n,m-npix,m)
        yy,xx = idx
        for i,j in [(1,1),(-1,1),(1,-1),(-1,-1)]:
            out_mask[i*yy,j*xx] = True
        return out_mask

    @staticmethod
    def fill_fraction(image,masked,mask):
        im = np.ma.array(image,mask=masked)
        ypix,xpix = np.nonzero(mask)
     
        # Number of counts per pixel in the streak region.
        # Should always be <= 1
        ncounts = im[ypix,xpix].sum()
        npix = im[ypix,xpix].count()
        if npix == 0: return 0
        else: return float(ncounts)/npix
       

    @staticmethod
    def length(mask,slope):
        ypix,xpix = np.nonzero(mask)
     
        # Rotate the streak to horizontal (hence the minus sign)
        alpha = -np.arctan(slope)
        xpixp = np.cos(alpha) * xpix - np.sin(alpha) * ypix
        xmin,xmax = xpixp.min(), xpixp.max()
        length = xmax - xmin
        
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
     
        ## Expected number of counts per pixel in the streak reagion.
        ## Should always be <= 1
        density = fill_fraction(image,masked,mask)
        
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
     
            #logging.info("Chunk %i: nobs=%g, nexp=%g, sigma=%g"%(i,nobs,nexp,sigma))
            nsigma = (nexp - nobs)/sigma
            logging.info("  Chunk %i: nexp=%g, nobs=%g, nsig=%g"%(i,nexp,nobs,nsigma))
            #if nobs < (nexp - nsig_clip * sigma): 
            if nsigma > nsig_clip: 
               #logging.debug("   Sparse chunk %i: nsig > %g"%(i,nsig_clip))
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
                logging.debug("   Clipping chunk %i"%i)
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
    def mmm(sky_vector, mxiter=50, atol=1e-10):
     
        """
        Robust sky fitting from DAOPHOT via IDL astronomy library.
        Used to estimate sky noise.
     
        Parameters:
        sky_vector : flat vector of sky-background level
        mxiter     : maximum iterations of the fitter
        atol       : absolute tolerance on value of skymod
        Returns:
        sky_fit    : tuple of 
                     skymod: Estimated mode of sky
                     sigma:  Standard deviation of the peak in thesky
                     skew:   Skewness of the peak
        """
     
        minsky = 20
        nsky=sky_vector.size
     
        integer = 0  # fix this later
     
        if (nsky <= minsky) :
            raise ValueError("Input vector must contain at least %s elements"%minsky)
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
            raise ValueError("No good sky found after cuts.")
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
                raise RuntimeError("Too many iterations")
                return np.array(-1.0),np.array(-1.0),np.array(-1.0)
     
            if ((maximm-minimm) < minsky) :
                raise RuntimeError("Too few valid sky elements")
                return np.array(-1.0),np.array(-1.0),np.array(-1.0)
     
            # Compute Chauvenet rejection criterion.
            r = np.log10((maximm-minimm).astype(np.float32))
            # What is this...?
            r = max([2.,(-0.1042*r+1.1695)*r + 0.8895])
     
            # Compute rejection limits (symmetric about the current mode).
            cut = r*sigma + 0.5*abs(skymn-skymod)
            # integer data?
            if (integer) :
                cut=cut.clip(min=1.5)
            cut1 = skymod - cut
            cut2 = skymod + cut
     
            # Recompute mean and sigma by adding and/or subtracting sky values
            # at both ends of the interval of acceptable values.
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
                raise RuntimeError("Outlier rejection rejected too many sky elements")
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
     
            # prevent oscillations by clamping down if sky adjustments are changing sign
            if dmod*old < 0 :
                clamp = 0.5*clamp
            skymod = skymod + clamp*dmod
            old=dmod

            # Check if change to skymod less than tolerance
            if abs(clamp*dmod) < atol:
                redo = False

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

        logging.info("Reading background file:%s" % bkgfile)
     
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
    general.add_argument("outname", help="Name of output FITS file.")
    general.add_argument('-v','--verbose', action="count", help="Output verbosity")
    #general.add_argument('--compress', action="store_true", help="RICE/fpack compress output")
    general.add_argument('--outdir',default="qa", help="Path to QA output files")

    # Indirect method for despyfits
    general.add_argument('--indirect_write',default=False,action="store_true",
                         help="Use indirect write mechanism in despyfits")
    general.add_argument('--indirect_write_prefix',default=None,action="store",
                         help="Set the prefix for the indirect write mechanism in despyfits")

    subparsers=parser.add_subparsers(dest='command',title='Available subcommands')

    # Cosmic-ray masking subcommmand
    title = 'cosmics'
    description = "Mask cosmic rays using the LSST python framework."
    cosmics = CosmicMasker.argparser(title,add_help=False)
    subparser = subparsers.add_parser(title,description=description,
                                      parents=[general,cosmics],
                                      formatter_class=formatter,
                                      help=description)
    #subparser.set_defaults(func=run_cosmics)

    # Streak masking subcommand
    title = 'streaks'
    description = "Mask satellites, UFOs, etc. using Hough transform."
    streaks = StreakMasker.argparser(title,add_help=False)
    subparser = subparsers.add_parser(title,description=description,
                                      parents=[general,streaks],
                                      formatter_class=formatter,
                                      help=description)
    #subparser.set_defaults(func=run_streaks)
    
    # Add star masks here...
    
    # Add bleed trails here...

    # Run all masking subcommands
    title = 'all'
    description = "Mask all image defects."
    subparser = subparsers.add_parser(title,description=description,
                                      parents=[general,cosmics,streaks],
                                      formatter_class=formatter,
                                      help=description)
    #subparser.set_defaults(func=run_all)
    parser.set_defaults(func=run)

    return parser

def run(args):
    create_logger(logging.DEBUG if args.verbose else logging.INFO)

    commands = [args.command]
    kwargs = vars(args)
    image = DESImage.load(args.filename)
    # passing outname into the class
    image.outname = args.outname 
    if 'all' in commands: commands = ['cosmics','streaks']
    for command in commands:
        logging.info('Running %s...'%command)
        start = time.time()
        if command == 'cosmics':
            cosmics = CosmicMasker(image, **kwargs)
        if command == 'streaks':
            streaks = StreakMasker(image, **kwargs)
        ## In the future...
        #if command == 'stars':
        #    stars = StarMasker(image, **kwargs)
        #if command == 'bleeds':
        #    bleeds = BleedMasker(image, **kwargs)

        logging.info("Ran %s in %s."%(command,elapsed_time(start,verb=False)))


    # Set up the writing method for despyfits -- only setup if defined,
    # otherwise will use the enviromental variable $DESPYFITS_USE_INDIRECT_WRITE
    if args.indirect_write:
        despyfits.DESImage.use_indirect_write    = args.indirect_write
    if args.indirect_write_prefix:
        despyfits.DESImage.indirect_write_prefix = args.indirect_write_prefix 

    # Update the header information
    timenow   = time.asctime() # Time stamp for new fits files
    rec = dict(name='DESIMMSK',value=timenow,comment="DESDM immask image")
    image.header.add_record(rec)

    # Flag for turning off writing
    if True:
        image.save(args.outname)
        logging.info("Wrote: %s" % args.outname)
    else:
        logging.warning("NO FILE SAVED!")
        
def elapsed_time(t1,verb=False):
    """ Format time output """
    import time
    t2    = time.time()
    stime = "%dm %2.2fs" % ( int( (t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    if verb:
        print >>sys.stderr,"Elapsed time: %s" % stime
    return stime

def extract_filename(filename):
    """ Safe extraction of filename """
    if filename[0] == "!": filename=filename[1:]
    filename = os.path.expandvars(filename)
    filename = os.path.expanduser(filename)
    return filename
    
if __name__ == "__main__":

    # Get the start time
    t0 = time.time()

    parser = cmdline()
    args = parser.parse_args()
    args.func(args)

    #print >>sys.stderr,"# Time:%s" % elapsed_time(t0)
    logging.info("Time: %s" % elapsed_time(t0))
