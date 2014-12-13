DESDM image masking module
---------------

Mask satellite streaks and cosmic rays 

Requires:

 numpy:      http://www.numpy.org/ version 1.6.2 or above
 scipy:      http://scipy.org/ version 0.12.0 or above
 matplotlib: http://matplotlib.org/ version 1.2.0 or above
 fitsio:     https://github.com/esheldon/fitsio version 0.9.4 or above (RICE compression enabled)
 pyhough:    https://github.com/erykoff/pyhough  
 despyutils: For transplanted wcsutil from Erin Sheldon

To install locally in $HOME:

 %> python setup.py install --home=$HOME/Python
 
To run streaks:
 %> bin/immask streaks DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits  --bkgfile DECam_00226647_47_bkg.fits.fz 
To run cosmic rays:
 %> bin/immask cosmics DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits 
To run everything:
 %> bin/immask all     DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits  --bkgfile DECam_00226647_47_bkg.fits.fz 

To generate an fpacked output image:

 %> bin/immask all DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits.fz --bkgfile DECam_00226647_47_bkg.fits.fz --compress

To see options:

 %> bin/immask <subcommand> --help


To run on NCSA cosmology cluster via eups:

unsetenv EUPS_DIR
unsetenv EUPS_PATH

source /des002/apps/RHEL6/dist/eups/desdm_eups_setup.csh

setup cfitsio_shared        3.360+3
setup immask                0.2.2+0
setup lsstset               7.1.0+4  


To Do:
---------------
- Load bad pixel bit mask directly from imsupport
- More extensive method documentation.
- Add decorator for masker class containing defaults.
- Re-factor 'immasklib' into 'image' and 'masker' files.
- Clean up conflict between '--compress' and '.fz' file endings.
