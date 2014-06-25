DESDM image masking module

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
 
To run:

 %> bin/immask DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits  --bkgfile DECam_00226647_47_bkg.fits.fz 
 %> bin/immask DECam_00226647_47.fits    clean_DECam_00226647_47.fits  --bkgfile DECam_00226647_47_bkg.fits

To generate an fpacked output image:

 %> bin/immask DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits.fz --bkgfile DECam_00226647_47_bkg.fits.fz --compress

to see all options:

 %> bin/immask --help


To run in cosmology cluster via eups

unsetenv EUPS_DIR
unsetenv EUPS_PATH

source /des002/apps/RHEL6/dist/eups/desdm_eups_setup.csh

setup cfitsio_shared        3.360+3
setup immask                0.2.2+0
setup lsstset               7.1.0+4  
