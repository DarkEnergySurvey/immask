DESDM image masking module
--------------------------
Authors: Alex Drlica-Wagner, Eli Rykoff, Felipe Menanteau

Mask satellite streaks and cosmic rays in DECam images

Requires:

 numpy:      http://www.numpy.org/ version 1.6.2 or above
 scipy:      http://scipy.org/ version 0.12.0 or above
 matplotlib: http://matplotlib.org/ version 1.2.0 or above
 fitsio:     https://github.com/esheldon/fitsio version 0.9.4 or above (RICE compression enabled)
 pyhough:    https://github.com/erykoff/pyhough  
 despyastro: For transplanted wcsutil from Erin Sheldon

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



