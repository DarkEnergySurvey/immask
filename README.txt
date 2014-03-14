DESDM Image Masking module

Masks Satellite streaks and Cosmic Rays 

Requires:

 nympy  1.6.2 or above
 scipy  0.12.0 or above
 fitsio:  https://github.com/esheldon/fitsio version 0.9.4 or above (RICE compression enabled)
 pyhough: https://github.com/erykoff/pyhough  

to run:

 %>bin/immask DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits  --bkgfile DECam_00226647_47_bkg.fits.fz 
 %>bin/immask DECam_00226647_47.fits    clean_DECam_00226647_47.fits  --bkgfile DECam_00226647_47_bkg.fits.fz 

to generate an fpacked output image:

 %>bin/immask DECam_00226647_47.fits.fz clean_DECam_00226647_47.fits.fz --bkgfile DECam_00226647_47_bkg.fits.fz --compress

to see all options:

 %>bin/immask --help
