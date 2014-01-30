def read_bkg_image(bkgfile):

    """
    Simple function to read in the SExtractor background image using
    DES extension keywords.
    """

    print "# Reading background file:%s" % bkgfile

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
    print "# Done..."
    FITS.close()
    return bkg

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

def fit_array_template(array,template,debug=False):
    """
    Fits the normalization of a template array to the data array.
    Parameters:
    array    : 2D numpy array data
    template : 2D numpy array template
    Returns:
    norm     : Best-fit normalization of the template
    """
    flatarray = array.flatten()
    flattemp = template.flatten()
    idx = np.nonzero((flatarray!=0) & (flattemp!=0))
    flatarray = flatarray[idx]
    flattemp = flattemp[idx]
    fn = lambda x: ((flatarray - x*flattemp)**2).sum()
    if debug:
        disp = True 
    else: 
        disp = False 
    return fmin(fn,x0=flatarray.mean()/flattemp.mean(),disp=disp)[0]

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
                              ('CUT','i2'),])
    objs['CUT'][:] = 0
    objs['BINNING'][:] = 0

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

def mask_between(mask,slope,inter1,inter2,factor=1.0):
    """
    slope and intercept seems to be the lower edge (ymin)
    of the mask (not sure why this is).
    """
    delta = np.abs(inter1 - inter2)*(factor - 1)/2.
    yy,xx = np.indices(mask.shape)
    if (inter1 > inter2):
        #logger.debug("Mask: %s + %s > b > %s - %s"%(inter1,delta,inter2,delta))
        select = (yy < slope * xx + inter1 + delta) & \
            (yy > slope * xx + inter2 - delta)
    else:
        #logger.debug("Mask: %s + %s > b > %s - %s"%(inter2,delta,inter1,delta))
        select = (yy < slope * xx + inter2 + delta) & \
            (yy > slope * xx + inter1 - delta)
    ypix,xpix = np.nonzero(select)
    mask[ypix,xpix] = True

def mean_filter(data,factor=4):
    y,x = data.shape
    binned = data.reshape(y//factor, factor, x//factor, factor)
    binned = binned.mean(axis=3).mean(axis=1)
    return binned

def bin_pixels(data,factor=4):
    return mean_filter(data,factor)

def percentile_filter(data,factor=4,q=50):
    # Untested...
    y,x = data.shape
    binned = data.reshape(y//factor, factor, x//factor, factor)
    binned = np.percentile(np.percentile(binned,q=q,axis=3),q=q,axis=1)
    return binned

def median_filter(data,factor=4):
    return percentile_filter(data,factor,q=50)


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

