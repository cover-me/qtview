'''
A collection of methods for data operation.
Methods with names start with '_': auxiliary operations.
The rest: filters who modify the data directly! No returned values.
Data shape: Any shape. Usually, it looks like: [x,y,w,...] and each of x,y,w,... is a 2d matrix.
'''

import numpy as np

def _create_kernel(x_dev, y_dev, cutoff, distr):
    distributions = {
        'gaussian': lambda r: np.exp(-(r**2) / 2.0),
        'exponential': lambda r: np.exp(-abs(r) * np.sqrt(2.0)),
        'lorentzian': lambda r: 1.0 / (r**2+1.0),
        'thermal': lambda r: np.exp(r) / (1 * (1+np.exp(r))**2)
    }
    func = distributions[distr]

    hx = int(np.floor((x_dev * cutoff) / 2.0))
    hy = int(np.floor((y_dev * cutoff) / 2.0))

    x = np.zeros(1) if x_dev==0 else np.linspace(-hx, hx, hx * 2 + 1) / x_dev
    y = np.zeros(1) if y_dev==0 else np.linspace(-hy, hy, hy * 2 + 1) / y_dev

    xv, yv = np.meshgrid(x, y)

    kernel = func(np.sqrt(xv**2+yv**2))
    kernel /= np.sum(kernel)

    return kernel

def _get_quad(x):
    '''
    Calculate patch (quadrilateral) corners for 2d plot.
    pcolormesh() need this method for non-evenly spaced XY coordinates.
    imshow() don't use this. It assumes XY coordinates are evenly spaced.
    See: https://cover-me.github.io/2019/02/17/Save-2d-data-as-a-figure.html, https://cover-me.github.io/2019/04/04/Save-2d-data-as-a-figure-II.html
    '''
    s1,s2 = x.shape
    x_pad = np.full((s1+2,s2+2), np.nan)
    x_pad[1:-1,1:-1] = x
    
    if s1>1:
        b1, b2 = x_pad[1], x_pad[2]
        t1, t2 = x_pad[-2], x_pad[-3]
        x_pad[0] = 2*b1 - b2
        x_pad[-1] = 2*t1 - t2
    else:
        x_pad[0] = x_pad[1] - 1
        x_pad[-1] = x_pad[1] + 1
        


    if s2>1: 
        l1, l2 = x_pad[:,1], x_pad[:,2]
        r1, r2 = x_pad[:,-2], x_pad[:,-3]
        x_pad[:,0] = 2*l1 - l2
        x_pad[:,-1] = 2*r1 - r2
    else:
        x_pad[:,0] = x_pad[:,1] - 1
        x_pad[:,-1] = x_pad[:,1] + 1
    
    x_quad = (x_pad[:-1,:-1]+x_pad[:-1,1:]+x_pad[1:,:-1]+x_pad[1:,1:])/4.
    
    return x_quad

# filters

def yderiv(d):
    '''
    y derivative, slightly different from qtplot
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
    '''
    y = d[1]
    z = d[2]
    dzdy0 = (z[1]-z[0])/(y[1]-y[0])
    dzdy1 = (z[-2]-z[-1])/(y[-2]-y[-1])
    z[1:-1] = (z[2:] - z[:-2])/(y[2:] - y[:-2])
    z[0] = dzdy0
    z[-1] = dzdy1
    return d


def xderiv(d):
    '''
    x derivative, slightly different from qtplot
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
    '''
    x = d[0]
    z = d[2]
    dzdx0 = (z[:,1]-z[:,0])/(x[:,1]-x[:,0])
    dzdx1 = (z[:,-2]-z[:,-1])/(x[:,-2]-x[:,-1])
    z[:,1:-1] = (z[:,2:] - z[:,:-2])/(x[:,2:] - x[:,:-2])
    z[:,0] = dzdx0
    z[:,-1] = dzdx1
    return d


def yintegrate(d):
    '''
    y integration
    '''
    y = d[1]
    dy = abs(y[1,0]-y[0,0])
    d[2] = scipy.integrate.cumulative_trapezoid(d[2],axis=0,dx=dy,initial=0)
    return d


def lowpass(d, x_width=0.5, y_height=0.5, method='gaussian'):
    """Perform a low-pass filter."""
    z = d[2]
    kernel = Operation._create_kernel(x_width, y_height, 7, method)
    z[:] = scipy.ndimage.filters.convolve(z, kernel)
    return d


def scale(d,x=[1,1,1]):
    '''
    Scale i-th term in d by x[i]
    '''
    for i, j in enumerate(x):
        d[i] *= j
    return d


def offset(d,x=[0,0,0]):
    '''
    Offset i-th term in d by x[i]
    '''
    for i, j in enumerate(x):
        d[i] += j
    return d


def g_in_g2(d, rin):
    """z = z/(1-(z*Rin))/7.74809e-5. z: conductance in unit 'S', R in unit 'ohm' (SI units)"""
    G2 = 7.74809e-5#ohm^-1, 2e^2/h
    d[2] = d[2]/(1-(d[2]*rin))/G2
    return d


def xy_limit(d,xmin=None,xmax=None,ymin=None,ymax=None):
    '''Crop data with xmin,xmax,ymin,ymax'''
    x = d[0]
    y = d[1]
    if not all([i is None for i in [xmin,xmax,ymin,ymax]]):
        x1 = 0 if xmin is None else np.searchsorted(x[0],xmin)
        x2 = -1 if xmax is None else np.searchsorted(x[0],xmax,'right')-1
        y1 = 0 if ymin is None else np.searchsorted(y[:,0],ymin)
        y2 = -1 if ymax is None else np.searchsorted(y[:,0],ymax,'right')-1
        return Operation.crop(d,x1,x2,y1,y2)
    else:
        return d


def crop(d, left=0, right=-1, bottom=0, top=-1):
    """Crop data by indexes. First and last values included"""
    right = d[0].shape[1] + right + 1 if right < 0 else right + 1
    top = d[0].shape[0] + top + 1 if top < 0 else top + 1
    if (0 <= left < right <= d[0].shape[1] 
        and 0 <= bottom < top <= d[0].shape[0]):
        return d[:,bottom:top,left:right]
    else:
        raise ValueError('Invalid crop parameters: (%s,%s,%s,%s)'%(left,right,bottom,top))


def autoflip(d):
    '''
    Make the order of elements in x and y good for imshow() and filters
    '''
    x = d[0]
    y = d[1]
    xa = abs(x[0,0]-x[0,-1])
    xb = abs(x[0,0]-x[-1,0])
    ya = abs(y[0,0]-y[0,-1])
    yb = abs(y[0,0]-y[-1,0])
    # Make sure x changes faster inside rows (xa>xb), y changes faster inside columns (yb>ya)
    if (xa<xb and yb<ya) or (xa>xb and yb<ya and yb/ya<xb/xa) or (xa<xb and yb>ya and ya/yb>xa/xb):
        d = np.transpose(d, (0, 2, 1))# swap axis 1 and 2
    # Make coordinates are in overall increasing order
    x = d[0]#note: x y won't unpdate after d changes. There maybe nan in last lines of x and y.
    y = d[1]
    if x[0,0]>x[0,-1]:
        d = d[:,:,::-1]
    if y[0,0]>y[-1,0]:
        d = d[:,::-1,:]
    return d


def _linecut_old(d,x=None,y=None):
    '''
    Extract data from a linecut, assume data is on grid, uniformly sampled, and autoflipped
    scipy.interpolate.interp2d is too slow, scipy.interpolate.RectBivariateSpline not good
    It's different from the simpler linecut in the interactive figure.
    '''
    if (x is None and y is None) or (x is None and len(np.shape(y))>0) or (y is None and len(np.shape(x))>0):
        raise ValueError('Invalid parameters for linecut')
    x0 = d[0][0]
    y0 = d[1][:,0]
    # horizontal linecut
    if x is None:
        x = x0
        y = np.full(len(x0), y)
    # vertical linecut
    if y is None:
        y = y0
        x = np.full(len(y0), x)
    #scale to "index" space
    indx = (x-x0[0])/(x0[-1]-x0[0])*(len(x0)-1)
    indy = (y-y0[0])/(y0[-1]-y0[0])*(len(y0)-1)
    z = scipy.ndimage.map_coordinates(d[2], [indy, indx], order=1)
    return np.vstack((x,y,z))


def linecut(d,x=None,y=None):
    '''
    Return a slice of d with x or y. Assume XY coordinates on 2d grid.
    '''
    if (x is None and y is None) or (x is None and len(np.shape(y))>0) or (y is None and len(np.shape(x))>0):
        raise ValueError('Invalid parameters for linecut')

    # vcut
    if y is None:
        x0 = d[0][0]
        indx = np.abs(x0 - x).argmin()
        return d[:,:,[indx]]


    # hcut
    if x is None:
        y0 = d[1][:,0]
        indy = np.abs(y0 - y).argmin()
        return d[:,[indy],:]


def hist2d(d, z_min, z_max, bins):
    """Convert every column into a histogram, default bin amount is sqrt(n)."""
    X,Y,Z = d[:3]
    hist = np.apply_along_axis(lambda x: np.histogram(x, bins, (z_min, z_max))[0], 0, Z)

    binedges = np.linspace(z_min, z_max, bins + 1)
    bincoords = (binedges[:-1] + binedges[1:]) / 2

    X = np.tile(X[0,:], (hist.shape[0], 1))
    Y = np.tile(bincoords[:,np.newaxis], (1, hist.shape[1]))
    Z = hist

    return np.stack([X,Y,Z])