from . import data, operation
import matplotlib as mpl
import matplotlib.pylab as plt
import ipywidgets as widgets
import numpy as np


def get_default_ps():
    '''
    return a defult plot setting
    '''
    return {'labels':['','',''],'xyUniform':True,'gamma':0,'gmode':'moveColor',
          'cmap':'seismic','vmin':None, 'vmax':None,'plotCbar':True}

def plot(fpath,**kw):
    d = data.Data2d(fpath,**kw)
    plot2d(d.data,**kw)

def plot2d(data,**kw):
    '''
    Plot 2D figure. We need this method because plotting 2d is not as easy as plotting 1d.
    imshow() and pcolormesh() should be used in different situations. imshow() is prefered if x and y are uniformly spaced.
    For some interesting issues, check these links:
    https://cover-me.github.io/2019/02/17/Save-2d-data-as-a-figure.html
    https://cover-me.github.io/2019/04/04/Save-2d-data-as-a-figure-II.html
    '''
    x,y,w = data    
    # plot setting
    ps = get_default_ps()
    for i in ps:
        if i in kw:
            ps[i] = kw[i]

    #save fig data
    if 'figdatato' in kw and kw['figdatato']:
        save2d(kw['figdatato'],x,y,w,ps['labels'],ps['xyUniform'])

    if 'ax' in kw and 'fig' in kw:
        # sometimes you want to use your own ax
        fig = kw['fig']
        ax = kw['ax']
    else:
        # Publication quality first. Which means you don't want large figures with small fonts as those default figures.
        figsize = kw['figsize'] if 'figsize' in kw else (3.375,2)
        fig, ax = plt.subplots(figsize=figsize)

    x1 = operation._get_quad(x)# explained here: https://cover-me.github.io/2019/02/17/Save-2d-data-as-a-figure.html
    y1 = operation._get_quad(y)
    imkw = {'cmap':ps['cmap'],'vmin':ps['vmin'],'vmax':ps['vmax']}
    gamma_real = 10.0**(ps['gamma'] / 100.0)# to be consistent with qtplot
    if gamma_real != 1:
        if ps['gmode']=='moveColor':# qtplot style
            imkw['cmap'] = _get_cmap_gamma(imkw['cmap'],gamma_real,1024)
        else:# matplotlib default style
            imkw['norm'] = mpl.colors.PowerNorm(gamma=gamma_real)

    #Imshow is better than pcolormesh if it xy is uniformly spaced. See the links in operation._get_quad() description.
    if ps['xyUniform']:
        #data need to be autoflipped when imported
        xy_range = (x1[0,0],x1[0,-1],y1[0,0],y1[-1,0])
        im = ax.imshow(w,aspect='auto',interpolation='nearest',origin='lower',extent=xy_range,**imkw)
        #If there is only one dataset, clip the image a little to set xy limits to true numbers
        if ax.get_xlim() + ax.get_ylim() == xy_range:               
            ax.set_xlim(x[0,0],x[0,-1])
            ax.set_ylim(y[0,0],y[-1,0])
    else:
        im = ax.pcolormesh(x1,y1,w,rasterized=True,**imkw)

    if ps['plotCbar']:
        if type(ps['plotCbar']) == dict:
            cbar = fig.colorbar(im,ax=ax,**ps['plotCbar'])
        else:
            cbar = fig.colorbar(im,ax=ax)
        cbar.set_label(ps['labels'][2])
    else:
        cbar = None
    ax.set_xlabel(ps['labels'][0])
    ax.set_ylabel(ps['labels'][1])


def plot1d(x,y,w,**kw):
    '''A simple 1d plot function'''
    ps = {'labels':['','','']}
    for i in ps:
        if i in kw:
            ps[i] = kw[i]
    if 'ax' in kw and 'fig' in kw:
        fig = kw['fig']
        ax = kw['ax']
    else:
        fig, ax = plt.subplots(figsize=(3.375,2))
    ax.plot(x[0],w[0])
    ax.set_xlabel(ps['labels'][0])
    ax.set_ylabel(ps['labels'][1])


def _get_cmap_gamma(cname,g,n=256):
    '''Get a listed cmap with gamma'''
    cmap = mpl.cm.get_cmap(cname, n)
    cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 1, n)**g))
    return cmap


def simpAx(ax=None,cbar=None,im=None,n=(None,None,None),apply=(True,True,True),pad=(-5,-15,-10)):
    '''Simplify the ticks'''
    if ax is None:
        ax = plt.gca()

    if apply[0]:
        _min,_max = ax.get_xlim()
        if n[0] is not None:
            a = 10**(-n[0])
            _min = np.ceil(_min/a)*a
            _max = np.floor(_max/a)*a
        ax.set_xticks([_min,_max])
        ax.xaxis.labelpad = pad[0]

    if apply[1]:
        _min,_max = ax.get_ylim()
        if n[1] is not None:
            a = 10**(-n[1])
            _min = np.ceil(_min/a)*a
            _max = np.floor(_max/a)*a
        ax.set_yticks([_min,_max])
        ax.yaxis.labelpad = pad[1]

    if apply[2]:
        #assumes a vertical colorbar
        if cbar is None:
            if im:
                cbar = im.colorbar
            else:
                ims = [obj for obj in ax.get_children() if isinstance(obj, mpl.image.AxesImage) or isinstance(obj,mpl.collections.QuadMesh)]
                if ims:
                    im = ims[0]
                    cbar = im.colorbar
                else:
                    im,cbar = None, None
        if cbar is not None and im is not None:
            _min,_max = cbar.ax.get_ylim()
            label = cbar.ax.get_ylabel()
            if n[2] is not None:
                a = 10**(-n[2])
                _min = np.ceil(_min/a)*a
                _max = np.floor(_max/a)*a
                #im.set_clim(_min,_max)
            cbar.set_ticks([_min,_max])
            cbar.ax.yaxis.labelpad = pad[2]
            cbar.ax.set_ylabel(label)  


def formatLabel(fname,s):
    '''
    see qtplot/export.py
    '''
    conversions = {
        '<filename>': os.path.split(fname)[-1],
        '<operations>': '',
    }
    for old, new in conversions.items():
        s = s.replace(old, new)
    for key, value in data.readSettings(fname).items():
        if isinstance(value, dict):
            for key_, value_ in value.items():
                s = s.replace('<%s:%s>'%(key,key_), '%s'%value_)
    return s

