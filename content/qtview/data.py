'''
A collection of functions loading or saving data
'''

import os, zipfile, time
import matplotlib as mpl
import numpy as np
from urllib.request import urlopen
from collections import OrderedDict
from . import operation

def get_opener(path_or_url):
    '''
    Return an open function for the file on internet, inside a ZIP file, or on local computer.
    '''
    if mpl.is_url(path_or_url):
        return lambda url,mode: urlopen(url)
    elif '.zip/' in path_or_url:
        return lambda url,mode: zipfile.ZipFile(url.split('.zip/')[0]+'.zip').open(url.split('.zip/')[1])
    else:
        return open
    

class Data2d:
    """
    Class for data loading, saving, and processing
    Data structure: d = [a1,a2,a3,...], a_i is 2d.
    """
    
    def __init__(self,fpath=None,**kw):
        if fpath:
            self.load_data(fpath,**kw)
        else:
            self.reset()
            
    def reset(self):
        # values updated after load data
        self.filename = ''
        self.raw_labels = None
        self.raw_data = None
        self.qtlab_settings = {}
    
        # values updated after process_data
        self.columns = None
        self.labels = None
        self.data = None
        
    def load_data(self,fpath,**kw):
        """
        Load data from fpath
        Data structure: d = [a1,a2,a3,...], a_i is 2d (heatmaps) or 1d (curves).
        """
        self.reset()
        self.load_qtlab_settings(fpath)
        
        if fpath.endswith('.mtx'):
            self.load_mtx(fpath)
        elif fpath.endswith('.npz'):
            self.load_npz(fpath)
        elif fpath.endswith('.dat'):
            self.load_dat(fpath)
        else:
            raise "Format not supported"
            
        self.filename = os.path.split(fpath)[1]
        self.process_data(**kw)
            
    def process_data(self,cook=None,cols=None,**kw):
        '''
        Apply cook (processing function) and cols (columns) to self.raw_data and self.raw_labels
        new data is updated to self.data and self.labels
        '''
        if cols is None:
            cols = [1,0,3] if len(self.raw_data)>3 else [0,1,2]
        self.columns = cols
        self.labels = [self.raw_labels[i] for i in cols]
        d = np.copy(self.raw_data[np.array(cols)])
        self.data = operation.autoflip(d)
        if cook:
            self.data = cook(self.data)
        
    def load_dat(self,fpath):
        """
        qtlab DAT file structure:
            Head: Comment lines with metadata, begin with "#". 
            Data: Tabular data with text numbers.
        for details, see https://github.com/cover-me/qtplot#dat-file-qtlab
        """
        sizes = []
        labels = []        
        open2 = get_opener(fpath)
        with open2(fpath, 'rb') as f:
            
            # metadata
            for line in f:
                line = line.decode()
                line = line.rstrip('\n\t\r')
                if line.startswith('#\tname'):
                    labels.append(line.split(': ', 1)[1])
                elif line.startswith('#\tsize'):
                    sizes.append(int(line.split(': ', 1)[1]))
                if len(line) > 0 and line[0] != '#':# where comments end
                    break
                    
            # data
            f.seek(0, 0)
            d0 = np.loadtxt(f)
            
            # organize the data
            row, col = d0.shape
            
            if len(labels) != col:
                raise Exception('Number of labels does not match the data.')
                
            if len(sizes)==1:
                sizes = [sizes[0],1]
            elif len(sizes)==2:
                pass
            elif np.prod(sizes[3:])==1:
                sizes = sizes[:2]
            else:
                raise Exception('Scan dimension not supported.')

            nx = sizes[0]
            ny = int(np.ceil(float(row)/nx))# ny maybe smaller than sizes[1] if scan is interrupted by a user

            # d0 is a tabular (2d), we want to convert it to an array of matrixes [X,Y,Z,...] (3d)
            d1 = np.full((nx*ny,col), np.nan)# initialize with na.nan
            d1[:row] = d0
            d1 = d1.T
            d1 = d1.reshape([col,ny,nx])
            
            # If scan interrupts in advance, there may be NaN in X,Y,Z,...
            # We calculate values where there is NaN for X and Y.
            if np.shape(d1)[1]>2:
                nans = np.isnan(d1[0,-1,:])
                d1[:2,-1,nans] = d1[:2,-2,nans]*2.-d1[:2,-3,nans]

            self.raw_data = d1
            self.raw_labels = labels

        
    
    def load_mtx(self,fpath):
        '''
        MTX file structure:
            Units, Dataset name, xname, xmin, xmax, yname, ymin, ymax, zname, zmin, zmax
            nx ny nz length
            [binary data....]
        mtx is created by Gary Steele, https://nsweb.tn.tudelft.nl/~gsteele/spyview/#mtx
        mtx files can be generated by spyview and qtplot
        XY coordinates are not stored in MTX files, so MTX is not good if X and Y are not on a 2d grid.
        NPZ is better.
        '''
        open2 = get_opener(fpath)
        with open2(fpath, 'rb') as f:
            line1 = f.readline().decode().rstrip('\n\t\r')
            if not line1.startswith('Units'):
                raise Exception('Not an MTX file!') 
            else:
                # xname,yname,zname,dataset name
                _ = line1.split(',') 
                labels = [x.strip() for x in [_[2],_[5],_[8],_[1]]]
                
                # nx ny nz element_length_in_byte
                line2 = f.readline().decode()
                nx,ny,nz,len_in_byte = [int(x) for x in line2.split(' ')]
                if nz != 1:
                    raise Exception('3d data not supported!')
                else:
                    x = np.linspace(float(_[3]),float(_[4]),nx)
                    y = np.linspace(float(_[6]),float(_[7]),ny)
                    z = np.linspace(float(_[9]),float(_[10]),nz)
                    z,y,x = np.meshgrid(z,y,x,indexing='ij')
                    
                    dtp = np.float64 if len_in_byte == 8 else np.float32# data type
                    w = np.frombuffer(f.read(),dtype=dtp).reshape([nx,ny,nz]).T

                    self.raw_data = np.stack([x[0],y[0],z[0],w[0]],axis=0)
                    self.raw_labels = labels

    
    def load_npz(self, fpath):
        """
        NPZ data is a dictionary:
        {"n0":a0,"n1":a1,"n2":a2,...},
        a_i's are all 2d, or a_0 and a_1 are 1d with (start, end, size)
        """
        d = np.load(fpath)
        labels = d.files
        data = [d[i] for i in labels]

        if data[0].shape == (3,) and data[1].shape == (3,):
            x0,x1,xsize = data[0]
            y0,y1,ysize = data[1]
            x = np.linspace(x0,x1,xsize)
            y = np.linspace(y0,y1,ysize)
            y,x = np.meshgrid(y,x,indexing='ij')
            data[0] = x
            data[1] = y

        self.raw_data = np.stack(data, axis=0)
        self.raw_labels = labels


    def save_npz(self,fpath,data,labels):
        data_dict = dict(zip(labels,data))
        X, Y = data[0], data[1]
        if self.are_xy_on_grid(X,Y):
            x0,x1,xsize = X[0,0],X[0,-1],len(X[0])
            y0,y1,ysize = Y[0,0],Y[-1,0],len(Y[:,0])
            data_dict[labels[0]] = np.array([x0,x1,xsize])
            data_dict[labels[1]] = np.array([y0,y1,ysize])
        np.savez(fpath,**data_dict)
        # print('NPZ data saved: %s'%fpath)

        
    def are_xy_on_grid(self,X,Y):
            """
            Check if XY coordinates are on a 2d grid.
            """
            return self.is_x_on_grid(X) and self.is_x_on_grid(Y.T)
        
    def is_x_on_grid(self,X):
        x_row0 = X[0]
        x0,x1,size = x_row0[0],x_row0[-1],len(x_row0)
        a = np.linspace(x0,x1,size)
        return np.all(X==a)

    def are_xy_ascending(self,X,Y):
            """
            check if XY coordinates are in the ascending order
            """
            x0 = X[0]
            y0 = Y[:,0]
            xmin,xmax,ymin,ymax = x0[0],x0[-1],y0[0],y0[-1]

            return xmin<=xmax and ymin<=ymax
              
    def read_set(self,dat_path):
        '''
        Read qtlab SET files (instrument settings)
        see Rubenknex/qtplot/qtplot/data.py.
        '''
        st = OrderedDict()
        set_path = dat_path.replace('.dat','.set')
        try:
            open2 = get_opener(set_path)
            with open2(set_path,'rb') as f:
                lines = f.readlines()
        except:
            print('Error opening setting file: %s'%set_path)
            return st
        
        current_instrument = None
        for line in lines:
            line = line.decode()
            line = line.rstrip('\n\t\r')
            if line == '':
                continue
            if not line.startswith('\t'):
                name, value = line.split(': ', 1)
                if (line.startswith('Filename: ') or
                   line.startswith('Timestamp: ')):
                    st.update([(name, value)])
                else:#'Instrument: ivvi'
                    current_instrument = value
                    new = [(current_instrument, OrderedDict())]
                    st.update(new)
            else:
                param, value = line.split(': ', 1)
                param = param.strip()
                new = [(param, value)]
                st[current_instrument].update(new)

        return st

    def save_mtx(self,fpath,data,labels):
        """
        MTX is not good, XY information lost, should avoid using it.
        """
        if not len(data)==3:
            raise Exception("Data size not right.")
        
        X,Y,Z = data
        
        if not self.are_xy_on_grid(X,Y):
            print('Warning! XY coordinates are not on 2d grids. XY information would lost!')
                   
        with open(fpath, 'wb') as f:
            labels = [i.replace(',','_') for i in labels]#',' is forbidden in MTX
            x0 = X[0]
            y0 = Y[:,0]
            xmin,xmax,ymin,ymax = x0[0],x0[-1],y0[0],y0[-1]
            # data_label,x_label,xmin,xmax,ylabel,ymin,ymax
            metadata = 'Units, %s,%s, %s, %s,%s, %s, %s,None, 0, 1\n'%(labels[2],labels[0],xmin,xmax,labels[1],ymin,ymax)
            f.write(metadata.encode())
            
            ny, nx = np.shape(Z)
            # dimensions nx,ny,nz=1,data_element_size
            metadata = '%d %d 1 %d\n'%(nx,ny,Z.dtype.itemsize)
            f.write(metadata.encode())
            
            # data
            Z.T.ravel().tofile(f)
            
            # print('MTX data saved: %s'%fpath)

    def save_dat(self,fpath,data,labels):
        if not len(data)==3:
            raise "Data size not right."
            
        fname = os.path.split(fpath)[1]
        meta = ''

        meta += '# Filename: %s\n' % fname
        meta += '# Timestamp: %s\n\n' % time.asctime(time.localtime())

        meta += '# Column %d\n' % 1
        meta += '#\tname: %s\n' % labels[0]
        meta += '#\tsize: %d\n' % data.shape[2]

        meta += '# Column %d\n' % 2
        meta += '#\tname: %s\n' % labels[1]
        meta += '#\tsize: %d\n' % data.shape[1]
        
        meta += '# Column %d\n' % 3
        meta += '#\tname: %s\n' % labels[2]
        meta += '\n'

        np.savetxt(fpath,data.reshape((3,-1)).T,fmt='%.12e',delimiter='\t',header=meta,comments='')
        # print('DAT data saved: %s'%fpath)
            
    def save_data(self,fpath,data,labels):
        if fpath.endswith('.mtx'):
            self.save_mtx(fpath,data,labels)
        elif fpath.endswith('.npz'):
            self.save_npz(fpath,data,labels)
        elif fpath.endswith('.dat'):
            self.save_dat(fpath,data,labels)
        else:
            raise('Format not recognized.')
            
            
    def load_qtlab_settings(self,fpath):
        path, ext = os.path.splitext(fpath)
        settings_file = path + '.set'
        self.qtlab_settings = {}
        if os.path.exists(settings_file):
            with open(settings_file) as f:
                lines = f.readlines()

            current_instrument = None
            for line in lines:
                line = line.rstrip('\n\t\r')
                if line == '':
                    continue
                if not line.startswith('\t'):
                    name, value = line.split(': ', 1)
                    if (line.startswith('Filename: ') or
                       line.startswith('Timestamp: ')):
                        self.qtlab_settings[name] = value
                    else:
                        current_instrument = value
                        self.qtlab_settings[current_instrument] = {}
                else:
                    param, value = line.split(': ', 1)
                    param = param.strip()
                    self.qtlab_settings[current_instrument][param] = value
    
 
    


    
    
