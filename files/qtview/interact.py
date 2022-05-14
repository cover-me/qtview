from . import data, operation, plot
import os
import matplotlib as mpl
import matplotlib.pylab as plt
import ipywidgets as widgets
import numpy as np

class Player:
    
    # CONSTANTS
    PLAYER_STATIC = False # set it True to make Player generate static figures (for previewing in viewers).
    
    def __init__(self,path_or_url,**kw):
        # data
        if not mpl.get_backend() == 'module://ipympl.backend_nbagg':
            raise "Need ipympl backend."

        d = data.Data(path_or_url,**kw)

 
        self.d = d
        self.kw_image = {'labels':d.labels}
        if 'labels' in kw:
            self.kw_image['labels'] = kw['labels']
        self.path = path_or_url
            
        
        self.create_ui()
        self.draw(event=None)
        
        self.slider_gamma.observe(self.on_gamma_change,'value')
        self.slider_vlim.observe(self.on_vlim_change,'value')
        self.dd_cmap.observe(self.on_cmap_change,'value')
        self.slider_xpos.observe(self.on_xpos_change,'value')
        self.slider_ypos.observe(self.on_ypos_change,'value')
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.b_save_data.on_click(self.save_data)
        self.b_save_cuts.on_click(self.save_cuts)
        
    def create_ui(self):
        x,y,z = self.d.data
        x0 = x[0]
        y0 = y[:,0]
        xmin,xmax,dx = x0[0],x0[-1],x0[1]-x0[0]
        ymin,ymax,dy = y0[0],y0[-1],y0[1]-y0[0]
        
        zmin,zmax = np.min(z),np.max(z)
        dz = (zmax-zmin)/100
        
        ## Tab of tools
        self.slider_xpos = widgets.FloatSlider(value=(xmin+xmax)/2,min=xmin,max=xmax,step=dx,description='x')
        self.slider_ypos = widgets.FloatSlider(value=(ymin+ymax)/2,min=ymin,max=ymax,step=dy,description='y')
        vb1 = widgets.VBox([self.slider_xpos,self.slider_ypos])
        
        self.slider_gamma = widgets.IntSlider(value=0,min=-100,max=100,step=1,description='gamma')
        self.slider_vlim = widgets.FloatRangeSlider(value=[zmin,zmax], min=zmin, max=zmax, step=dz, description='vlim')
        self.dd_cmap = widgets.Dropdown(value='seismic', options=plt.colormaps(), description='cmap:', disabled=False)
        vb2 = widgets.VBox([self.slider_gamma,self.slider_vlim,self.dd_cmap])
        
    
        self.b_save_data = widgets.Button(description='Save data')
        self.b_save_cuts = widgets.Button(description='Save cuts')
        self.dd_data_type = widgets.Dropdown(value='dat', options=['dat','npz','mtx'], description='',disabled=False)
        self.html_info = widgets.HTML()
        vb3 = widgets.VBox([self.dd_data_type,self.b_save_data,self.b_save_cuts, self.html_info])

        ## Top layer ui
        ui = widgets.Box([vb1,vb2,vb3])
        self.out = widgets.Output()
        
        display(ui,self.out)
            
        
    def draw(self,event):
        # axs
        fig, axs = plt.subplots(1,2,figsize=(6.5,2.5))# main plot and h linecut
        
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = True
        plt.subplots_adjust(wspace=0.4,bottom=0.2)
        axs[1].yaxis.tick_right()
        axs[1].tick_params(axis='x', colors='tab:orange')
        axs[1].tick_params(axis='y', colors='tab:orange')
        axv = fig.add_axes(axs[1].get_position(), frameon=False)#ax vertical linecut
        axv.xaxis.tick_top()
        axv.tick_params(axis='x', colors='tab:blue')
        axv.tick_params(axis='y', colors='tab:blue')
        
        self.fig = fig
        self.ax = axs[0]
        self.axv = axv
        self.axh = axs[1]


        gm = self.slider_gamma.value
        v0,v1 = self.slider_vlim.value
        cmap = self.dd_cmap.value
        if cmap not in plt.colormaps():
            cmap = 'seismic'
        self.kw_image['gamma'],self.kw_image['vmin'],self.kw_image['vmax'],self.kw_image['cmap']=gm,v0,v1,cmap
        plot.plot2d(self.d.data,fig=self.fig,ax=self.ax,**self.kw_image)
        self.im = [obj for obj in self.ax.get_children() if isinstance(obj, mpl.image.AxesImage) or isinstance(obj,mpl.collections.QuadMesh)][0]

        # prepare for linecuts
        self.axv.set_ylim(self.ax.get_ylim())
        self.axh.set_xlim(self.ax.get_xlim())
        

        # vlinecut
        xpos = self.slider_xpos.value
        d_vcut = self.get_v_cut(self.d.data,xpos)
        [self.linev1] = axs[0].plot(d_vcut[0],d_vcut[1],'tab:blue')
        [self.linev2] = axv.plot(d_vcut[2],d_vcut[1],'tab:blue')
        self.d_vcut = d_vcut

        # hlinecut
        ypos = self.slider_ypos.value
        d_hcut = self.get_h_cut(self.d.data,ypos)
        [self.lineh1] = axs[0].plot(d_hcut[0],d_hcut[1],'tab:orange')
        [self.lineh2] = axs[1].plot(d_hcut[0],d_hcut[2],'tab:orange')
        self.d_hcut = d_hcut
        
        plt.show()

    def get_v_cut(self,data,xpos):
        x0 = data[0,0,:]
        indx = np.abs(x0 - xpos).argmin()# x0 may be a non uniform array
        return data[:,:,indx]
        
    def get_h_cut(self,data,ypos):
        y0 = data[1,:,0]
        indy = np.abs(y0 - ypos).argmin()
        return data[:,indy,:]        
        
        
    def on_gamma_change(self,change):
        cmpname = self.dd_cmap.value
        if cmpname not in plt.colormaps():
            cmpname = 'seismic'
        g = change['new']
        g = 10.0**(g / 100.0)# to be consistent with qtplot
        if g!= 1:
            self.im.set_cmap(plot._get_cmap_gamma(cmpname,g,1024))
        else:
            self.im.set_cmap(cmpname)

    def on_cmap_change(self,change):
        cmap = change['new']
        if cmap in plt.colormaps():
            self.im.set_cmap(cmap)
    
    def on_vlim_change(self,change):
        v0,v1 = change['new']
        self.im.set_clim(v0,v1)
    
    def on_xpos_change(self,change):
        xpos = change['new']
        d_vcut = self.get_v_cut(self.d.data,xpos)
        self.linev1.set_xdata(d_vcut[0])
        self.linev2.set_xdata(d_vcut[2])
        self.axv.relim()
        self.axv.autoscale_view()
        self.v_hcut = d_vcut

    def on_ypos_change(self,change):
        ypos = change['new']
        d_hcut = self.get_h_cut(self.d.data,ypos)
        self.lineh1.set_ydata(d_hcut[1])
        self.lineh2.set_ydata(d_hcut[2])
        self.axh.relim()
        self.axh.autoscale_view()
        self.d_hcut = d_hcut

    def on_mouse_click(self,event):
        x,y = event.xdata,event.ydata
        if self.slider_xpos.value != x:
            self.on_xpos_change({'new':x})
        if self.slider_ypos.value != y:
            self.on_ypos_change({'new':y})
    
    def save_data(self,_):
        self.html_info.value = 'Saving...'
        
        d_type = self.dd_data_type.value
        labels = self.kw_image['labels']

        fname = os.path.split(self.path)[1]
        fname = os.path.splitext(fname)[0]
        fname = '%s.fig.%s'%(fname,d_type)
        
        self.d.save_data(fname,self.d.data,labels)
        
        self.html_info.value = 'File saved:<br>%s'%(fname)
        
    def save_cuts(self,_):
        self.html_info.value = 'Saving...'
        
        d_type = self.dd_data_type.value
        labels = self.kw_image['labels']
        
        fname = os.path.split(self.path)[1]
        fname = os.path.splitext(fname)[0]
        
        
        # vlincut
        fnamev = fname+'.vcut.'+d_type
        self.d.save_data(fnamev,self.d_vcut[:,np.newaxis,:],labels)

        # hlincut
        fnameh = fname+'.hcut.'+d_type
        self.d.save_data(fnameh,self.d_vcut[:,np.newaxis,:],labels)

        self.html_info.value = 'Files saved:<br>%s<br>%s'%(fnamev,fnameh)