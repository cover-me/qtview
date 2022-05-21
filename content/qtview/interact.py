from . import data, operation, plot
import os,inspect,ast
import matplotlib as mpl
import matplotlib.pylab as plt
import ipywidgets as widgets
import numpy as np


LAYOUT_BTN = widgets.Layout(height='25',padding='0px',margin='0px',width='100px')
LAYOUT_LABEL = widgets.Layout(height='25px',padding='0px',width='100px')
LAYOUT_INPUT = widgets.Layout(height='25px',padding='0px',width='100px')

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 'medium',
    'figure.titlesize': 'medium',
    # Padding
    'axes.labelpad': 1,
    'axes.titlepad': 1,
    'xtick.major.pad': 1,
    'xtick.minor.pad': 1,
    'ytick.major.pad': 1,
    'ytick.minor.pad': 1,
    # Ticks
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    
    'figure.dpi':100,
})

class ProcessQueue:
    def __init__(self,parent):
        self.parent = parent
        self.queue = []
        self.index = -1
        self.select_area = widgets.VBox(())
        self.input_area = widgets.VBox(())
        self.ui = widgets.HBox([self.select_area,self.input_area])
    
    def add(self,name,func_and_args):
        '''
        name: name of the function
        func_and_args: (func, arg_names, arg_defaults)
        '''
        enabled = widgets.Checkbox(value=True,indent=False)
        enabled.layout = widgets.Layout(width='15px',margin='0px')
        enabled.observe(self.parent.on_data_change,'value')
        
        doc_string = func_and_args[0].__doc__.strip()
        selected = widgets.ToggleButton(value=False,description=name,tooltip=doc_string,layout=LAYOUT_BTN)
        selected.observe(self.on_selection_change,'value')
        
        select_item = widgets.HBox([enabled,selected])
        
        input_list = []
        
        for arg_name, arg_default in zip(func_and_args[1],func_and_args[2]):
            input_list.append(widgets.Label(value=arg_name,layout=LAYOUT_LABEL))
            _ = widgets.Text(value=str(arg_default),continuous_update=False,layout=LAYOUT_INPUT)
            _.observe(self.parent.on_data_change,'value')
            input_list.append(_)

        input_area = widgets.VBox(input_list)
        
        self.queue.append((select_item,input_area,func_and_args))
        selected.value = True# select this item, trigger self.on_selection_change
        
        self.update_ui()
        self.parent.on_data_change()
        
    def update_ui(self):
        self.select_area.children = [i[0] for i in self.queue]
        if self.index>-1:
            self.input_area.children = (self.queue[self.index][1],)            
        else:
            self.input_area.children = ()
    
    def remove(self,event=None):
        if len(self.queue):
            e = self.queue[self.index]
            for i in e[:2]:
                for j in i.children:
                    j.close()
            self.queue.pop(self.index)
            
            if self.index > len(self.queue)-1:
                self.index = len(self.queue)-1# could be -1 

            if self.index>-1:
                new_sel = self.queue[self.index][0].children[1]
                self.change_selected_silent(new_sel)
            
            self.update_ui()
            self.parent.on_data_change()
    
    def up(self,event=None):
        if self.index>0:
            self.queue[self.index],self.queue[self.index-1] = self.queue[self.index-1],self.queue[self.index]
            self.index -= 1
            self.update_ui()
            self.parent.on_data_change()
            
    def down(self,event=None):
        if self.index<len(self.queue)-1:
            self.queue[self.index],self.queue[self.index+1] = self.queue[self.index+1],self.queue[self.index]
            self.index += 1
            self.update_ui()
            self.parent.on_data_change()
            
    def clear(self,event=None):
        for i in self.queue:
            for j in i[:2]:
                for k in j.children:
                    k.close()
        self.queue = []
        self.index = -1
        self.select_area.children = []
        self.input_area.children = []
        self.parent.on_data_change()

    def change_selected_silent(self,s):
        s.unobserve(self.on_selection_change,'value')
        s.value = not s.value
        s.observe(self.on_selection_change,'value')  
        
    def on_selection_change(self,change):
        if change['new']==False:# already selected
            self.change_selected_silent(change['owner'])
        else:
            if self.index>-1:
                btn_old = self.queue[self.index][0].children[1]
                self.change_selected_silent(btn_old)
            self.index = self.get_index()
            self.update_ui()
            
    def get_index(self):
        for i in range(len(self.queue)):
            btn_selected = self.queue[i][0].children[1]
            if btn_selected.value:
                return i
        return -1

class Operations:
    '''
    An UI and methods for data operations (filters).
    '''
    def __init__(self,main_ui):
        self.main_ui = main_ui
        self.functions = self.get_functions()# A dictionary of functions in operation module. {fun_name: (func_callable,arg_names,arg_defaults),...}
        
        # Function list for selecting
        self.sel_funcs = widgets.Select(options=self.functions.keys(),description='')
        self.sel_funcs.layout = widgets.Layout(height='120px',width='100px')
        
        # Add, remove, move, clear and the Queue
        self.pq = ProcessQueue(parent=self)

        layout_btn = widgets.Layout(width='100px')
        self.btn_add = widgets.Button(description='Add',layout=layout_btn)
        self.btn_add.on_click(self.on_add)
        self.btn_up = widgets.Button(description='Up',layout=layout_btn)
        self.btn_up.on_click(self.pq.up)
        self.btn_down = widgets.Button(description='Down',layout=layout_btn)
        self.btn_down.on_click(self.pq.down)
        self.btn_remove = widgets.Button(description='Remove',layout=layout_btn)
        self.btn_remove.on_click(self.pq.remove)        
        self.btn_clear = widgets.Button(description='Clear',layout=layout_btn)
        self.btn_clear.on_click(self.pq.clear)
        buttons = widgets.VBox([self.btn_add,self.btn_up,self.btn_down,self.btn_remove,self.btn_clear])

        # Top UI
        self.ui = widgets.HBox([self.sel_funcs,buttons,self.pq.ui],layout={'width':'450px','border':'1px solid #ccc','padding':'2px 0px 0px 2px','margin':'0px 5px 2px 0px'})

        
    def on_data_change(self,change=None):
        # get the process function
        
        # Operation function change, cols change, or after loading data
        func_list = []
        for i in self.pq.queue:
            # i: (select_item,input_area,func_and_args)
            # select_item: widgets.HBox([enabled,selected])
            # input_area: VBox of [label, text,..., label, text, HTML]
            # func_and_args: (func, arg_names, arg_defaults)
            
            is_enabled = i[0].children[0].value
            if is_enabled:
                func,arg_names,arg_defaults = i[2]

                params = {}
                for counter in range(len(arg_names)):
                    name = arg_names[counter]
                    label = i[1].children[counter*2].value
                    if name != label:
                        raise "Error parameters."
                    val_default = arg_defaults[counter]
                    val_str = i[1].children[counter*2+1].value
                    if type(val_default)==str:
                        val = val_str
                    else:
                        val = ast.literal_eval(val_str)
                    params[name] = val
                    
                func_list.append((func,params))
            
        def process_function(d):
            for i,j in func_list:
                d = i(d,**j)
            return d
        
        # get columns
        cols = [self.main_ui.dd_x.value,self.main_ui.dd_y.value,self.main_ui.dd_z.value]

        self.main_ui.d.process_data(cook=process_function,cols=cols)
        
        # Update GUI
        self.main_ui.updata_vlim_bound_silently()
        if self.main_ui.c_auto_reset.value:
            # reset gamma and vlimit values
            self.main_ui.reset_cmap(silent=True)

        self.main_ui.redraw()                    
            
        
    def get_functions(self):
        '''
        Get functions from module operation.
        Return a dictionary whose item is like:
        fun_name: (func_callable,arg_names,arg_defaults)
        '''
        op_members = [i for i in inspect.getmembers(operation, inspect.isfunction) if not i[0].startswith('_')]# [(name,func)...]
        functions = []
        
        for i in op_members:
            a = inspect.getfullargspec(i[1])
            args = [] if a.args is None else a.args
            defaults = [] if a.defaults is None else a.defaults
            
            # only choose functions like: func(d,arg1=v1,arg2=v2,...)
            if (len(args)-1)==len(defaults) and args[0]=='d':
                functions.append((i[0],(i[1],args[1:],defaults)))#(name,(func,arg_names,arg_defaults))

        return dict(functions)     
    
    def on_add(self,event):
        name = self.sel_funcs.value
        self.pq.add(name,self.functions[name])    

        
        
        
        
        
class Player:
    '''
    Plot data interactively
    '''
    def __init__(self):
        if not mpl.get_backend() == 'module://ipympl.backend_nbagg':
            raise "Need ipympl backend."
            
        self.counter = 0# should remove in the future
 
        self.d = None# raw data, processed data, and methods for loading data
        self.operations = Operations(self)# provide UI and methods for data processing
        
        self.figures = None
        self.axes = None
        self.lines = None
        
        self.init_ui()        
        self.init_export_folder()

        self.on_path_change()
    
    def init_export_folder(self):
        self.export_folder = 'qtview exported data'
        if not os.path.exists(self.export_folder):
            os.makedirs(self.export_folder)

    def init_ui(self):
        '''
        Initialize user interface.
        toolbox1: operations
        toolbox2: plot settings
        fig_box1: main figure
        fig_box2: linecuts
        '''
        # to minimize width: layout={'width':'auto'}, to maximize width: layout={'width':'100%'}
        
        # HTML style
        html_sty = '''<style>
.widget-box {flex-wrap:wrap}
.widget-vbox .widget-label {width:auto;}
.widget-box .widget-hbox {flex-wrap:nowrap}
.widget-inline-hbox .widget-readout {min-width:30px}
.widget-dropdown > select {flex:1}
:root {--jp-widgets-font-size:10pt; --jp-widgets-inline-height:20px;--jp-widgets-input-padding:2px;--jp-widgets-inline-margin:2px;}
</style>
'''
        
        html_sty = widgets.HTML(html_sty)# applies even outside a box
        
        
        # toolbox1
        toolbox1 = self.operations.ui
        
        # toolbox2
        
        ## Dropdowns
        self.dd_file_path = widgets.Dropdown(description='',layout={'width':'100px'})# tooltip does not work at this moment
        self.update_file_list()
        self.dd_file_path.observe(self.on_path_change,'value')
        
        self.dd_x = widgets.Dropdown(layout={'width':'110px'})
        self.dd_x.observe(self.operations.on_data_change,'value')
        
        self.dd_y = widgets.Dropdown(layout={'width':'110px'})
        self.dd_y.observe(self.operations.on_data_change,'value')
        
        self.dd_z = widgets.Dropdown(layout={'width':'110px'})
        self.dd_z.observe(self.operations.on_data_change,'value')
        
        ## Sliders
        self.slider_gamma = widgets.IntSlider(value=0,min=-100,max=100,step=1,description='gamma',layout={'width':'150px'})
        self.slider_gamma.observe(self.on_gamma_change,'value')
        
        self.slider_vlim = widgets.FloatRangeSlider(value=[0,1], min=0, max=1, step=0.01, description='clim',readout_format='.2e',layout={'width':'280px'})
        self.slider_vlim.observe(self.on_vlim_change,'value')
        
        ## Buttons and checks
        self.btn_swp_xy = widgets.Button(description='Swap XY',layout={'width':'80px'})
        self.btn_swp_xy.on_click(self.on_swp_xy)
        
        self.b_reset = widgets.Button(description='Reset C',layout={'width':'70px'})
        self.b_reset.on_click(self.reset_cmap)
        
        self.b_save_data = widgets.Button(description='Save data',layout={'width':'82px'})
        self.b_save_data.on_click(self.save_data)
        
        self.c_auto_reset = widgets.Checkbox(value=True,description='Auto C',indent=False,layout={'width':'auto'})
        self.c_show_cuts = widgets.Checkbox(value=True,description='Linecuts',indent=False,layout={'width':'auto'})
        self.c_show_cuts.observe(self.on_show_cuts_change,'value')


        ## Dropdowns2
        self.dd_cmap = widgets.Dropdown(value='seismic', options=plt.colormaps(), description='cmap:', indent=False, disabled=False, layout={'width':'130px'})
        self.dd_cmap.observe(self.on_cmap_change,'value')
        
        self.dd_data_type = widgets.Dropdown(value='dat', options=['dat','npz','mtx'], description='Save as',disabled=False,layout={'width':'95px'})
        self.dd_data_source = widgets.Dropdown(value='figure', options=['figure','linecuts','raw'], description='Save from',disabled=False,layout={'width':'125px'})
        self.dd_plot_method = widgets.Dropdown(options=['imshow (default)','pcolormesh: if XY non-uniformly spaced'], description='Plot by',disabled=False,layout={'width':'125px'})

        ## information area
        self.html_info = widgets.HTML(value='Left-click on the image to show linecuts.',layout={'width':'auto'})
        
        widget_lines = []
        widget_lines.append(widgets.HBox([self.dd_file_path,self.dd_x,self.dd_y,self.dd_z]))
        widget_lines.append(widgets.HBox([self.slider_gamma,self.slider_vlim]))
        widget_lines.append(widgets.HBox([self.dd_cmap,self.b_reset,self.btn_swp_xy,self.c_auto_reset,self.c_show_cuts])) 
        widget_lines.append(widgets.HBox([self.dd_data_type,self.dd_data_source,self.b_save_data,self.dd_plot_method]))
        widget_lines.append(self.html_info)

        toolbox2 = widgets.VBox(widget_lines,layout=toolbox1.layout)
        
        # fig_boxes
        self.init_figures()
        fig_box1 = widgets.VBox([self.figures[0].canvas])
        fig_box2 = widgets.VBox([self.figures[1].canvas])
     

        ## Top layer ui
        self.toolboxes = widgets.Box([toolbox2,toolbox1])
        self.fig_boxes = widgets.Box([fig_box1,fig_box2])
        
        display(html_sty,self.toolboxes,self.fig_boxes)
        
    def init_figures(self):
        plt.ioff()
        a = 1
        fig = plt.figure(figsize=(4*a,3*a))#,facecolor='green')# main plot
        fig_cut = plt.figure(figsize=(4*a,3*a))# linecuts
        plt.ion()
        
        fig.canvas.mpl_connect('button_press_event', self.on_cut_pos_change)
        
        self.figures = [fig,fig_cut]
        for i in self.figures:
            i.canvas.header_visible = False
            # i.canvas.toolbar_visible = True
            i.canvas.toolbar_position = 'left'
            i.canvas.resizable = True

    def on_path_change(self,change=None):
        # fpath can be path, path inside zip, or url (url fetch is limited in JupyterLite, based on js, need to add "await" and run in independent cells)
        fpath = self.dd_file_path.value
        self.clear_linecuts()
        if self.d is None:
            self.d = data.Data2d(fpath)
        else:
            self.d.load_data(fpath)

        self.init_columns()
        self.operations.on_data_change()# this includes redraw()
            
    def clear_linecuts(self):
        if self.lines:
            for i in self.lines:
                i.set_data([],[])

    def init_columns(self,silent=True):
        raw_labels = self.d.raw_labels
        options = list(zip(raw_labels,range(len(raw_labels))))
        if silent:
            self.dd_x.unobserve(self.operations.on_data_change,'value')
            self.dd_y.unobserve(self.operations.on_data_change,'value')
            self.dd_z.unobserve(self.operations.on_data_change,'value')
            
        self.dd_x.options = options
        self.dd_y.options = options
        self.dd_z.options = options

        if self.dd_x.index == self.dd_y.index:
            # default columns
            if len(raw_labels)>3:
                # raw data or mtx
                cols = [1,0,3] if self.d.filename.endswith('.dat') else [0,1,3]
            else:
                cols = [0,1,2]
            self.dd_x.value = cols[0]
            self.dd_y.value = cols[1]
            self.dd_z.value = cols[2]
        
        if silent:
            self.dd_x.observe(self.operations.on_data_change,'value')
            self.dd_y.observe(self.operations.on_data_change,'value')
            self.dd_z.observe(self.operations.on_data_change,'value')
        
        
    def on_swp_xy(self,event=None):
        if not (self.dd_x.value is None or self.dd_y.value is None):
            self.dd_x.unobserve(self.operations.on_data_change,'value')
            vx = self.dd_x.value
            self.dd_x.value = self.dd_y.value
            self.dd_x.observe(self.operations.on_data_change,'value')
            self.dd_y.value = vx# this will trigger a change event

        
    def update_file_list(self):
        f_list = [i for i in os.listdir() if os.path.splitext(i)[1] in ['.dat','.mtx','.npy']]
        self.dd_file_path.options = f_list
            
    def on_show_cuts_change(self,change):
        if not change['new']:
            self.html_info.value = ''
            self.clear_linecuts()
            self.figures[0].canvas.draw()
            self.figures[1].canvas.draw()
            
        
    def on_cut_pos_change(self,click_event):
        if self.c_show_cuts.value and self.d is not None:
            x, y = click_event.xdata, click_event.ydata
            hx,hy,hz = np.copy(operation.linecut(self.d.data,y=y).reshape(3,-1))# [X, Y0, Z]
            vx,vy,vz = np.copy(operation.linecut(self.d.data,x=x).reshape(3,-1))# [X0, Y, Z]
            self.html_info.value = 'Cuts at (%s,%s)'%(vx[0],hy[0])# may be slightly different from x,y
            ax,axh,axv = self.axes

            if self.lines:
                l1h,l1v,l2h,l2v = self.lines
                l1h.set_data(hx,hy)
                l1v.set_data(vx,vy)
                l2h.set_data(hx,hz)
                l2v.set_data(vz,vy)
                z1,z2 = hz.min(),hz.max()
                dz = (z2-z1)/20
                axh.set_ylim(z1-dz,z2+dz)
                
                z1,z2 = vz.min(),vz.max()
                dz = (z2-z1)/20                
                axv.set_xlim(z1-dz,z2+dz)
            else:
                # lines = [l1h,l1v,l2h,l2v]
                self.lines = ax.plot(hx,hy,'tab:blue',vx,vy,'tab:orange')
                self.lines += axh.plot(hx,hz,'tab:blue')  
                self.lines += axv.plot(vz,vy,'tab:orange')
                
            self.figures[0].canvas.draw()
            self.figures[1].canvas.draw()

        
    def reset_cmap(self,event=None,silent=False):
        if silent:
            self.slider_gamma.unobserve(self.on_gamma_change,'value')
            self.slider_vlim.unobserve(self.on_vlim_change,'value')
        self.slider_gamma.value = 0
        self.slider_vlim.value = (self.slider_vlim.min,self.slider_vlim.max)
        if silent:
            self.slider_gamma.observe(self.on_gamma_change,'value')
            self.slider_vlim.observe(self.on_vlim_change,'value')
    
    def updata_vlim_bound_silently(self,reset_value=False):
        z = self.d.data[2]
        zmin,zmax = np.nanmin(z),np.nanmax(z)
        dz = (zmax-zmin)/100
        self.slider_vlim.unobserve(self.on_vlim_change,'value')
        if zmin>self.slider_vlim.max:# can not set zmin > max, can "="
            self.slider_vlim.max = zmax
            self.slider_vlim.min = zmin
        else:
            self.slider_vlim.min = zmin
            self.slider_vlim.max = zmax
        self.slider_vlim.step = dz
        if reset_value or self.slider_vlim.lower == self.slider_vlim.upper:
            self.slider_vlim.value = (zmin,zmax)
        self.slider_vlim.observe(self.on_vlim_change,'value')
        
    def create_linecut_axes(self,fig_cut):
        ch = 'tab:blue'
        cv = 'tab:orange'
        axh = fig_cut.add_axes([0.28,0.13,0.54,0.72])
        
        axh.yaxis.tick_right()
        axh.yaxis.set_label_position("right")
        axh.spines['bottom'].set_color(ch)
        axh.spines['top'].set_visible(False)
        axh.spines['left'].set_visible(False)
        axh.spines['right'].set_color(ch)
        axh.tick_params(axis='x', colors=ch)
        axh.tick_params(axis='y', colors=ch)
        axh.set_xlabel(self.d.labels[0],color=ch)
        axh.set_ylabel(self.d.labels[2],color=ch)

        axv = fig_cut.add_axes(axh.get_position())# ax vertical linecut
        axv.xaxis.tick_top()
        axv.xaxis.set_label_position("top")
        axv.spines['top'].set_color(cv)
        axv.spines['bottom'].set_visible(False)
        axv.spines['right'].set_visible(False)
        axv.spines['left'].set_color(cv)
        axv.tick_params(axis='x', colors=cv)
        axv.tick_params(axis='y', colors=cv)
        axv.set_xlabel(self.d.labels[2],color=cv)
        axv.set_ylabel(self.d.labels[1],color=cv)
        
        return axh,axv
        
    
    def redraw(self,event=None):

        fig,fig_cut = self.figures
        fig.clear()
        fig_cut.clear()# is more convenient because l1 and l2 has been cleared
        self.lines = None

        ax = fig.add_axes([0.28,0.13,0.54,0.72])
        ax_cbar = fig.add_axes([0.84,0.13,0.03,0.72])
        ax.set_title(self.d.filename)
        axh, axv = self.create_linecut_axes(fig_cut)
        
        self.axes = [ax,axh,axv]

        if self.d:
            gm = self.slider_gamma.value
            v0,v1 = self.slider_vlim.value
            cmap = self.dd_cmap.value
            is_xy_uniform = False
            if self.dd_plot_method.index == 0:# imshow
                is_xy_uniform = True
            kw = {'labels':self.d.labels,'gamma':gm, 'vmin':v0, 'vmax':v1, 'cmap':cmap, 'xyUniform':is_xy_uniform, 'plotCbar':{'cax':ax_cbar}}
            plot.plot2d(self.d.data,fig=fig,ax=ax,**kw)
            self.im = [obj for obj in ax.get_children() if isinstance(obj, mpl.image.AxesImage) or isinstance(obj,mpl.collections.QuadMesh)][0]

            # prepare for linecuts
            axv.set_ylim(ax.get_ylim())
            axh.set_xlim(ax.get_xlim())
            
        if self.counter == 0:# somehow if we canvas.draw() at the first time the figure would disappear.
            self.counter = 1
        else:
            self.figures[0].canvas.draw()
            self.figures[1].canvas.draw()

        
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
        self.figures[0].canvas.draw()

    def on_cmap_change(self,change):
        cmap = change['new']
        if cmap in plt.colormaps():
            self.im.set_cmap(cmap)
        self.figures[0].canvas.draw()
    
    def on_vlim_change(self,change):
        v0,v1 = change['new']
        self.im.set_clim(v0,v1)
        self.figures[0].canvas.draw()

    def save_data(self,event=None):
        if self.dd_data_source.value == 'figure':
            self.save_data_figure()
        elif self.dd_data_source.value == 'linecuts':
            self.save_data_cuts()
        elif self.dd_data_source.value == 'raw':
            self.save_data_raw()
    
    def save_data_figure(self):
        self.html_info.value = 'Saving...'
        
        d_type = self.dd_data_type.value

        fname = self.d.filename
        fname = os.path.splitext(fname)[0]
        fname = '%s/%s.2d.%s'%(self.export_folder,fname,d_type)
        
        self.d.save_data(fname,self.d.data,self.d.labels)
        
        self.html_info.value = 'File saved: %s'%(fname)
        
    def save_data_cuts(self):
        self.html_info.value = 'Saving...'
        
        d_type = self.dd_data_type.value
        
        fname = self.d.filename
        fname = os.path.splitext(fname)[0]
        
        
        # vlincut
        fnamev = '%s/%s.vcut.%s'%(self.export_folder,fname,d_type)
        self.d.save_data(fnamev,self.d_vcut[:,np.newaxis,:],self.d.labels)# save_data only takes 2d data

        # hlincut
        fnameh = '%s/%s.hcut.%s'%(self.export_folder,fname,d_type)
        self.d.save_data(fnameh,self.d_hcut[:,np.newaxis,:],self.d.labels)

        self.html_info.value = 'Files saved: %s<br>%s'%(fnamev,fnameh)
        
    def save_data_raw(self):
        self.html_info.value = 'Saving...'
        
        d_type = self.dd_data_type.value
        
        fname = self.d.filename
        fname = os.path.splitext(fname)[0]
        fname = '%s/%s.raw.%s'%(self.export_folder,fname,d_type)
        
        self.d.save_data(fname,self.d.raw_data,self.d.raw_labels)
        
        self.html_info.value = 'File saved: %s'%(fname)
