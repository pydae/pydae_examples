from pydae.svg_tools import svg
import ipywidgets as widgets
import numpy as np
import sympy as sym
import pydae.build_cffi as db
from pydae.grid_bpu import bpu

grid = bpu('bus3_trafo.json')

g_list = grid.dae['g'] 
h_dict = grid.dae['h_dict']
f_list = grid.dae['f']
x_list = grid.dae['x']
params_dict = grid.dae['params_dict']


sys = {'name':'bus3_trafo',
       'params_dict':params_dict,
       'f_list':f_list,
       'g_list':g_list,
       'x_list':x_list,
       'y_ini_list':grid.dae['y_ini'],
       'y_run_list':grid.dae['y_run'],
       'u_run_dict':grid.dae['u_run_dict'],
       'u_ini_dict':grid.dae['u_ini_dict'],
       'h_dict':h_dict}

dblr = db.builder(sys)
dblr.build()

import bus3_trafo
grid = bus3_trafo.bus3_trafo_class()


def set_path_style(self,object_id,new_style):
    #style="fill:#337ab7"
    for path in self.root.findall('.//{http://www.w3.org/2000/svg}path'):
        if 'id' in path.attrib:
            if path.attrib['id'] == object_id: 
                #if 'style' in rect.attrib:
                #    rect.attrib['style'].update(new_style)
                #else:
                path.attrib['style'] = new_style
   

class dashboard(bus3_trafo.bus3_trafo_class):
    
    def __init__(self,svg_file):
        
        super().__init__()
        
        grid = bus3_trafo.bus3_trafo_class()
       
        self.grid = grid
    
        self.svg_fig = svg(svg_file)

        self.sld_P_B3 = widgets.FloatSlider(orientation='horizontal',description = "P (GW)", 
                                    value=0, min=-2,max= 2,step=0.1,continuous_update=False)
        self.sld_Q_B3 = widgets.FloatSlider(orientation='horizontal',description = "Q (Gvar)", 
                                    value=0, min=-2,max= 2,step=0.1,continuous_update=False)

        self.sld_tap = widgets.FloatSlider(orientation='horizontal',description = "t (pu)", 
                                    value=1, min=0.95,max=1.05,step=0.01,continuous_update=False)
        self.sld_ang = widgets.FloatSlider(orientation='horizontal',description = "α (º)", 
                                    value=0, min=-30,max= 30,step=1,continuous_update=False)


        self.html_grid = widgets.HTML(self.svg_fig.tostring())
        
        self.update(0)

        self.sld_P_B3.observe(self.update, names='value')
        self.sld_Q_B3.observe(self.update, names='value')
        self.sld_tap.observe( self.update, names='value')
        self.sld_ang.observe( self.update, names='value')

        self.layout_row1 = widgets.HBox([self.html_grid,widgets.VBox([self.sld_P_B3,self.sld_Q_B3,self.sld_tap,self.sld_ang])])
        #layout_row2 = widgets.HBox([sel_vsc,,text_post])
        #layout_row3 = widgets.HBox([sld_q_vsc_R10])

        self.layout = self.layout_row1

    def update(self,change):
        #options=['VSC R10-S10','VSC R14-S14','VSC I02-H02','VSC C16-D16','VSC C09-D09'],
        tap = self.sld_tap.value
        ang = self.sld_ang.value

        P_B3 = self.sld_P_B3.value*1e9
        Q_B3 = self.sld_Q_B3.value*1e9
        self.grid.ini({'tap_B1_B2':tap,'ang_B1_B2':np.deg2rad(ang),'P_B3':-P_B3,'Q_B3':-Q_B3 },'xy_0.json')


        self.svg_fig.set_text(  'tap',f't ={self.grid.get_value("tap_B1_B2"):5.2f}')
        self.svg_fig.set_text('alpha',f'α ={np.rad2deg(self.grid.get_value("ang_B1_B2")):4.1f}º')

        P_B1_B3 = self.grid.get_value("p_line_B1_B3")
        Q_B1_B3 = self.grid.get_value("q_line_B1_B3")
        S_B1_B3 = (P_B1_B3**2 + Q_B1_B3**2)**0.5
        self.svg_fig.set_text('P_B1_B3',f'P = {P_B1_B3/10:0.2f} GW')
        self.svg_fig.set_text('Q_B1_B3',f'Q = {Q_B1_B3/10:0.2f} Gvar')
        self.svg_fig.set_text('S_B1_B3',f'S = {S_B1_B3/10:0.2f} GVA')

        P_B2_B3 = self.grid.get_value("p_line_B2_B3")
        Q_B2_B3 = self.grid.get_value("q_line_B2_B3")
        S_B2_B3 = (P_B2_B3**2 + Q_B2_B3**2)**0.5
        self.svg_fig.set_text('P_B2_B3',f'P = {P_B2_B3/10:0.2f} GW')
        self.svg_fig.set_text('Q_B2_B3',f'Q = {Q_B2_B3/10:0.2f} Gvar')
        self.svg_fig.set_text('S_B2_B3',f'S = {S_B2_B3/10:0.2f} GVA')

        self.svg_fig.set_text('P_B3',f'P = {-self.grid.get_value("P_B3")/1e9:4.2f} GW')
        self.svg_fig.set_text('Q_B3',f'Q = {-self.grid.get_value("Q_B3")/1e9:4.2f} Gvar')


        S_max = 1.1

        #svg_fig.set_color('path','line_B1_B3',[1,1,0])
        S = S_B1_B3/10
        red  = np.clip(255*((S - 0.0)/(S_max - 0))**4,0,255)
        width = 0.5+S/5
        set_path_style(self.svg_fig,'line_B1_B3',f'fill:none;stroke:#{int(red):02x}0000;stroke-width:{width};stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1')

        #set_path_style(svg_fig,'line_B2_B3','fill:none;stroke:#000000;stroke-width:0.529167;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1')

        #if S_B2_B3/10 > 1.02:

        S = S_B2_B3/10
        red  = np.clip(255*((S - 0.0)/(S_max - 0))**4,0,255)
        width = 0.5+S/5
        set_path_style(self.svg_fig,'line_B2_B3',f'fill:none;stroke:#{int(red):02x}0000;stroke-width:{width};stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1')

        #SVG(svg_fig.tostring())

        self.html_grid.value = self.svg_fig.tostring()
    
    def show(self):
        
        return self.layout
