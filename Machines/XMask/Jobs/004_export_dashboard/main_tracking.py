
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import itertools

# BOKEH
import bokeh.plotting as bk
import bokeh.models as bkmod
import bokeh.layouts as bklay
import bokeh.palettes as bkpalettes

# bk.output_notebook()

# xsuite
import xtrack as xt
import xmask as xm
import xfields as xf
import xpart as xp

# Custom imports
import bokeh_tools as bktools
import Presets as bkpresets

# BBStudies
import sys
sys.path.append('/Users/pbelanger/ABPLocal/BBStudies')
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.InteractionPoint as inp
import BBStudies.Physics.Detuning as tune
import BBStudies.Plotting.BBPlots as bbplt
import BBStudies.Physics.Base as phys
import BBStudies.Physics.Constants as cst


import ruamel.yaml
ryaml = ruamel.yaml.YAML()
def read_configuration(config_path="config.yaml"):
    # Read configuration for simulations
    with open(config_path, "r") as fid:
        config = ryaml.load(fid)


    return config


# Setting default values
#=====================================
_default_fig_width  = 1500
_default_fig_height = 400

_top_tab_height     = 300
_bot_tab_height     = 500
_default_fig_pad    = 100

BOKEH_FIGS          = {}
#=====================================


# Loading data
#=====================================
bunch_number = '0000'
data_path    = '../003_particle_dist_and_track/zfruits/BBB_Signature_V2/DATA/'

data      = xPlus.Tracking_Interface.from_parquet(data_path,partition_name='BUNCH',partition_ID=bunch_number)
_cpt      = xPlus.Tracking_Interface.from_parquet(data_path.replace('DATA','CHECKPOINTS'),partition_name='BUNCH',partition_ID=bunch_number)
data._checkpoint = _cpt._checkpoint
#=====================================


# MASTER SLIDER
#=====================================
padding = 100
legend_estimate = 130
chunk_slider = bkmod.Slider(start=data.data['Chunk ID'].min(), end=data.data['Chunk ID'].max(), value=0, step=1, title="Chunk ID",width = _default_fig_width-2*padding-legend_estimate,margin=[20,0,0,padding])
# chunk_slider_clone = chunk_slider.clone()
# chunk_slider_clone.value = chunk_slider.value
#=====================================

# INTENSITY PLOT
#=====================================
BOKEH_FIGS['Intensity'] = bkpresets.make_intensity_fig(data,slider=chunk_slider,title='Intensity',padding=padding,width=_default_fig_width,height=_top_tab_height)
grid1 = bklay.gridplot([[BOKEH_FIGS['Intensity']]],toolbar_location='right')
#=====================================

# Phase space plots:
#=====================================
adjustment = 50
BOKEH_FIGS['x-px'] = bkpresets.make_scatter_fig(data.checkpoint_sig,xy=('x_sig','px_sig'),slider=chunk_slider,title='x norm. phase space',width=int(_default_fig_width/3.5)+adjustment,height=_bot_tab_height,padding=0)
BOKEH_FIGS['y-py'] = bkpresets.make_scatter_fig(data.checkpoint_sig,xy=('y_sig','py_sig'),slider=chunk_slider,title='y norm. phase space',width=int(_default_fig_width/3.5),height=_bot_tab_height,padding=0)
BOKEH_FIGS['zeta-pzeta'] = bkpresets.make_scatter_fig(data.checkpoint_sig,xy=('zeta_sig','pzeta_sig'),slider=chunk_slider,title='zeta norm. phase space',width=int(_default_fig_width/3.5),height=_bot_tab_height,padding=0)


BOKEH_FIGS['x-px'].min_border_left  = padding
bktools.set_aspect(BOKEH_FIGS['x-px']       , x_lim=(-6,6),y_lim=(-6,6), aspect=1, margin=padding-adjustment)
bktools.set_aspect(BOKEH_FIGS['y-py']       , x_lim=(-6,6),y_lim=(-6,6), aspect=1, margin=0)
bktools.set_aspect(BOKEH_FIGS['zeta-pzeta'] , x_lim=(-1,1),y_lim=(-1,1), aspect=1, margin=0)


BOKEH_FIGS['x-px'].xaxis.axis_label = r'$$\tilde x /\sqrt{\varepsilon_{x}}$$'
BOKEH_FIGS['x-px'].yaxis.axis_label = r'$$\tilde p_x /\sqrt{\varepsilon_{x}}$$'

BOKEH_FIGS['y-py'].xaxis.axis_label = r'$$\tilde y /\sqrt{\varepsilon_{y}}$$'
BOKEH_FIGS['y-py'].yaxis.axis_label = r'$$\tilde p_y /\sqrt{\varepsilon_{y}}$$'

BOKEH_FIGS['zeta-pzeta'].xaxis.axis_label = r'$$\tilde \zeta /\sqrt{\varepsilon_{\zeta}}$$'
BOKEH_FIGS['zeta-pzeta'].yaxis.axis_label = r'$$\tilde p_\zeta /\sqrt{\varepsilon_{\zeta}}$$'


grid2 = bklay.gridplot([[BOKEH_FIGS['x-px'] ,BOKEH_FIGS['y-py'] ,BOKEH_FIGS['zeta-pzeta']]],toolbar_location='right')
#=====================================


# JxJy plots
#=====================================
BOKEH_FIGS['JxJy'] = bkpresets.make_JxJy_fig(data,slider = chunk_slider,title='(Jx,Jy) distribution',width = int(1.5*_default_fig_width/3),height=_bot_tab_height,padding = padding)
BOKEH_FIGS['JxJy'].xaxis.axis_label = r'$$J_x/\varepsilon_{x}$$'
BOKEH_FIGS['JxJy'].yaxis.axis_label = r'$$J_y/\varepsilon_{y}$$'
bktools.set_aspect(BOKEH_FIGS['JxJy'] , x_lim=(-5,65),y_lim=(-1,50), aspect=0.9)
# grid3 = bklay.gridplot([[BOKEH_FIGS['JxJy']]],toolbar_location='right')
#=====================================

# Collimation fig:
#=====================================
BOKEH_FIGS['Collimation'] = bkpresets.make_collimation_fig(data,slider = chunk_slider,title='Collimation',width = int(1.5*_default_fig_width/3),height=_bot_tab_height,padding = padding)
bktools.set_aspect(BOKEH_FIGS['Collimation'] , x_lim=(-4e-3,4e-3),y_lim=(-4e-3,4e-3), aspect=0.9)
grid3 = bklay.gridplot([[BOKEH_FIGS['JxJy'],BOKEH_FIGS['Collimation']]],toolbar_location='right')
#=====================================



# IMPORTING COLLIDER

# Importing Collider and Twiss
#-------------------------------------
# collider = xt.Multiline.from_json('../001_configure_collider/zfruits/collider_001.json')
collider = xt.Multiline.from_json(f'../001_configure_collider/zfruits/collider_BUNCHED/collider_BUNCH_{bunch_number}.json')
twiss = {}
twiss['lhcb1'] = collider['lhcb1'].twiss().to_pandas()
twiss['lhcb2'] = collider['lhcb2'].twiss().reverse().to_pandas()
#-------------------------------------


# Filtering twiss to get rid of slices, entries and exits
#-------------------------------------
light_twiss = {}
for sequence in ['lhcb1','lhcb2']:
    light_twiss[sequence] = xPlus.filter_twiss(twiss[sequence].set_index('name'),entries=['drift','..','_entry','_exit']).reset_index()
#-------------------------------------


# Making figures
#-------------------------------------
BOKEH_FIGS = {}
BOKEH_FIGS['twiss']   =  bkpresets.make_Twiss_Fig(collider,light_twiss,width=_default_fig_width,height=_default_fig_height,
                                                  twiss_columns=['x','y','px','py','betx','bety','alfx','alfy','dx','dy','dpx','dpy','mux','muy'])
BOKEH_FIGS['lattice'] =  bkpresets.make_LHC_Layout_Fig(collider,twiss,width=_default_fig_width,height=_default_fig_height)
#-------------------------------------

# Setting up axes
#-------------------------------------
BOKEH_FIGS['lattice'].xaxis[1].visible = False
BOKEH_FIGS['twiss'].x_range = BOKEH_FIGS['lattice'].x_range
BOKEH_FIGS['lattice'].min_border_left  = padding
BOKEH_FIGS['twiss'].min_border_left    = padding

# grid_collider = bklay.gridplot([[BOKEH_FIGS['lattice']],[BOKEH_FIGS['twiss']]],toolbar_location='right')
grid_collider = bklay.column(BOKEH_FIGS['lattice'],BOKEH_FIGS['twiss'])
#-------------------------------------









# # From charGPT
# # Function to convert a list of dictionaries to an HTML string with headers
# def dict_to_html(dictionaries, headers, margin=10, width= int(_default_fig_width)-10):
#     html_content = ""
#     for i, d in enumerate(dictionaries):
#         # Add a tab character before the div content
#         html_content += f"\t<div style='background-color: #f4f4f4; padding: 10px; border-radius: 5px; margin: 0 0 0 {margin}px; width: {width}px;'><h2>{headers[i]}:</h2><ul style='list-style-type: none; padding: 0; margin: 0;'>"
#         for key, value in d.items():
#             # Use a table with fixed width for the key column
#             html_content += f"\t\t<li style='line-height: 1;'><table><tr><td style='width: 110px;'><strong>{key}:</strong></td><td style='padding-left: 10px;'>{value}</td></tr></table></li>"
#         html_content += "</ul></div>"
#         if i < len(dictionaries) - 1:
#             # Add a tab character before the horizontal line
#             html_content += "\t<hr style='margin: 5px;'>"
#     return html_content


# Create a Bokeh Div component and set its content
metadata = data.to_dict()
metadata.pop('parquet_data');
metadata.pop('W_matrix');
metadata.pop('particle_on_co');
metadata['chunk size'] = (data.data.stop_at_turn - data.data.start_at_turn).max()
# tracking_info = bkmod.Div(text=dict_to_html([metadata],['Tracking Info']),width=_default_fig_width, height=_bot_tab_height)
tracking_info = bktools.dict_to_div([metadata],['Tracking Info'],width=_default_fig_width-10, height=_bot_tab_height, margin=10, force_width=120)

config_collider = read_configuration('../001_configure_collider/config.yaml')['config_collider']
collider_info = bktools.dict_to_div([config_collider],['Collider config.yaml'],width=2*_default_fig_width-10, height=_bot_tab_height+_top_tab_height, margin=10, force_width=0)


# Final layout
#=====================================
tab_margin = 20
bottom_tabs = bkmod.Tabs(tabs=[ bkmod.TabPanel(child=grid2, title="Phase Space"), 
                                bkmod.TabPanel(child=grid3, title="Collimation"),
                                bkmod.TabPanel(child=tracking_info, title="Info")])
bottom_tabs = bkmod.Row(bottom_tabs, margin=(0,0,0,tab_margin))

tab_margin  = 0
global_tabs = bkmod.Tabs(tabs=[bkmod.TabPanel(child=bklay.column(chunk_slider,grid1,bottom_tabs), title="Tracking"),
                               bkmod.TabPanel(child=grid_collider, title="Collider Object"),
                               bkmod.TabPanel(child=collider_info, title="Config")])
global_tabs = bkmod.Row(global_tabs, margin=(tab_margin,tab_margin,tab_margin,tab_margin))
HTML_LAYOUT = global_tabs
# HTML_LAYOUT = bklay.column(chunk_slider,grid1,grid2,grid3)
#=====================================



# Setting font size
for _fig in BOKEH_FIGS.values():
    _fig.xaxis.axis_label_text_font_size = "15px"
    _fig.yaxis.axis_label_text_font_size = "15px"

# Exporting to HTML
#=====================================
bktools.export_HTML(HTML_LAYOUT,f'zfruits/tracking_results_B{bunch_number}.html',f'Tracking Results')
#=====================================

