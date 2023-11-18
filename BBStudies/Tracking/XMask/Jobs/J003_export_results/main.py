
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


# BBStudies

import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.XMask.Utils as xutils
import BBStudies.Physics.Constants as cst
import BBStudies.Plotting.Bokeh.Tools as bktools
import BBStudies.Plotting.Bokeh.Presets as bkpresets


# Setting default values
#=====================================
_default_fig_width  = 1500
_default_fig_height = 400

_top_tab_height     = 300
_bot_tab_height     = 500
_default_fig_pad    = 100
#=====================================

def tracking_to_HTML(data_path,checkpoint_path,partition_name,partition_ID):

    
    # Loading data
    #=====================================
    BOKEH_FIGS  = {}
    data        = xPlus.Tracking_Interface.from_parquet(data_path         ,partition_name=partition_name,partition_ID=partition_ID)
    _cpt        = xPlus.Tracking_Interface.from_parquet(checkpoint_path   ,partition_name=partition_name,partition_ID=partition_ID)
    data._checkpoint = _cpt._checkpoint
    #=====================================


    # MASTER SLIDER
    #=====================================
    padding         = _default_fig_pad
    legend_estimate = 130
    chunk_slider = bkmod.Slider(start=data.data['Chunk ID'].min(), end=data.data['Chunk ID'].max(), value=0, step=1, title="Chunk ID",width = _default_fig_width-2*padding-legend_estimate,margin=[20,0,0,padding])
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
    collider_path   = data.config['tracking']['collider_path']
    collider        = xt.Multiline.from_json(collider_path)

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




    # Adding tracking info
    #=====================================
    metadata = data.to_dict()
    metadata.pop('parquet_data');
    metadata.pop('W_matrix');
    metadata.pop('particle_on_co');
    metadata['chunk size'] = (data.data.stop_at_turn - data.data.start_at_turn).max()
    
    tracking_info = bktools.dict_to_HTML(metadata, header_text="Tracking Info", header_margin_top=20, header_margin_bottom=0,margin=20,indent= 2,nested_scale = 0.98, max_width = _default_fig_width)
    #=====================================


    # Adding collider config
    #=====================================
    config_list = {}
    for key in collider.metadata.keys():
        _info = bktools.dict_to_HTML(collider.metadata[key], header_text=key, header_margin_top=20, header_margin_bottom=0,margin=20,indent= 2,nested_scale = 0.98, max_width = _default_fig_width)
        config_list[key] = _info
    
    # Adding tracking config
    _info = bktools.dict_to_HTML(data.config, header_text='config_J002.yaml', header_margin_top=20, header_margin_bottom=0,margin=20,indent= 2,nested_scale = 0.98, max_width = _default_fig_width)
    config_list['config_J002.yaml'] = _info
    #=====================================



    # Final layout
    #=====================================
    tab_margin = 20
    bottom_tabs = bkmod.Tabs(tabs=[ bkmod.TabPanel(child=grid2, title="Phase Space"), 
                                    bkmod.TabPanel(child=grid3, title="Collimation"),
                                    bkmod.TabPanel(child=tracking_info, title="Info")])
    bottom_tabs = bkmod.Row(bottom_tabs, margin=(0,0,0,tab_margin))

    tab_margin  = 0
    global_tabs = bkmod.Tabs(tabs=[ bkmod.TabPanel(child=bklay.column(chunk_slider,grid1,bottom_tabs), title="Tracking"),
                                    bkmod.TabPanel(child=grid_collider, title="Collider Object")] + \
                                  [ bkmod.TabPanel(child=config_list[key], title=key) for key in config_list.keys()])
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
    html_filepath = str(Path(data_path).parents[0]) + f'/HTML/{partition_name}_{partition_ID}.html'
    xutils.mkdir(html_filepath)
    bktools.export_HTML(HTML_LAYOUT,html_filepath,f'Tracking - {partition_name} - {partition_ID}')
    #=====================================


# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-d", "--data_path"        , help = "Path to DATA folder"      ,default = '')
    aparser.add_argument("-c", "--checkpoint_path"  , help = "Path to CHECKPOINT folder",default = '')
    aparser.add_argument("-n", "--partition_name"   , help = "Partition Name"           ,default = '')
    aparser.add_argument("-i", "--partition_ID"     , help = "Partition ID"             ,default = '')
    
    
    args = aparser.parse_args()
    
    
    tracking_to_HTML(   data_path       = args.data_path,
                        checkpoint_path = args.checkpoint_path,
                        partition_name  = args.partition_name,
                        partition_ID    = args.partition_ID)
    #===========================