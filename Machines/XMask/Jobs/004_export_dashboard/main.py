
import numpy as np
import pandas as pd
from pathlib import Path
import gc

# BOKEH
import bokeh.plotting as bk
import bokeh.models as bkmod
import bokeh.layouts as bklay
import bokeh.palettes as bkpalettes


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



# Setting default values
#------------------------------------------------
_default_fig_width  = 1500
_default_fig_height = 400
_default_fig_pad    = 100



# Importing Collider and Twiss
#-------------------------------------
collider = xt.Multiline.from_json('../001_configure_collider/zfruits/collider_001.json')
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
#-------------------------------------


# Adding space to twiss fig for the legend, then adding pad
#-------------------------------------
BOKEH_FIGS['twiss'].height = 3*_default_fig_height
BOKEH_FIGS['twiss'].min_border_bottom = 2*_default_fig_height

for name,_fig in BOKEH_FIGS.items():
    if 'widget' in name.lower():
        continue
    _fig.min_border_left  = _default_fig_pad
    _fig.min_border_right = _default_fig_pad
#-------------------------------------

# Final layout
#=====================================
HTML_LAYOUT = bklay.column(BOKEH_FIGS['lattice'],BOKEH_FIGS['twiss'])
#=====================================


# Exporting to HTML
#=====================================
bktools.export_HTML(HTML_LAYOUT,'zfruits/collider_rendering.html',f'Collider rendering')
#=====================================






    


