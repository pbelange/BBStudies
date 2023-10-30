
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

# xsuite
import xtrack as xt
import xmask as xm
import xfields as xf
import xpart as xp

# Custom imports
import bokeh_tools as bktools

# BBStudies
import sys
sys.path.append('/Users/pbelanger/ABPLocal/BBStudies')
sys.path.append('../../../../')
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.InteractionPoint as inp
import BBStudies.Physics.Detuning as tune
import BBStudies.Plotting.BBPlots as bbplt
import BBStudies.Physics.Base as phys
import BBStudies.Physics.Constants as cst







def make_LHC_Layout_Fig(collider,twiss,width=2000,height=400):
    """
    Should provide collider and twiss for both beams ['lhcb1','lhcb2']
    """
    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = height, 
                    width           = width,
                    title           = "Lattice", 
                    tools           = "box_zoom,pan,reset,save,hover,wheel_zoom",
                    active_drag     = "box_zoom",
                    active_scroll   = "wheel_zoom",
                    toolbar_location= "right")

    # No grid 
    fig.grid.visible = False

    # Saving tools to tags
    fig.tags    = [ {str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools},
                    {'palette':bkpalettes.Spectral11 + bkpalettes.PiYG11 + bkpalettes.RdPu9 + bkpalettes.RdGy11}]
    fig.tags[0]['BoxZoomTool'].update(dimensions = 'width')
    fig.tags[0]['WheelZoomTool'].update(dimensions = 'height')
    fig.tags[0]['HoverTool'].update(tooltips = [('Type','@type'),('Element','@name'),('s [m]','$x{0}')])
    #=====================================

    # Fixing common reference for BBLR
    considered_strength = []
    for sequence in ['lhcb1','lhcb2']:
        bblr_info   = bktools.extract_bblr_info(collider[sequence],twiss[sequence])
        considered_strength.append(bblr_info.strength_x.abs().max())
        considered_strength.append(bblr_info.strength_y.abs().max())
    ref_bblr = np.max(considered_strength)

    # Iterating over beams
    for sequence,baseline,beam_color,_direction in zip(['lhcb1','lhcb2'],[1.5,-1.5],['blue','red'],[-np.pi/2,np.pi/2]):
        beam       = sequence[-2:]
        lattice_df = bktools.extract_lattice_info(collider[sequence],twiss[sequence])
        bblr_info  = bktools.extract_bblr_info(collider[sequence],twiss[sequence])
        collimator_info = bktools.extract_collimation_info(collider[sequence],twiss[sequence])

        tw         = twiss[sequence]

        # Adding 0 line
        #------------------------------------
        fig.hspan(y=[baseline],line_width=[1], line_color="black")
        fig.hspan(y=[-1.05+baseline,1.05+baseline],line_width=[2,2], line_color=beam_color)
        fig.varea(x=[tw.s.min(),tw.s.max()],y1=[-1.05+baseline,-1.05+baseline], y2=[1.05+baseline,1.05+baseline], alpha=0.05,fill_color=beam_color)   

        # Adding beam direction
        #------------------------------------
        # s_along = np.arange(tw.s.min(),tw.s.max(),200)
        # fig.hspan(y=[-1.3*baseline/np.abs(baseline)+baseline],line_width=[2], line_color=beam_color,line_dash='dashed')
        # fig.triangle(x = s_along,y=(-1.3*baseline/np.abs(baseline)+baseline)*np.ones(len(s_along)),size=8,angle=_direction, color=beam_color,line_dash='dashed',line_width=2,alpha=0.5)
        
        

        # Making step function for each element
        #------------------------------------
        for ee_type,color in zip(['dipole','quadrupole','sextupole','octupole','bblr_x','bblr_y'],['royalblue','firebrick','forestgreen','darkmagenta','black','orange']):
            

            
            if 'bblr' in ee_type:
                # Adding BBLR contribution
                plane = ee_type.split('_')[-1]
                group = bblr_info.rename(columns={f'strength_{plane}':'knl'})
                legend_label = f'BBLR (dQ{plane})' #ee_type.upper()
            else:
                group   = lattice_df.groupby('type').get_group(ee_type)
                legend_label = ee_type.capitalize()

            element_line_x = [[_entry,_entry,_exit,_exit] for _entry,_exit in zip(group.s_entry,group.s_exit)] 
            element_line_y = [[0,_knl,_knl,0] for _knl in group.knl] 

            element_df = pd.DataFrame({'x':np.array(element_line_x).flatten(),'y':np.array(element_line_y).flatten()})    
            element_line_x = np.array(element_line_x).flatten()
            element_line_y = np.array(element_line_y).flatten()
            element_baseline = np.zeros(len(element_line_x))

            if ee_type == 'dipole':
                element_line_y   = 0.2*np.divide(element_line_y,element_line_y,out=np.zeros(len(element_line_y)),where=element_line_y!=0)
                element_baseline = -element_line_y 
            elif ee_type == 'bblr':
                normalisation_y  = ref_bblr
                element_line_y   = element_line_y/normalisation_y
            else:
                normalisation_y  = np.max(np.abs(element_line_y))
                element_line_y   = element_line_y/normalisation_y

            # Source to pass element name to hover tool, then plot with varea
            source = bkmod.ColumnDataSource(pd.DataFrame({'name':np.repeat(group.name,4),'type':np.repeat(group.type,4),'x':element_line_x,'y':element_line_y+baseline,'baseline':element_baseline+baseline}))
            _varea = fig.varea(x='x',y1='baseline', y2='y', alpha=0.6,fill_color=color,source=source,legend_label=legend_label)
            
            # Adding circles for some elements
            if ee_type in ['sextupole','octupole','bblr_x','bblr_y']:
                fig.circle(group.s,group.knl/normalisation_y + baseline, size=3, color=color, alpha=0.5,legend_label=legend_label)


        # Adding collimators
        #------------------------------------
        for ee_type in ['Primary','Secondary','Absorber','Tertiary','BBCW UP','BBCW DOWN']:
            color = beam_color
            if 'BBCW' not in ee_type:
                group   = collimator_info.groupby('type').get_group(ee_type)
                legend_label = ee_type.capitalize()
            else:
                # Adding BBCW, splitting in UP and DOWN
                bbcw_group    = collimator_info.groupby('type').get_group('BBCW')
                if ee_type == 'BBCW UP':
                    group   = bbcw_group.loc[bbcw_group.name.apply(lambda name:name[4] in ['t','e'])]
                else:
                    group  = bbcw_group.loc[bbcw_group.name.apply(lambda name:name[4] in ['b','i'])]
                legend_label = ee_type.upper()
                color = 'limegreen'

            element_line_x = [[_entry,_entry,_exit,_exit] for _entry,_exit in zip(group.s_entry,group.s_exit)] 
            element_line_y = [[0,_h,_h,0] for _h in group.strength] 

            element_df = pd.DataFrame({'x':np.array(element_line_x).flatten(),'y':np.array(element_line_y).flatten()})    
            element_line_x = np.array(element_line_x).flatten()
            element_line_y = np.array(element_line_y).flatten()
            element_baseline = np.zeros(len(element_line_x))

            # Source to pass element name to hover tool, then plot with varea
            source = bkmod.ColumnDataSource(pd.DataFrame({'name':np.repeat(group.name,4),'type':np.repeat(group.type,4),
                                            'x':element_line_x,
                                            'y':1.05+baseline,
                                            'baseline':(1.05-element_line_y)+baseline,
                                            'y_bottom':(-1.05+element_line_y)+baseline,
                                            'baseline_bottom':-1.05+baseline}))
            
            
            if 'BBCW' not in ee_type:
                _varea = fig.varea(x='x',y1='baseline', y2='y', alpha=1,fill_color=color,source=source)
                _varea = fig.varea(x='x',y1='baseline_bottom', y2='y_bottom', alpha=1,fill_color=color,source=source)
            else:
                if ee_type == 'BBCW UP':
                    _varea = fig.varea(x='x',y1='baseline', y2='y', alpha=1,fill_color=color,source=source)
                else:
                    _varea = fig.varea(x='x',y1='baseline_bottom', y2='y_bottom', alpha=1,fill_color=color,source=source)



        if sequence == 'lhcb1':

            fig.legend.location    = "top_left"
            fig.legend.click_policy= "hide"


            # Adding  IP locations on top axis
            sequence = 'lhcb1'
            IPs    = tw.loc[tw.name.isin([f'ip{i}' for i in range(1,8+1)]),['name','s']]
            arcs_s = tw.loc[tw.name.isin([f's.arc.{arc}.{beam}' for arc in [f'{np.roll(range(1,9),-i)[0]}{np.roll(range(1,9),-i)[1]}' for i in range(0,8)]]),['name','s']]
            arcs_e = tw.loc[tw.name.isin([f'e.arc.{arc}.{beam}' for arc in [f'{np.roll(range(1,9),-i)[0]}{np.roll(range(1,9),-i)[1]}' for i in range(0,8)]]),['name','s']]
            arc_mids = pd.DataFrame({'name':[f'ARC {arc}' for arc in arcs_s.sort_values('name').name.apply(lambda arc: arc.split('.')[2])],
                                    's':(arcs_s.sort_values('name').s.values + arcs_e.sort_values('name').s.values)/2})

            IPs.loc[:,'name'] = IPs.name.str.upper()
            _label_overrides = IPs.set_index('s')['name'].to_dict()
            _label_overrides.update(arc_mids.set_index('s')['name'].to_dict())


    ax,axis_name = bktools.new_x_axis(fig,axis_name='IPs',side='above')
    fig.extra_x_ranges[axis_name] = fig.x_range
    fig.xaxis[0].ticker = sorted(list(_label_overrides.keys()))
    fig.xaxis[0].major_label_overrides = _label_overrides
    fig.xaxis[0].major_tick_line_width = 3
    fig.xaxis[0].major_tick_in = 5

    fig.yaxis[0].ticker = [1.5,-1.5]
    fig.yaxis[0].major_label_overrides = {1.5:'Beam 1',-1.5:'Beam 2'}


    return fig  
#=========================================================================================================================




#=========================================================================================================================
def make_Twiss_Fig(collider,twiss,width=2000,height=400,twiss_columns = None):



    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = height, 
                    width           = width,
                    title           = "Twiss parameters", 
                    tools           = "box_zoom,pan,reset,save,hover,wheel_zoom",
                    active_drag     = "box_zoom",
                    active_scroll   = "wheel_zoom",
                    toolbar_location= "right")


    # Saving tools to tags
    interlaced_palette = list(itertools.chain(*zip(bkpalettes.Category20c[20],bkpalettes.Category20b[20])))
    fig.tags    = [ {str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools},
                    {'palette':interlaced_palette}]#bkpalettes.Category20c + bkpalettes.Spectral11 + bkpalettes.PiYG11 + bkpalettes.RdPu9 + bkpalettes.RdGy11}]
    fig.tags[0]['WheelZoomTool'].update(dimensions = 'height')
    # fig.tags[0]['HoverTool'].update(tooltips = [('Variable', '$name'),('s [m]','$x{0}'),(f'Value', '$y'),('Element','@name')])
    fig.tags[0]['HoverTool'].update(tooltips = [('Variable', '$name'),('s [m]','@s'),(f'Value', '$y'),('Element','@name')])
    #=====================================


    # Iterating over beams, adding all twiss variables
    legends = {}
    for beam,tw,ls,legend_colour in zip(['b1','b2'],[twiss['lhcb1'],twiss['lhcb2']],['solid',[3,1]],['blue','red']):
        if twiss_columns is None:
            source = bkmod.ColumnDataSource(tw.drop(columns=['W_matrix']))
        else:
            source = bkmod.ColumnDataSource(tw[['name','s']+twiss_columns])
        
        
        _keys      = [col for col in source.column_names if col not in ['name','s','index']]
        _line_list = []
        for key,color in zip(_keys,fig.tags[1]['palette']):
            
            # Setting default visible lines: betx and bety for b1 
            if (key in ['betx','bety'])&(beam=='b1'):
                _visible = True
            else:
                _visible = False
            
            # Plotting line
            _line = fig.line(x='s',y=key, line_width=2, color=color, alpha=0.8, line_dash=ls,name=key,legend_label=key,source=source,visible=_visible)
            _line_list.append((key,[_line]))

        # Creating separate legends for each beam
        legends[beam] = bkmod.Legend(items=_line_list,click_policy="hide",title=f'Beam {beam[-1]}')
        
        legends[beam].border_line_width = 2
        legends[beam].border_line_color = legend_colour
        legends[beam].border_line_alpha = 0.8




    fig.add_layout(legends['b1'], 'right')
    fig.add_layout(legends['b2'], 'right')
    
    # Removing original legend
    fig.legend[-1].visible=False
    
    # Specifying axis
    fig.xaxis.axis_label = "Position, s [m]"
    fig.yaxis.axis_label = "Value"

    # Setting auto-scaling of y-axis to only visible glyphs
    fig.y_range.only_visible = True
    
    return fig