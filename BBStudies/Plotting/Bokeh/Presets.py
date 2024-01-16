
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import itertools
import scipy.stats as sciStats

# BOKEH
import bokeh.plotting as bk
import bokeh.models as bkmod
import bokeh.layouts as bklay
import bokeh.palettes as bkpalettes
import bokeh.util.hex as bkhex
import bokeh.transform as bktrfm
import bokeh.colors as bkcolors

# xsuite
import xtrack as xt
import xmask as xm
import xfields as xf
import xpart as xp



# BBStudies
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Physics.Constants as cst
import BBStudies.Plotting.Bokeh.Tools as bktools







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

    # Include BBHO in same reference
    # considered_strength = []
    for sequence in ['lhcb1','lhcb2']:
        bbho_info   = bktools.extract_bbho_info(collider[sequence],twiss[sequence])
        considered_strength.append(bblr_info.strength_x.abs().max())
        considered_strength.append(bblr_info.strength_y.abs().max())
    ref_bb = np.max(considered_strength)

    # Iterating over beams
    for sequence,baseline,beam_color,_direction in zip(['lhcb1','lhcb2'],[1.5,-1.5],['blue','red'],[-np.pi/2,np.pi/2]):
        beam       = sequence[-2:]
        lattice_df = bktools.extract_lattice_info(collider[sequence],twiss[sequence])
        bblr_info  = bktools.extract_bblr_info(collider[sequence],twiss[sequence])
        bbho_info  = bktools.extract_bbho_info(collider[sequence],twiss[sequence])
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
        for ee_type,color in zip(['dipole','quadrupole','sextupole','octupole','bblr_x','bblr_y','bbho_x','bbho_y'],['royalblue','firebrick','forestgreen','darkmagenta','black','deeppink','black','deeppink']):
            

            
            if 'bblr' in ee_type:
                # Adding BBLR contribution
                plane = ee_type.split('_')[-1]
                group = bblr_info.rename(columns={f'strength_{plane}':'knl'})
                # legend_label = f'BBLR (dQ{plane})'
                legend_label = f'BB (kick-{plane})'
            elif 'bbho' in ee_type:
                # Adding BBLR contribution
                plane = ee_type.split('_')[-1]
                group = bbho_info.rename(columns={f'strength_{plane}':'knl'})
                legend_label = f'BB (kick-{plane})' #ee_type.upper()
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
            elif 'bb' in ee_type:
                normalisation_y  = ref_bb
                if normalisation_y == 0:
                    normalisation_y = 1
                element_line_y   = element_line_y/normalisation_y
            else:
                normalisation_y  = np.max(np.abs(element_line_y))
                element_line_y   = element_line_y/normalisation_y

            # Source to pass element name to hover tool, then plot with varea
            source = bkmod.ColumnDataSource(pd.DataFrame({'name':np.repeat(group.name,4),'type':np.repeat(group.type,4),'x':element_line_x,'y':element_line_y+baseline,'baseline':element_baseline+baseline}))
            _varea = fig.varea(x='x',y1='baseline', y2='y', alpha=0.6,fill_color=color,source=source,legend_label=legend_label)
            
            # Adding circles for some elements
            if ee_type in ['sextupole','octupole','bblr_x','bblr_y','bbho_x','bbho_y']:
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


#=========================================================================================================================
def make_scatter_fig(df,xy,alpha=0.3,slider=None,title=None,width=2000,height=400,padding = None):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = height, 
                    width           = width,
                    title           = title, 
                    tools           = "box_zoom,pan,reset,save,hover,wheel_zoom,crosshair",
                    active_drag     = "pan",
                    active_scroll   = "wheel_zoom",
                    active_inspect  = None,
                    toolbar_location= "right")


    # Saving tools to tags
    _palette = bkpalettes.Spectral10
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools},
                {'palette':_palette}]
    fig.tags[0]['HoverTool'].update(tooltips = [('Particle', '@particle'),(f'Coordinates', '($x,$y)')])
    #=====================================


    # Creating source
    #=====================================
    x,y = xy
    if slider is not None:
        to_source = bktools.source_from_groupby(df,by='Chunk ID',columns = [x,y])
        to_source.insert(0,'particle',df.groupby('Chunk ID').get_group(0).particle)
        source = bkmod.ColumnDataSource(to_source)
    else:
        source = bkmod.ColumnDataSource(df[['particle',x,y]])

    #=====================================


    # Plotting
    #=====================================
    if slider is not None:
        fig.scatter(f'{x}:active',f'{y}:active', alpha=alpha, source=source)
    else:
        fig.scatter(x,y, alpha=alpha, source=source)
    #=====================================

        # Adding slider callback
    #=====================================
    if slider is not None:
        callback = bkmod.callbacks.CustomJS(args=dict(slider = slider,source = source), code=f"""
                    //=========================================================
                    source.data['{x}:active'] = source.data['{x}:'+slider.value.toString()];
                    source.data['{y}:active'] = source.data['{y}:'+slider.value.toString()];
                    source.change.emit()
                    //=========================================================""")

        slider.js_on_change('value', callback)
    #=====================================

    # Axis and Legend
    #=====================================

    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y
    if padding is not None:
        fig.min_border_right = padding
        fig.min_border_left  = padding

    #=====================================

    return fig


#=========================================================================================================================
def make_intensity_fig(data,slider,title=None,width=2000,height=400,padding = None):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = height, 
                    width           = width,
                    title           = title, 
                    tools           = "box_zoom,pan,reset,save,wheel_zoom",
                    active_drag     = "box_zoom",
                    active_scroll   = "wheel_zoom",
                    toolbar_location= "right")


    # Saving tools to tags
    # _palette = bkpalettes.Viridis8
    _palette = bkpalettes.Spectral10
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools},
                {'palette':_palette}]

    # Putting legend outside
    fig.add_layout(bkmod.Legend(), 'right')
    #=====================================

    # Adding chunk selection rectangle
    chunk_df = data.data[['Chunk ID','start_at_turn','stop_at_turn']].groupby('Chunk ID').mean()
    chunk_df.insert(0,'x',(chunk_df['stop_at_turn']+chunk_df['start_at_turn'])/2)
    to_source = bktools.source_from_groupby(chunk_df,by='Chunk ID',columns = ['x'])
    block_width =(chunk_df['stop_at_turn']-chunk_df['start_at_turn']).max()
    to_source.insert(0,'width',block_width)
    to_source.insert(1,'height',10)
    to_source.insert(2,'y',0.5)
    
    # source = bkto_source
    source_chunk  = bkmod.ColumnDataSource(to_source)
    fig.rect(x='x:active', y='y', width='width', height='height',alpha=0.2,source=source_chunk)


    absolute_min = 1
    for coll_opening,color in zip([3,4,5,6,7,8,9,10][::-1],fig.tags[1]['palette']):
        # Creating source
        #=====================================
        intensity_df = data.compute_intensity(coll_opening=coll_opening)
        intensity_df = intensity_df[1:]
        intensity_df.insert(3,'Norm. Count',np.abs(intensity_df['count'])/intensity_df.loc[1,'count'])
        source       = bkmod.ColumnDataSource(intensity_df[['start_at_turn','Norm. Count']])
        absolute_min = min(absolute_min,intensity_df['Norm. Count'].min())
        #=====================================


        # Plotting
        #=====================================
        legend_opening = str(coll_opening).ljust(4-len(str(coll_opening)))
        fig.step(x='start_at_turn',y='Norm. Count', source=source,legend_label=f'{legend_opening} σ_coll, [I(0) = {str(intensity_df.loc[1,"count"]).ljust(7)} p+]',line_width=2,color=color)
        #=====================================

    # Slider


    callback = bkmod.callbacks.CustomJS(args=dict(slider = slider,source = source_chunk), code="""
                //=========================================================
                source.data['x:active'] = source.data['x:'+slider.value.toString()];
                source.change.emit()
                //=========================================================""")

    slider.js_on_change('value', callback)
    

    # Axis and Legend
    #=====================================

    fig.xaxis.axis_label = 'Turn'
    fig.yaxis.axis_label = r'Fraction of surviving particles'
    fig.x_range          = bkmod.Range1d(-100, data.data.start_at_turn.max()+block_width+100)
    fig.y_range          = bkmod.Range1d(0.98*absolute_min, 1.005)
    
    fig.legend.title     = r'Collimators Opening'
    fig.legend.click_policy="hide"

    if padding is not None:
        fig.min_border_right = padding
        fig.min_border_left  = padding
    

    #=====================================

    return fig


#=========================================================================================================================
def update_coll(_df,coll_opening = 5):
    
    # Resetting values
    #-----------------------------
    _width  = _df['width'].unique()[0]
    _height = _df['height'].unique()[0]
    _df['xs']   = 6*[np.array([0,0,_width,_width])]
    _df['ys']   = 6*[np.array([-_height/2,_height/2,_height/2,-_height/2])]

    # Distributing colls on x-axis
    #-----------------------------
    _df['xs'] += coll_opening*_df['sigma']
    _df.loc[_df.name.str.contains('left|bottom'),'xs']   *= -1

    # Rotating according to angle
    _x,_y  = np.stack(_df['xs']),np.stack(_df['ys'])
    _angle = np.stack(_df['angle'].apply(lambda angle: list(np.repeat(angle,4))))
    _x_rot = _x*np.cos(_angle) - _y*np.sin(_angle)
    _y_rot = _x*np.sin(_angle) + _y*np.cos(_angle)


    _df['xs'] = list(_x_rot)
    _df['ys'] = list(_y_rot)



    return _df

def make_ROI(x1,x2,y1,y2,s1,s2,collx1,collx2,colly1,colly2,colls1,colls2):
    _out_ROI_min  =((np.abs(x1)>collx1)|(np.abs(x2)>collx1) | 
                    (np.abs(y1)>colly1)|(np.abs(y2)>colly1) | 
                    (np.abs(s1)>colls1)|(np.abs(s2)>colls1))
    _out_ROI_max  =((np.abs(x1)>collx2)|(np.abs(x2)>collx2) | 
                    (np.abs(y1)>colly2)|(np.abs(y2)>colly2) | 
                    (np.abs(s1)>colls2)|(np.abs(s2)>colls2))
    
    ROI = ((_out_ROI_min) & (~_out_ROI_max))
    # Splitting in 3 planes
    ROI_x = ROI&((np.abs(x1)>collx1)|(np.abs(x2)>collx1))
    ROI_y = ROI&((np.abs(y1)>colly1)|(np.abs(y2)>colly1))
    ROI_s = ROI&(~ROI_x)&(~ROI_y)


    return ROI_x,ROI_y,ROI_s

def plane_condition(x1,x2,collx1,collx2):
    _out_ROI_min  =((np.abs(x1)>collx1)|(np.abs(x2)>collx1)) 
    _out_ROI_max  =((np.abs(x1)>collx2)|(np.abs(x2)>collx2))
    
    ROI = ((_out_ROI_min) & (~_out_ROI_max))

    return ROI

def make_collimation_fig(data,slider = None,title=None,width=2000,height=400,padding = None):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = height, 
                    width           = width,
                    title           = title, 
                    tools           = "box_zoom,pan,reset,save,hover,wheel_zoom",
                    active_drag     = "box_zoom",
                    active_scroll   = "wheel_zoom",
                    toolbar_location= "right")


    # Saving tools to tags
    # _palette = bkpalettes.Viridis8
    _palette = bkpalettes.Spectral10
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools},
                {'palette':_palette}]
    # fig.tags[0]['WheelZoomTool'].update(dimensions = 'height')
    # fig.tags[0]['HoverTool'].update(tooltips = [('Variable', '$name'),('s [m]','$x{0}'),(f'Value', '$y'),('Element','@name')])
    fig.tags[0]['HoverTool'].update(tooltips = [('Collimator [sigma_coll]','@opening'),('Count', '@{counts:active}')])

    # Putting legend outside
    # fig.add_layout(bkmod.Legend(), 'right')
    #=====================================

    coll_opening = 10
    coll_alpha = np.deg2rad(127.5)
    pipe_r     = cst.LHC_W_BEAM_SCREEN/2
    coll_df    = pd.DataFrame({'name'   :  ['H_left','H_right','V_top','V_bottom','S_top','S_bottom'],
                               'width'  :  pipe_r*np.ones(6),
                               'height' :2*pipe_r*np.ones(6),
                               'xs'     :6*[np.zeros(4)],
                               'ys'     :6*[np.zeros(4)],
                                'angle' :[0,0,np.pi/2,np.pi/2,coll_alpha,coll_alpha],
                                'sigma' :[data.sig_x_coll,data.sig_x_coll,data.sig_y_coll,data.sig_y_coll,data.sig_skew_coll,data.sig_skew_coll]})

        
    coll_df = update_coll(coll_df,coll_opening=coll_opening)
    source  = bkmod.ColumnDataSource(coll_df)
    
    fig.patches(xs='xs', ys='ys',alpha=1,color='gray',source=source)


    # Crop at beam pipe
    _x_pipe = np.array([-2*pipe_r] + list(np.linspace(-pipe_r,pipe_r,200)) + [2*pipe_r])
    _y_pipe = np.sqrt(pipe_r**2 - _x_pipe**2)
    _y_pipe[np.abs(_x_pipe)>pipe_r] = 0
    fig.varea(x=_x_pipe,y1=2*pipe_r*np.ones(len(_x_pipe)),y2=_y_pipe  ,color='white',alpha=1)
    fig.varea(x=_x_pipe,y2=-2*pipe_r*np.ones(len(_x_pipe)),y1=-_y_pipe,color='white',alpha=1)


        # Creating Hextiles
    #=====================================
    coll_values = np.linspace(0,10,50)
    n_bins      = 300
    coll_sig = np.max([data.sig_x_coll,data.sig_y_coll])
    XX,YY    = np.meshgrid(np.linspace(-coll_values[-1]*coll_sig,coll_values[-1]*coll_sig,n_bins),
                            np.linspace(-coll_values[-1]*coll_sig,coll_values[-1]*coll_sig,n_bins))

    _size        = np.min(np.abs(np.diff(XX.flatten())))
    _orientation = 'pointytop'

    hextiles = bkhex.hexbin(XX.flatten(), YY.flatten(), _size)
    _x,_y    = bkhex.axial_to_cartesian(hextiles.q,hextiles.r,size=_size,orientation=_orientation)
    theta_unskew= -np.deg2rad(127.5)
    _x_skew       = _x*np.cos(theta_unskew) - _y*np.sin(theta_unskew)
    _y_skew       = _x*np.cos(-theta_unskew) - _y*np.sin(-theta_unskew)
    hextiles.insert(0,'x',_x)
    hextiles.insert(1,'y',_y)
    hextiles.insert(2,'x_skew',_x_skew)
    hextiles.insert(3,'y_skew',_y_skew)
    hextiles.insert(4,'opening',np.nan)
    hextiles.counts = np.nan
    hextiles.rename(columns={'counts':'counts:active'},inplace=True)
    #=====================================



    _sig_x    = data.sig_x_coll
    _sig_y    = data.sig_y_coll
    _sig_skew = data.sig_skew_coll
    # Looping over chunks

    # name  = 0
    # group = data.data.groupby('Chunk ID').get_group(name)
    # group = group.set_index('particle').loc[[10]]
    

    for name,group in data.data.groupby('Chunk ID'):
        hextiles.insert(len(hextiles.columns),f'counts:{name}',np.nan)
        for coll_min,coll_max in zip(coll_values[:-1],coll_values[1:]):
            
            # Identifying Hex in ROI
            
            hex_x,hex_y,hex_s  = make_ROI( hextiles['x'],hextiles['x'],
                                            hextiles['y'],hextiles['y'],
                                            hextiles['x_skew'],hextiles['y_skew'],
                                            coll_min*_sig_x,coll_max*_sig_x,
                                            coll_min*_sig_y,coll_max*_sig_y,
                                            coll_min*_sig_skew,coll_max*_sig_skew)
            hextiles.loc[(hex_x|hex_y|hex_s),f'opening'] = np.mean([coll_min,coll_max])


            # Counts per collimators
            #------------------------------------
            count_x = plane_condition(group['x_min'],group['x_max'],coll_min*_sig_x,coll_max*_sig_x)
            count_y = plane_condition(group['y_min'],group['y_max'],coll_min*_sig_y,coll_max*_sig_y)
            count_s = plane_condition(group['skew_min'],group['skew_max'],coll_min*_sig_skew,coll_max*_sig_skew)
            
            # Initializing counts
            #------------------------------------
            if count_x.sum()+count_y.sum()+count_s.sum() != 0:
                hextiles.loc[(hex_x|hex_y|hex_s),f'counts:{name}'] = 0

            # Show plane by plane:
            #------------------------------------
            hextiles.loc[hex_x,f'counts:{name}'] = count_x.sum()
            hextiles.loc[hex_y,f'counts:{name}'] = count_y.sum()
            hextiles.loc[hex_s,f'counts:{name}'] = count_s.sum()

            # Cleaning counts
            #------------------------------------
            if count_x.sum()+count_y.sum()+count_s.sum() == 0:
                hextiles.loc[(hex_x|hex_y|hex_s),f'counts:{name}'] = np.nan


    hextiles['counts:active'] = hextiles['counts:0']

    data_col  = [col for col in hextiles.columns if 'counts:' in col]
    max_value = np.max(hextiles[data_col].sum(axis=1)) 
    source    = bkmod.ColumnDataSource(hextiles[['q','r','opening']+data_col])
    nan_color = bkcolors.RGB(255,255,255,a=0)
    # cmap      = bktrfm.linear_cmap('counts:active', 'Plasma256', 0, 1500,nan_color=nan_color)
    cmap      = bktrfm.log_cmap('counts:active', 'Magma256', 1, max_value,nan_color=nan_color)
    fig.hex_tile(q="q", r="r", size= _size, line_color=None, source=source,alpha=1,fill_color=cmap)


    color_bar = bkmod.ColorBar(title='Counts',color_mapper=cmap['transform'])
    fig.add_layout(color_bar, 'right')


    # Slider
    #=====================================
    callback = bkmod.callbacks.CustomJS(args=dict(slider = slider,source = source), code="""
                //=========================================================
                source.data['counts:active'] = source.data['counts:'+slider.value.toString()];
                source.change.emit()
                //=========================================================""")

    slider.js_on_change('value', callback)

    # Axis and Legend
    #=====================================

    fig.xaxis.axis_label = r'$$x\ [\text{m}]$$'
    fig.yaxis.axis_label = r'$$y\ [\text{m}]$$'
    if padding is not None:
        fig.min_border_right = padding
        fig.min_border_left  = padding

    #=====================================

    return fig




def make_JxJy_fig(data,slider = None,title=None,width=2000,height=400,padding = None):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = height, 
                    width           = width,
                    title           = title, 
                    tools           = "box_zoom,pan,reset,save,hover,wheel_zoom",
                    active_drag     = "pan",
                    active_scroll   = "wheel_zoom",
                    toolbar_location= "right")

    # No grid
    # fig.grid.visible = False

    # Saving tools to tags
    # _palette = bkpalettes.Viridis8
    _palette = bkpalettes.Spectral10
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools},
                {'palette':_palette}]
    # fig.tags[0]['WheelZoomTool'].update(dimensions = 'height')
    # fig.tags[0]['HoverTool'].update(tooltips = [('Variable', '$name'),('s [m]','$x{0}'),(f'Value', '$y'),('Element','@name')])
    fig.tags[0]['HoverTool'].update(tooltips = [('Coordinates','(@x,@y)'),('Count', '@{counts:active}')])

    # Putting legend outside
    # fig.add_layout(bkmod.Legend(), 'right')
    #=====================================

    # Extraction action
    J_df = data.checkpoint[['Chunk ID','turn','particle']]
    J_df.insert(3,'Jx/emitt',1/2 * (data.checkpoint_sig.x_sig**2 + data.checkpoint_sig.px_sig**2))
    J_df.insert(4,'Jy/emitt',1/2 * (data.checkpoint_sig.y_sig**2 + data.checkpoint_sig.py_sig**2))
    # J_df.dropna(inplace=True)

    # Making Hextile grid
    J_min  = 0
    J_max  = 50
    n_bins = 300

    _size        = (J_max-J_min)/n_bins
    _orientation = 'pointytop'


    # Creating hextile template
    XX,YY    = np.meshgrid(np.arange(J_min,J_max+0.9*_size,_size),np.arange(J_min,J_max+0.9*_size,_size))
    hextiles_template = bkhex.hexbin(XX.flatten(),YY.flatten(), size=_size,orientation=_orientation)
    hextiles_template['counts']  = 0
    _x,_y    = bkhex.axial_to_cartesian(hextiles_template.q,hextiles_template.r,size=_size,orientation=_orientation)
    hextiles_template.insert(0,'x',_x)
    hextiles_template.insert(1,'y',_y)
    hextiles_template = hextiles_template.rename(columns={'counts':'counts:active'}).set_index(['q','r'])
    

    # Looping over chunks
    for name,group in J_df.groupby('Chunk ID'):
        # Forcing corner values to have same grid.
        within_limits = (group['Jx/emitt'] < J_max)&(group['Jy/emitt'] < J_max)
        data_x = np.array(list(group['Jx/emitt'][within_limits]) + [J_min,J_max])
        data_y = np.array(list(group['Jy/emitt'][within_limits]) + [J_min,J_max])
        _hex = bkhex.hexbin(data_x,data_y, size=_size,orientation=_orientation)

        # Removing corner values
        _hex = _hex[1:]

        # Adding lost particles counts
        if sum(~within_limits) != 0:
            _hex.loc[len(_hex),'counts'] = sum(~within_limits)
        else:
            _hex.loc[len(_hex),'counts'] = np.nan

        # Adding chunk ID
        hextiles_template.insert(name+1,f'counts:{name}',_hex.set_index(['q','r'])['counts'])


    # setting empty bins to 0
    # hextiles_template = hextiles_template.fillna(0).reset_index()
    # hextiles_template = hextiles_template.reset_index()
    hextiles_template.reset_index(inplace=True)
    hextiles_template['counts:active'] = hextiles_template['counts:0']



    source    = bkmod.ColumnDataSource(hextiles_template)
    nan_color = bkcolors.RGB(255,255,255,a=0)
    cmap      = bktrfm.linear_cmap('counts:active', 'Viridis256', 0, data.n_parts//n_bins,nan_color=nan_color)
    fig.hex_tile(q="q", r="r", size= _size, line_color=None, source=source,alpha=1,fill_color=cmap)

    hline = fig.hspan(y=[0], line_width=[2], line_color="black")
    vline = fig.vspan(x=[0], line_width=[2], line_color="black")
    hline.level = 'underlay'
    vline.level = 'underlay'

    # source = bkmod.ColumnDataSource(J_df)
    # fig.scatter('Jx/emitt','Jy/emitt', alpha=0.6, source=source)


    color_bar = bkmod.ColorBar(title='Counts',color_mapper=cmap['transform'])
    fig.add_layout(color_bar, 'right')


    # Slider
    #=====================================
    callback = bkmod.callbacks.CustomJS(args=dict(slider = slider,source = source), code="""
                //=========================================================
                source.data['counts:active'] = source.data['counts:'+slider.value.toString()];
                source.change.emit()
                //=========================================================""")

    slider.js_on_change('value', callback)

    # Axis and Legend
    #=====================================

    fig.xaxis.axis_label = 'Jx/emitt'
    fig.yaxis.axis_label = 'Jy/emitt'
    if padding is not None:
        fig.min_border_right = padding
        fig.min_border_left  = padding

    #=====================================

    return fig