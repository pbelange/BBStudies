
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import itertools
import importlib
from IPython.display import clear_output

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
import xobjects as xo

# BBStudies
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.Utils as xutils
import BBStudies.Physics.Constants as cst
import BBStudies.Plotting.Bokeh.Tools as bktools
import BBStudies.Plotting.Bokeh.Presets as bkpresets
import BBStudies.Physics.Base as phys


# ==================================================================================================
# --- Functions to load collider with a given context
# ==================================================================================================
def load_collider(collider_path = '../001_configure_collider/zfruits/collider_001.json'):

    collider = xt.Multiline.from_json(collider_path)
    context = xo.ContextCpu(omp_num_threads='auto')

    return collider,context



# ==================================================================================================
# --- Functions to plot resulting distribution
# ==================================================================================================
# Setting default values
#=====================================
_default_fig_width  = 1500
_default_fig_height = 400

_top_tab_height     = 300
_bot_tab_height     = 500
_default_fig_pad    = 100
#=====================================

def particles_to_HTML(particles,coordinates,collider,config,nemitt,rfbucket,export_path):

    BOKEH_FIGS  = {}
    
    _default_fig_width  = 1800
    _bot_tab_height     = 400
    padding             = 20 
    adjustment          = 0


    # Normalized space
    #=====================================
    data = pd.DataFrame({'particle' :particles.particle_id,
                         'x_sig'    :coordinates[0,:],
                         'px_sig'   :coordinates[1,:],
                         'y_sig'    :coordinates[2,:],
                         'py_sig'   :coordinates[3,:],
                         'zeta_sig' :coordinates[4,:],
                         'pzeta_sig':coordinates[5,:]})

    BOKEH_FIGS['n x-y'] = bkpresets.make_scatter_fig(data,xy=('x_sig','y_sig'),title='x-y norm. transv. space',width=int(_default_fig_width/4.5)+adjustment,height=_bot_tab_height,padding=0)
    BOKEH_FIGS['n x-px'] = bkpresets.make_scatter_fig(data,xy=('x_sig','px_sig'),title='x norm. phase space',width=int(_default_fig_width/4.5),height=_bot_tab_height,padding=0)
    BOKEH_FIGS['n y-py'] = bkpresets.make_scatter_fig(data,xy=('y_sig','py_sig'),title='y norm. phase space',width=int(_default_fig_width/4.5),height=_bot_tab_height,padding=0)
    BOKEH_FIGS['n zeta-pzeta'] = bkpresets.make_scatter_fig(data,xy=('zeta_sig','pzeta_sig'),title='zeta norm. phase space',width=int(_default_fig_width/4.5),height=_bot_tab_height,padding=0)


    BOKEH_FIGS['n x-px'].min_border_left  = padding
    _lim = data.loc[:,data.columns[1:5]].abs().max().max()
    _lim = np.ceil(1.1*_lim)
    bktools.set_aspect(BOKEH_FIGS['n x-y']       , x_lim=(-_lim,_lim),y_lim=(-_lim,_lim), aspect=1, margin=padding-adjustment)
    bktools.set_aspect(BOKEH_FIGS['n x-px']       , x_lim=(-_lim,_lim),y_lim=(-_lim,_lim), aspect=1, margin=0)
    bktools.set_aspect(BOKEH_FIGS['n y-py']       , x_lim=(-_lim,_lim),y_lim=(-_lim,_lim), aspect=1, margin=0)

    _lim = data.loc[:,data.columns[5:]].abs().max().max()
    _lim = np.ceil(1.1*_lim)
    bktools.set_aspect(BOKEH_FIGS['n zeta-pzeta'] , x_lim=(-_lim,_lim),y_lim=(-_lim,_lim), aspect=1, margin=0)

    BOKEH_FIGS['n x-y'].xaxis.axis_label = r'$$\tilde x /\sqrt{\varepsilon_{x}}$$'
    BOKEH_FIGS['n x-y'].yaxis.axis_label = r'$$\tilde y /\sqrt{\varepsilon_{x}}$$'

    BOKEH_FIGS['n x-px'].xaxis.axis_label = r'$$\tilde x /\sqrt{\varepsilon_{x}}$$'
    BOKEH_FIGS['n x-px'].yaxis.axis_label = r'$$\tilde p_x /\sqrt{\varepsilon_{x}}$$'

    BOKEH_FIGS['n y-py'].xaxis.axis_label = r'$$\tilde y /\sqrt{\varepsilon_{y}}$$'
    BOKEH_FIGS['n y-py'].yaxis.axis_label = r'$$\tilde p_y /\sqrt{\varepsilon_{y}}$$'

    BOKEH_FIGS['n zeta-pzeta'].xaxis.axis_label = r'$$\tilde \zeta /\sqrt{\varepsilon_{\zeta}}$$'
    BOKEH_FIGS['n zeta-pzeta'].yaxis.axis_label = r'$$\tilde p_\zeta /\sqrt{\varepsilon_{\zeta}}$$'


    norm_qp = bklay.gridplot([[BOKEH_FIGS['n x-y'],BOKEH_FIGS['n x-px'] ,BOKEH_FIGS['n y-py'] ,BOKEH_FIGS['n zeta-pzeta']]],toolbar_location='right')
    #=====================================




    # REAL SPACE
    #=====================================
    data = pd.DataFrame({'particle':particles.particle_id,'x':particles.x,'px':particles.px,'y':particles.y,'py':particles.py,'zeta':particles.zeta,'pzeta':particles.pzeta})

    
    BOKEH_FIGS['x-y']       = bkpresets.make_scatter_fig(data,xy=('x','y'),title='x-y transv. space',width=int(_default_fig_width/4.5)+adjustment,height=_bot_tab_height,padding=0)
    BOKEH_FIGS['x-px']      = bkpresets.make_scatter_fig(data,xy=('x','px'),title='x phase space',width=int(_default_fig_width/4.5),height=_bot_tab_height,padding=0)
    BOKEH_FIGS['y-py']      = bkpresets.make_scatter_fig(data,xy=('y','py'),title='y phase space',width=int(_default_fig_width/4.5),height=_bot_tab_height,padding=0)
    BOKEH_FIGS['zeta-pzeta']= bkpresets.make_scatter_fig(data,xy=('zeta','pzeta'),title='zeta phase space',width=int(_default_fig_width/4.5),height=_bot_tab_height,padding=0)
    BOKEH_FIGS['x-px'].min_border_left  = padding

    # Custom JavaScript to format tick values   
    chatGPT_tick = bkmod.FuncTickFormatter(code="""
                                                    function roundToSignificantDigits(num, n) {
                                                        if(num == 0) {
                                                            return 0;
                                                        }
                                                        var d = Math.ceil(Math.log10(num < 0 ? -num: num));
                                                        var power = n - d;
                                                        var magnitude = Math.pow(10, power);
                                                        var shifted = Math.round(num * magnitude);
                                                        return shifted / magnitude;
                                                    }
                                                    var roundedTick = roundToSignificantDigits(tick, 3); // Adjust '3' to your preferred number of significant digits
                                                    return roundedTick.toExponential();
                                                """)

    BOKEH_FIGS['x-y'].xaxis.axis_label = r'$$x$$'
    BOKEH_FIGS['x-y'].yaxis.axis_label = r'$$y$$'

    BOKEH_FIGS['x-px'].xaxis.axis_label = r'$$x$$'
    BOKEH_FIGS['x-px'].yaxis.axis_label = r'$$p_x$$'

    BOKEH_FIGS['y-py'].xaxis.axis_label = r'$$y$$'
    BOKEH_FIGS['y-py'].yaxis.axis_label = r'$$p_y$$'

    BOKEH_FIGS['zeta-pzeta'].xaxis.axis_label = r'$$\zeta$$'
    BOKEH_FIGS['zeta-pzeta'].yaxis.axis_label = r'$$p_\zeta$$'

    BOKEH_FIGS['x-y'].xaxis.formatter = chatGPT_tick
    BOKEH_FIGS['x-px'].xaxis.formatter = chatGPT_tick
    BOKEH_FIGS['y-py'].xaxis.formatter = chatGPT_tick
    BOKEH_FIGS['zeta-pzeta'].xaxis.formatter = chatGPT_tick
    BOKEH_FIGS['x-y'].yaxis.formatter = chatGPT_tick
    BOKEH_FIGS['x-px'].yaxis.formatter = chatGPT_tick
    BOKEH_FIGS['y-py'].yaxis.formatter = chatGPT_tick
    BOKEH_FIGS['zeta-pzeta'].yaxis.formatter = chatGPT_tick


    color = 'mediumvioletred'
    ls    = 'solid'
    label = f'RF Bucket'

    config_J001 = collider.metadata['config_J001']
    sigma_z     = config_J001['config_collider']['config_beambeam'][f'sigma_z']

    for zcut in list(np.linspace(0.001,rfbucket.zeta_max,10)) + [sigma_z,2*sigma_z,3*sigma_z]:
        zeta_vec,delta_vec = rfbucket.invariant(zcut,npoints = 1000)

        if zcut/sigma_z in [1,2,3]:
            color = 'black'
            ls    = 'dotted'
            label = f'σ'
        
        line_top = BOKEH_FIGS['zeta-pzeta'].line(x=zeta_vec,y=delta_vec, line_width=4, color=color, alpha=0.4, line_dash=ls, legend_label=label)
        line_bot = BOKEH_FIGS['zeta-pzeta'].line(x=zeta_vec,y=-delta_vec, line_width=4, color=color, alpha=0.4, line_dash=ls, legend_label=label)
        line_top.level = 'overlay'
        line_bot.level = 'overlay'


    qp = bklay.gridplot([[BOKEH_FIGS['x-y'],BOKEH_FIGS['x-px'] ,BOKEH_FIGS['y-py'] ,BOKEH_FIGS['zeta-pzeta']]],toolbar_location='right')




    # IMPORTING COLLIDER
    # Importing Twiss
    #-------------------------------------
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


    # Adding info
    #=====================================
    metadata = {'Name'                  : config['particles']['name'],
                'Number of particles'   : f'{len(particles.x):,}',
                'Distribution type'     : config['particles']['type'],
                'nemitt_x'              : f'{nemitt[0]:.3e}',
                'nemitt_y'              : f'{nemitt[1]:.3e}',
                'nemitt_zeta'           : f'{nemitt[2]:.3e}',
                'r_scale'               : f'{config[config["particles"]["type"]]["r_scale"]}',
                '--------------------'  : '--------------------',
                'Collider' : Path(config['collider']['path']).stem,
                'Sequence' : config['collider']['sequence'],
                'Cycle at' : config['collider']['cycle_at'],
                }
    info = bktools.dict_to_HTML(metadata, header_text="Particles Info", header_margin_top=20, header_margin_bottom=0,margin=20,indent= 2,nested_scale = 0.98, max_width = _default_fig_width)
    #=====================================

    # Adding collider config
    #=====================================
    config_list = {}
    for key in collider.metadata.keys():
        _info = bktools.dict_to_HTML(collider.metadata[key], header_text=key, header_margin_top=20, header_margin_bottom=0,margin=20,indent= 2,nested_scale = 0.98, max_width = _default_fig_width)
        config_list[key] = _info
    #=====================================


    # Final layout
    #=====================================
    tab_margin = 20
    bottom_tabs = bkmod.Tabs(tabs=[bkmod.TabPanel(child=info, title="Info")])
    bottom_tabs = bkmod.Row(bottom_tabs, margin=(0,0,0,tab_margin))

    top_tabs = bkmod.Tabs(tabs=[bkmod.TabPanel(child=norm_qp, title="Norm. Phase Space"),
                                bkmod.TabPanel(child= qp    , title="Phys. Phase Space")])
    top_tabs = bkmod.Row(top_tabs, margin=(0,0,0,tab_margin))

    tab_margin  = 0
    global_tabs = bkmod.Tabs(tabs=[ bkmod.TabPanel(child=bklay.column(top_tabs,bottom_tabs), title="Particles"),
                                    bkmod.TabPanel(child=grid_collider, title="Collider Object")] + \
                                  [ bkmod.TabPanel(child=config_list[key], title=key) for key in config_list.keys()])
    global_tabs = bkmod.Row(global_tabs, margin=(tab_margin,tab_margin,tab_margin,tab_margin))
    HTML_LAYOUT = global_tabs
    #=====================================



    # Setting font size
    for _fig in BOKEH_FIGS.values():
        _fig.xaxis.axis_label_text_font_size = "15px"
        _fig.yaxis.axis_label_text_font_size = "15px"

    # Exporting to HTML
    #=====================================
    bktools.export_HTML(HTML_LAYOUT,export_path,f'Particles - {metadata["Name"]}')
    #=====================================




# ==================================================================================================
# --- Main function
# ==================================================================================================
def particle_dist(config = None,config_path = 'config.yaml'):

    # Loading config
    #==============================
    if config is None:
        config = xutils.read_YAML(config_path)
    #==============================

    # Preparing output folder
    #==============================
    for _path in [config['particles']['path']]:
        if _path is not None:
            xutils.mkdir(_path) 
    #==============================



    # LOADING COLLIDER
    #==============================
    print('LOADING COLLIDER')
    sequence        = config['collider']['sequence']
    ee_at_dict      = config['elements'][sequence]
    collider,context = load_collider(collider_path = config['collider']['path'])


    # Cycling line at_element
    line    = collider[sequence]
    cycle_at= config['collider']['cycle_at']
    if line.element_names[0] != ee_at_dict[cycle_at]:
        print('CYCLING LINE') 
        line.cycle(name_first_element=ee_at_dict[cycle_at], inplace=True)

    # Building tracker
    line.build_tracker(_context=context)
    #==============================


    # Parsing emittance
    #==============================
    # Extracting emittance from previous config
    config_J001 = collider.metadata['config_J001']

    nemitt_x,nemitt_y = (config_J001['config_collider']['config_beambeam'][f'nemitt_{plane}'] for plane in ['x','y'])
    sigma_z           = config_J001['config_collider']['config_beambeam'][f'sigma_z']

    # # Computing RF bucket emittance
    rfbucket    = xPlus.RFBucket(line)
    nemitt_zeta = rfbucket.compute_emittance(sigma_z=sigma_z)
    #==============================

    # Generating particles
    #==============================
    num_particles = config['particles']['num_particles']
    mtd_config = config[config['particles']['type']]

    # HYPERSPHERE
    #==============================
    if config['particles']['type'] == 'hypersphere':
        coordinates = phys.hypersphere( N       = num_particles, 
                                        D       = 6, 
                                        r       = [ mtd_config['r_scale'][0],
                                                    mtd_config['r_scale'][1],
                                                    mtd_config['r_scale'][2],
                                                    mtd_config['r_scale'][3],
                                                    mtd_config['r_scale'][4],
                                                    mtd_config['r_scale'][5]],
                                        seed    = mtd_config['seed'], 
                                        unpack  = True)
    
    # GRID
    #==============================
    elif config['particles']['type'] == 'grid':
        grid_size = int(np.floor(np.sqrt(num_particles)))
        x,y = np.meshgrid(  mtd_config['offset_x']+np.linspace(0,mtd_config['max_x'],grid_size),
                            mtd_config['offset_y']+np.linspace(0,mtd_config['max_y'],grid_size))
        
        num_particles = len(x.flatten())
        coordinates = np.array([x.flatten(),
                                np.zeros(num_particles),
                                y.flatten(),
                                np.zeros(num_particles),
                                mtd_config['zeta']*np.ones(num_particles),
                                np.zeros(num_particles)])
    #==============================

                    

    condition = (np.abs(coordinates[0,:]-2)<2)
    coordinates = coordinates[:,condition]

    
    # Generating xsuite particles
    particles = xp.build_particles( line        = line,
                                    x_norm      = coordinates[0,:],
                                    px_norm     = coordinates[1,:],
                                    y_norm      = coordinates[2,:],
                                    py_norm     = coordinates[3,:],
                                    zeta_norm   = coordinates[4,:],
                                    pzeta_norm  = coordinates[5,:],
                                    nemitt_x    = nemitt_x, 
                                    nemitt_y    = nemitt_y,
                                    nemitt_zeta = nemitt_zeta,
                                    _context    = context)
    
    # Going back to normalized:
    XX_sig = xPlus._W_phys2norm(particles.x,particles.px,particles.y,particles.py,particles.zeta,particles.pzeta, 
                                W_matrix    = twiss_init.W_matrix,
                                co_dict     = twiss_init.particle_on_co.copy(_context=xo.context_default).to_dict(), 
                                nemitt_x    = nemitt_x, 
                                nemitt_y    = nemitt_y, 
                                nemitt_zeta = nemitt_zeta)
    
    # Checking:
    assert np.allclose(XX_sig[0,:],coordinates[0,:],atol=1e-13, rtol=0), 'Error in x'
    assert np.allclose(XX_sig[1,:],coordinates[1,:],atol=1e-13, rtol=0), 'Error in px'
    assert np.allclose(XX_sig[2,:],coordinates[2,:],atol=1e-13, rtol=0), 'Error in y'
    assert np.allclose(XX_sig[3,:],coordinates[3,:],atol=1e-13, rtol=0), 'Error in py'
    assert np.allclose(XX_sig[4,:],coordinates[4,:],atol=1e-13, rtol=0), 'Error in zeta'
    assert np.allclose(XX_sig[5,:],coordinates[5,:],atol=1e-13, rtol=0), 'Error in pzeta'


    # Exporting
    export_path = config['particles']['path'] + f'/{config["particles"]["name"]}.parquet'
    data = pd.DataFrame({'particle':particles.particle_id,  'x'     : particles.x,
                                                            'px'    : particles.px,
                                                            'y'     : particles.y,
                                                            'py'    : particles.py,
                                                            'zeta'  : particles.zeta,
                                                            'pzeta' : particles.pzeta,
                                                            'x_norm'    : coordinates[0,:],
                                                            'px_norm'   : coordinates[1,:],
                                                            'y_norm'    : coordinates[2,:],
                                                            'py_norm'   : coordinates[3,:],
                                                            'zeta_norm' : coordinates[4,:],
                                                            'pzeta_norm': coordinates[5,:],
                                                            })
    data.to_parquet(export_path)


    # Plotting:
    export_path = config['particles']['path'] + f'/VIEWER_{config["particles"]["name"]}.html'
    particles_to_HTML(particles,coordinates,collider,config,[nemitt_x,nemitt_y,nemitt_zeta],rfbucket,export_path)
    #==============================




    return particles




# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-c", "--config",      help = "Config file"  ,default = 'config.yaml')
    args = aparser.parse_args()
    
    
    assert Path(args.config).exists(), 'Invalid config path'
    
    particles = particle_dist(config_path=args.config)
    #===========================



