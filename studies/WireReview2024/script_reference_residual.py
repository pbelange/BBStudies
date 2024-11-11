
import numpy as np
import pandas as pd
import gc 

# xsuite
import xtrack as xt
import xobjects as xo

# BBStudies
import BBStudies.Tracking.Buffers as xBuff
import BBStudies.Tracking.Utils as xutils




# Default locations
# ----------------------------
beam_list = ['b1', 'b2']
loc_list  = ['4l1','4r1','4l5','4r5']
ip_list   = ['ip1', 'ip5']
tank_list = ['a', 'b', 'c']
# ----------------------------


# ==================================================================================================
# --- Main script
# ==================================================================================================
def prepare_line(Jid,config):
    # Parameters
    #-------------------------------------
    seq         = config['sequence']
    disable_ho  = config['disable_ho']
    # as_wires    = True

    beam_name = seq[-2:]
    s_marker  = config['s_marker']
    e_marker  = config['e_marker']
    #-------------------------------------


    # Loading collider
    #-------------------------------------
    collider    = xt.Multiline.from_json(config['collider_path'])
    line0       = collider[seq]

    # Adjusting beam-beam
    #-------------------------------------
    _direction  = 'clockwise' if seq == 'lhcb1' else 'anticlockwise'
    bblr_names_all  = collider._bb_config['dataframes'][_direction].groupby('label').get_group('bb_lr').index.to_list()
    bbho_names_all  = collider._bb_config['dataframes'][_direction].groupby('label').get_group('bb_ho').index.to_list()



    # Keeping only the active bblr
    active_bblr     = [nn for nn in bblr_names_all if line0.element_refs[nn].scale_strength._value != 0]
    active_strength = [line0.element_refs[nn].scale_strength._value for nn in bblr_names_all if line0.element_refs[nn].scale_strength._value != 0]



    # Adjusting wires
    #-------------------------------------
    # Power master knobs
    collider.vars[f'bbcw_enable_ff_tune'] = 1
    if config['d_normalized']:
        for loc in loc_list:
            collider.vars[f'i_wire.{loc}.{beam_name}']  = 0
            collider.vars[f'dn_wire.{loc}.{beam_name}'] = 50
    else:
    # Link all tanks to common knob
        for loc  in loc_list:
            # Define master knobs
            collider.vars[f'i_wire.{loc}.{beam_name}'] = 0
            collider.vars[f'd_wire.{loc}.{beam_name}'] = 1 
            # Link tank knobs
            for tank in tank_list:
                # Distance to wire
                collider.vars[f'd_wire.{tank}.{loc}.{beam_name}'] = collider.vars[f'd_wire.{loc}.{beam_name}']


    # Extracting nemitt
    line0.metadata['nemitt_x'] = collider.metadata['config_collider']['config_beambeam']['nemitt_x']
    line0.metadata['nemitt_y'] = collider.metadata['config_collider']['config_beambeam']['nemitt_y']





    # Ref Twiss
    #===========================================
    rescale_bblr = np.linspace(0,1,11)[int(Jid)]

    for nn in bblr_names_all:
        line0.element_refs[nn].scale_strength = 0
    for nn in bbho_names_all:
        line0.element_refs[nn].scale_strength = 0
    
    twiss0      = line0.twiss4d()
    twiss_init  = twiss0.get_twiss_init(at_element=s_marker)
    
    # Restoring active bblr and bbho
    #--------------------------------
    for nn,ss in zip(active_bblr,active_strength):
        line0.element_refs[nn].scale_strength = ss*rescale_bblr
    
    if not disable_ho:
        for nn in bbho_names_all:
            line0.element_refs[nn].scale_strength = 1
    #===========================================


    # Killing some BBLR (forward physics test)
    #===========================================
    if 'kill_bblr' in list(config.keys()):
        if config['kill_bblr'] is not None:
            for to_kill in config['kill_bblr']:
                for nn in bblr_names_all:
                    if f'bb_lr.{to_kill}' in nn: 
                        line0.element_refs[nn].scale_strength = 0
    #===========================================

    return line0,twiss0,twiss_init,beam_name,s_marker,e_marker


def reference_residual(Jid = '0', config_file    = 'config.yaml'):
    
    # Parameter space
    #=======================================================================
    config = xutils.read_YAML(config_file)


    # Wires OFF
    #---------------
    config['d_normalized'] = True
    _Iw = 0 
    _dw = 10
    #---------------

    # Tori
    #---------------
    n_part = int(config['n_part'])
    #------------------
    r_min   = config['r_min']
    r_max   = config['r_max']
    n_r     = config['n_r']
    n_angles= config['n_angles']
    #--------------------
    radial_list = np.linspace(r_min, r_max, n_r)
    theta_list  = np.linspace(0, np.pi/2, n_angles + 2)[1:-1]
    rr,tt       = np.meshgrid(radial_list, theta_list)
    #--------------------
    rx_vec, ry_vec = rr*np.cos(tt), rr*np.sin(tt)
    #--------------------------------
    fx  = 1/2/np.sqrt(2)
    fy  = 1/2/np.sqrt(3)
    fz  = -1/2/np.sqrt(5)/100
    #=======================================================================



    # GENERATING TORI
    #=======================================================================
    Tx= 2*np.pi*fx*np.arange(n_part)
    Ty= 2*np.pi*fy*np.arange(n_part)
    init_coord = {'x_n':[],'px_n':[],'y_n':[],'py_n':[],'zeta_n':[],'pzeta_n':[]}
    for rx,ry in zip(rx_vec.flatten(),ry_vec.flatten()):
        Gx  = rx*np.exp(1j*Tx)
        Gy  = ry*np.exp(1j*Ty)

        init_coord[f'x_n']  += list(np.real(Gx))
        init_coord[f'px_n'] += list(-np.imag(Gx))
        init_coord[f'y_n']  += list(np.real(Gy))
        init_coord[f'py_n'] += list(-np.imag(Gy))
    #=======================================================================


    # Prepare line
    #=======================================================================
    context = xo.ContextCpu(omp_num_threads=config['num_threads'])
    line0,twiss0,twiss_init,beam_name,s_marker,e_marker = prepare_line(Jid,config)


    # Adjusting wires
    #-------------------------------------
    # Power master knobs
    line0.vars[f'bbcw_enable_ff_tune'] = 1
    for loc in loc_list:
        line0.vars[f'i_wire.{loc}.{beam_name}']  = 0
        line0.vars[f'dn_wire.{loc}.{beam_name}'] = 12



    line        = line0.select(s_marker,e_marker)
    bbcw_names  = [nn for nn in line.element_names if 'bbcw' in nn]


    # Monitor
    #-------------------------------------
    monitor_name = 'buffer_monitor'
    n_torus = len(rx_vec.flatten())  
    monitor = xt.ParticlesMonitor(  _context      = context,
                                    num_particles = int(n_torus*n_part) ,
                                    start_at_turn = 0, 
                                    stop_at_turn  = 1)
    line.insert_element(index=line.element_names[-1], element=monitor, name=monitor_name)
    #-------------------------------------
    twiss = line.twiss4d(start=line.element_names[0],end=line.element_names[-1],init=twiss_init)
    #=======================================================================




    # Adjusting wires 
    #-------------------------------------
    # Power master knobs
    if config['d_normalized']:
        d_knob = 'dn'
    else:
        d_knob = 'd'

    line.vars[f'bbcw_enable_ff_tune'] = 1
    for loc in config['bbcw_locations']:
        line.vars[f'i_wire.{loc}.{beam_name}']          = _Iw
        line.vars[f'{d_knob}_wire.{loc}.{beam_name}']   = _dw

    twiss = line.twiss4d(start=line.element_names[0],end=line.element_names[-1],init=twiss_init)
    #=======================================================================



    # Buffer
    #=======================================================================
    buffer  = xBuff.TORUS_Buffer(complex2tuple=False,skip_naff=True)
    #---------------------------------------------------------
    buffer.n_torus      = n_torus
    buffer.n_points     = n_part
    buffer.twiss        = twiss.get_twiss_init(at_element=monitor_name)
    buffer.nemitt_x     = line0.metadata['nemitt_x']    
    buffer.nemitt_y     = line0.metadata['nemitt_y']    
    buffer.nemitt_zeta  = None # To avoid any rescaling
    #---------------------------------------------------------
    #=======================================================================




    # TRACKING
    #=======================================================================
    particles = line.build_particles(   x_norm   = init_coord['x_n'],
                                        px_norm  = init_coord['px_n'],
                                        y_norm   = init_coord['y_n'],
                                        py_norm  = init_coord['py_n'],
                                        method   = '4d',
                                        nemitt_x = line0.metadata['nemitt_x'],
                                        nemitt_y = line0.metadata['nemitt_y'],
                                        nemitt_zeta     = None,
                                        W_matrix        = twiss.W_matrix[0],
                                        particle_on_co  = twiss.particle_on_co.copy(),
                                        _context        = context)

    monitor.reset(start_at_turn = 0,stop_at_turn = 1)
    line.track(particles, num_turns= 1,turn_by_turn_monitor=True)
    #=======================================================================


    # Processing
    #==============================
    buffer.process(monitor=monitor)
    df_buffer = buffer.to_pandas().groupby('turn').get_group(0).set_index('torus')
    #==============================

    
    gc.collect()
    # os.system('cls||clear')

    residual = []
    for rx,ry,(idx,torus) in zip(rx_vec.flatten(),ry_vec.flatten(),df_buffer.iterrows()):

        # Initial CS-Action
        Jx0 = rx**2/2
        Jy0 = ry**2/2

        # Average CS-Action
        Jx_avg = torus.Jx
        Jy_avg = torus.Jy

        # Relative error
        _err = np.sqrt((Jx_avg-Jx0)**2 + (Jy_avg-Jy0)**2)/np.sqrt(Jx0**2 + Jy0**2)
        
        residual.append(_err)
    residual = np.array(residual)


    df = pd.DataFrame({ 'Iw'        :_Iw,
                        f'{d_knob}' :_dw,
                        'rx'        :rx_vec.flatten(),
                        'ry'        :ry_vec.flatten(),
                        'r'         :rr.flatten(),
                        'angle'     :tt.flatten(),
                        'residual'  :residual})
    #===============================


    # Exporting to parquet
    xutils.mkdir(config['out_path'])
    df.to_parquet(config['out_path'] + f'/REF_JOB_{str(Jid).zfill(4)}.parquet')

    return df,line,twiss_init,config






# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-id"  , "--id_job"        , help = "Job ID"               , default = '0')
    aparser.add_argument("-c"   , "--config"        , help = "Config file"          , default = './config.yaml')
    args = aparser.parse_args()
    
    
    
    # Main function
    df,line,twiss_init,config = reference_residual(     Jid         = args.id_job,
                                                        config_file = args.config)
    
    