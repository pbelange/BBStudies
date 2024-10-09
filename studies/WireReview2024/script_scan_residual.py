from pathlib import Path
import numpy as np
import pandas as pd
import time 
import gc 
import os


import matplotlib.pyplot as plt
import matplotlib.colors as colors

# xsuite
import xtrack as xt
import xfields as xf
import xobjects as xo

# BBStudies
import BBStudies.Physics.Constants as cst
import BBStudies.Tracking.Buffers as xBuff
import BBStudies.Tracking.Utils as xutils
import BBStudies.Plotting.BBPlots as bbplt

# Matplotlib config
#============================
FIGPATH  = './'
FIG_W = 6
FIG_H = 6


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "xtick.labelsize":14,
    "ytick.labelsize":14,
    "axes.labelsize":16,
    "axes.titlesize":16,
    "legend.fontsize":14,
    "legend.title_fontsize":16
})
plt.rc('text.latex', preamble=r'\usepackage{physics}')
for key in plt.rcParams.keys():
    if 'date.auto' in key:
        plt.rcParams[key] = "%H:%M"
#============================




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
def prepare_line(collider_file):
    # Parameters
    #-------------------------------------
    seq         = 'lhcb1'
    ip_name     = 'ip1'
    disable_ho  = True
    # as_wires    = True

    if seq == 'lhcb1':
        beam_name   = 'b1'
        s_marker    = f'e.ds.l{ip_name[-1]}.b1'
        # e_marker    = f's.ds.r{ip_name[-1]}.b1'
        e_marker    = f'{ip_name}'
    else:
        beam_name   = 'b2'
        s_marker    = f's.ds.r{ip_name[-1]}.b2'
        # e_marker    = f'e.ds.l{ip_name[-1]}.b2'
        e_marker    = f'{ip_name}'
    #-------------------------------------


    # Loading collider
    #-------------------------------------
    collider    = xt.Multiline.from_json(collider_file)
    line0       = collider[seq]

    # Adjusting beam-beam
    #-------------------------------------
    _direction  = 'clockwise' if seq == 'lhcb1' else 'anticlockwise'
    bblr_names  = collider._bb_config['dataframes'][_direction].groupby('ip_name').get_group(ip_name).groupby('label').get_group('bb_lr').index.to_list()
    bbho_names  = collider._bb_config['dataframes'][_direction].groupby('ip_name').get_group(ip_name).groupby('label').get_group('bb_ho').index.to_list()
    bblr_names_all  = collider._bb_config['dataframes'][_direction].groupby('label').get_group('bb_lr').index.to_list()
    bbho_names_all  = collider._bb_config['dataframes'][_direction].groupby('label').get_group('bb_ho').index.to_list()



    if disable_ho:
        for nn in bbho_names_all:
            line0.element_refs[nn].scale_strength = 0

    # Making sure all LR are enabled
    for nn in bblr_names_all:
        assert line0.element_refs[nn].scale_strength._value == 1, f'LR element {nn} is not enabled'



    # Adjusting wires
    #-------------------------------------
    # Power master knobs
    collider.vars[f'bbcw_enable_ff_tune'] = 1
    for loc in loc_list:
        collider.vars[f'i_wire.{loc}.{beam_name}']  = 0
        collider.vars[f'dn_wire.{loc}.{beam_name}'] = 50


    # Extracting nemitt
    line0.metadata['nemitt_x'] = collider.metadata['config_collider']['config_beambeam']['nemitt_x']
    line0.metadata['nemitt_y'] = collider.metadata['config_collider']['config_beambeam']['nemitt_y']


    # Ref Twiss
    #===========================================
    for nn in bblr_names_all:
        line0.element_refs[nn].scale_strength = 0
    twiss0      = line0.twiss4d()
    twiss_init  = twiss0.get_twiss_init(at_element=s_marker)
    for nn in bblr_names_all:
        line0.element_refs[nn].scale_strength = 1
    #===========================================


    return line0,twiss0,twiss_init,beam_name,ip_name,s_marker,e_marker


def scan_residual(  collider_file  = 'collider.json',
                    num_threads    = 'auto'):
    
    
    # Parameter space
    #=======================================================================
    # Wire
    #---------------
    dn_grid, I_grid = np.meshgrid(  np.linspace(10,18,30),
                                    np.linspace(0,200,30))

    #---------------

    # Tori
    #---------------
    n_part = int(1e3)
    #------------------
    r_min   = 4.0
    r_max   = 10.0
    n_r     = 40
    n_angles= 15
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
    context = xo.ContextCpu(omp_num_threads=num_threads)
    line0,twiss0,twiss_init,beam_name,ip_name,s_marker,e_marker = prepare_line(collider_file = collider_file)


    # Adjusting wires
    #-------------------------------------
    # Power master knobs
    line0.vars[f'bbcw_enable_ff_tune'] = 1
    for loc in [f'4l{ip_name[-1]}',f'4r{ip_name[-1]}']:
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



    # Scanning
    iter_avg = 0
    for _dn,_Iw,iter_idx in zip(dn_grid.flatten(),I_grid.flatten(),range(len(I_grid.flatten()))):
        if iter_idx != 0 :
            print(f'ITER: {iter_idx:4d}/{len(I_grid.flatten())} ({iter_avg:.3f} s/iter, {iter_avg*(len(I_grid.flatten())-iter_idx)/60:.3f} min remaining)')
        t1 = time.perf_counter()

        # Adjusting wires in both lines
        #-------------------------------------
        # Power master knobs
        line.vars[f'bbcw_enable_ff_tune'] = 1
        for loc in [f'4l{ip_name[-1]}']:
            line.vars[f'i_wire.{loc}.{beam_name}']  = _Iw
            line.vars[f'dn_wire.{loc}.{beam_name}'] = _dn

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
        os.system('cls||clear')

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


        df = pd.DataFrame({ 'Iw' :_Iw,
                            'dw' :_dn,
                            'rx' :rx_vec.flatten(),
                            'ry' :ry_vec.flatten(),
                            'r'      :rr.flatten(),
                            'angle'  :tt.flatten(),
                            'residual':residual})
        #===============================


        # Exporting to parquet
        xutils.mkdir('./CR_HL_data/outputs')
        df.to_parquet(f'./CR_HL_data/outputs/OUT_JOB_{str(iter_idx).zfill(4)}.parquet')

        t2 = time.perf_counter()
        if iter_idx == 0:
            iter_avg = (t2-t1)
        else:
            iter_avg = np.mean([iter_avg,(t2-t1)])
    # return df,line,twiss_init





# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-coll", "--collider"      , help = "Collider file"        , default = 'CR_HL_data/colliders/collider_baseline_250.json')
    aparser.add_argument("-m"   , "--multithread"   , help = "Num. Threads"         , default = 'auto')
    args = aparser.parse_args()
    
    
    
    # Main function
    scan_residual(  collider_file   = args.collider,
                    num_threads     = args.multithread)

    #===========================
