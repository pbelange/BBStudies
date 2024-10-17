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
    "text.usetex": False,
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
def prepare_line(config):
    # Parameters
    #-------------------------------------
    seq         = config['sequence']
    ip_name     = config['ip']
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

    # Extraction name of BB elements
    #-------------------------------------
    _direction  = 'clockwise' if seq == 'lhcb1' else 'anticlockwise'
    bblr_names  = collider._bb_config['dataframes'][_direction].groupby('ip_name').get_group(ip_name).groupby('label').get_group('bb_lr').index.to_list()
    bbho_names  = collider._bb_config['dataframes'][_direction].groupby('ip_name').get_group(ip_name).groupby('label').get_group('bb_ho').index.to_list()
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
    for nn in bblr_names_all:
        line0.element_refs[nn].scale_strength = 0
    for nn in bbho_names_all:
        line0.element_refs[nn].scale_strength = 0
    twiss0      = line0.twiss4d()
    twiss_init  = twiss0.get_twiss_init(at_element=s_marker)
    
    # Restoring active bblr and bbho
    #--------------------------------
    for nn,ss in zip(active_bblr,active_strength):
        line0.element_refs[nn].scale_strength = ss
    
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





    return line0,twiss0,twiss_init,beam_name,ip_name,s_marker,e_marker


def compute_residual(Jid = '0', config_file    = 'config.yaml'):
    
    # Parameter space
    #=======================================================================
    config = xutils.read_YAML(config_file)


    # Wire
    #---------------
    d_grid, I_grid  = np.meshgrid(  np.linspace(config['d_min'],config['d_max'],config['n_d']),
                                    np.linspace(config['I_min'],config['I_max'],config['n_I']))
    _dw , _Iw = d_grid.flatten()[int(Jid)], I_grid.flatten()[int(Jid)]
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
    line0,twiss0,twiss_init,beam_name,ip_name,s_marker,e_marker = prepare_line(config)


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
    df.to_parquet(config['out_path'] + f'/OUT_JOB_{str(Jid).zfill(4)}.parquet')

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
    aparser.add_argument("-plt" , "--plot"          , help = "Show results as plot" , action  = "store_true")
    aparser.add_argument("-splt" , "--splot"        , help = "Save results as plot" , action  = "store_true")
    aparser.add_argument("-DA"  , "--critical_DA"   , help = "Set crit. DA for plot", default = 6)
    aparser.add_argument("-CRA"  , "--critical_residual"   , help = "Set crit. DA for plot", default = None)
    args = aparser.parse_args()
    
    
    
    # Main function
    df,line,twiss_init,config = compute_residual(   Jid         = args.id_job,
                                                    config_file = args.config)
    
    # # Exporting to parquet
    # xutils.mkdir('./outputs')
    # df.to_parquet(f'./outputs/OUT_JOB_{str(args.id_job).zfill(4)}.parquet')


    if args.plot or args.splot:

        if args.critical_residual is None:
            # Finding critical residual
            critical_DA  = args.critical_DA
            critical_roi = (np.abs(df.r - critical_DA) < 0.2)
            DA_min = []
            for _cr in df.residual[critical_roi]:
                # Estimating DA
                _DA_per_angle = []
                for name,group in df.groupby(pd.cut(df.angle,1000),observed=True):
                    if np.any(group.residual>(_cr)):
                        _DA_per_angle.append(np.min(group.r[group.residual>(_cr)]))
                    else:
                        _DA_per_angle.append(np.max(group.r))
                DA_min.append(np.min(np.array(_DA_per_angle)))
            critical_res = np.array(df.residual[critical_roi])[np.argmin(np.abs(np.array(DA_min)-critical_DA))]
        else:
            critical_res = float(args.critical_residual)
            critical_DA = None

        # Estimating DA
        _DA_per_angle = []
        for name,group in df.groupby(pd.cut(df.angle,1000),observed=True):
            if np.any(group.residual>(critical_res)):
                _DA_per_angle.append(np.min(group.r[group.residual>(critical_res)]))
            else:
                _DA_per_angle.append(np.max(group.r))
        DA_min = np.min(np.array(_DA_per_angle))
        DA_avg = np.mean(np.array(_DA_per_angle))


        vmin = critical_res*(1-0.25)
        vmax = critical_res*(1+0.25)

        plt.figure()
        plt.scatter(df.rx,df.ry,s=20,c=df.residual,cmap='RdGy_r',vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(pad=-0.1)
        #---------------
        cbar_exponent = int(f'{critical_res:.2e}'.split('e')[1])
        cbar.ax.set_yticks([1.05*vmin, critical_res, 0.95*vmax])
        cbar.ax.set_yticklabels([f'{tk/10**(cbar_exponent):5.2f}' for tk in cbar.ax.get_yticks()])
        cbar.ax.text(1.3, 1.01, rf'$\times 10^{{{cbar_exponent}}}$', transform=cbar.ax.transAxes,fontsize=16)
        #---------------

        plt.axis('square')
        plt.xlim([0,df.r.max() + 0.5])
        plt.ylim([0,df.r.max() + 0.5])
        plt.gca().set_facecolor('#fafafa')  # off-white background


        plt.xlabel(r'$r_x$',fontsize=14)
        plt.ylabel(r'$r_y$',fontsize=14)
        cbar.set_label(r'Non-linear Residual')

        bbplt.polar_grid(rho_ticks=np.arange(0,20+1,1),phi_ticks=np.linspace(0,np.pi/2,21+2)[1:-1],alpha=0.3)
        _theta = np.linspace(0,np.pi/2,100)
        if critical_DA is not None:
            plt.plot(critical_DA*np.cos(_theta),critical_DA*np.sin(_theta),'--',lw=2,color='C0',label=rf'Calibration')
        plt.plot(DA_min*np.cos(_theta),DA_min*np.sin(_theta),'-',lw=3,color='C3',label=rf'CRA min: {DA_min:.2f} $\sigma$')
        plt.plot(DA_avg*np.cos(_theta),DA_avg*np.sin(_theta),'-',lw=3,color='C1',label=rf'CRA avg: {DA_avg:.2f} $\sigma$')
        plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper right',fontsize=10)
        plt.tight_layout()

        if args.splot:
            xutils.mkdir(config['out_path'] + '/figures')
            plt.savefig(config['out_path'] + f'/figures/OUT_JOB_{str(args.id_job).zfill(4)}.png',dpi=50)
        else:
            plt.show()
    
    #===========================
