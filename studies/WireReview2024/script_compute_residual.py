from pathlib import Path
import numpy as np
import pandas as pd

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
def prepare_line(Iw,dn,collider_file):
    # Parameters
    #-------------------------------------
    seq         = 'lhcb1'
    ip_name     = 'ip1'
    disable_ho  = True

    if seq == 'lhcb1':
        beam_name   = 'b1'
        s_marker    = f'e.ds.l{ip_name[-1]}.b1'
        e_marker    = f's.ds.r{ip_name[-1]}.b1'
    else:
        beam_name   = 'b2'
        s_marker    = f's.ds.r{ip_name[-1]}.b2'
        e_marker    = f'e.ds.l{ip_name[-1]}.b2'
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
    if disable_ho:
        for nn in bbho_names:
            line0.element_refs[nn].scale_strength = 0

    # Making sure all LR are enabled
    for nn in bblr_names:
        assert line0.element_refs[nn].scale_strength._value == 1, f'LR element {nn} is not enabled'
        # line0.element_refs[nn].scale_strength = 0.7

    # Adjusting wires
    #-------------------------------------
    # Power master knobs
    for loc in loc_list:
        collider.vars[f'i_wire.{loc}.{beam_name}']  = Iw
        collider.vars[f'dn_wire.{loc}.{beam_name}'] = dn

    twiss0      = line0.twiss4d()
    twiss_init  = twiss0.get_twiss_init(at_element=s_marker)
    line        = line0.select(s_marker,e_marker)

    # Extracting nemitt
    line.metadata['nemitt_x'] = collider.metadata['config_collider']['config_beambeam']['nemitt_x']
    line.metadata['nemitt_y'] = collider.metadata['config_collider']['config_beambeam']['nemitt_y']

    return line,twiss_init


def compute_residual(Jid            = 0,
                     collider_file  = './colliders/collider_bbcw.json',
                     num_threads    = 'auto'):
    
    
    # Parameter space
    #=======================================================================
    # Wire
    #---------------
    dn_grid, I_grid = np.meshgrid(  np.linspace(10,18,50),
                                    np.linspace(0,200,50))
    
    chosen_dn = dn_grid.flatten()[int(Jid)]
    chosen_I  = I_grid.flatten()[int(Jid)]
    #---------------

    # Tori
    #---------------
    n_part = int(1e3)
    #------------------
    r_min   = 4.0
    r_max   = 10.0
    n_r     = 50
    n_angles= 31
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


    # Beam info
    #---------------



    # Prepare line
    #=======================================================================
    context = xo.ContextCpu(omp_num_threads=num_threads)
    line,twiss_init = prepare_line(Iw = chosen_I,dn = chosen_dn,collider_file = collider_file)

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
    
    line.build_tracker(_context=context)
    twiss = line.twiss4d(start=line.element_names[0],end=line.element_names[-1],init=twiss_init)
    #=======================================================================

    # Buffer
    #=======================================================================
    buffer  = xBuff.TORUS_Buffer(complex2tuple=False,skip_naff=True)
    #---------------------------------------------------------
    buffer.n_torus      = n_torus
    buffer.n_points     = n_part
    buffer.twiss        = twiss.get_twiss_init(at_element=monitor_name)
    buffer.nemitt_x     = line.metadata['nemitt_x']    
    buffer.nemitt_y     = line.metadata['nemitt_y']    
    buffer.nemitt_zeta  = None # To avoid any rescaling
    #---------------------------------------------------------
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

    # TRACKING
    #=======================================================================
    particles = line.build_particles(   x_norm   = init_coord['x_n'],
                                        px_norm  = init_coord['px_n'],
                                        y_norm   = init_coord['y_n'],
                                        py_norm  = init_coord['py_n'],
                                        method   = '4d',
                                        nemitt_x = line.metadata['nemitt_x'],
                                        nemitt_y = line.metadata['nemitt_y'],
                                        nemitt_zeta     = None,
                                        W_matrix        = twiss.W_matrix[0],
                                        particle_on_co  = twiss.particle_on_co.copy(),
                                        _context        = context)

    line.track(particles, num_turns= 1,turn_by_turn_monitor=True,with_progress=True)
    #=======================================================================


    # Processing
    #==============================
    buffer.process(monitor=monitor)
    df_buffer = buffer.to_pandas().groupby('turn').get_group(0).set_index('torus')
    
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


    df = pd.DataFrame({'Iw' :chosen_I,
                       'dw' :chosen_dn,
                       'rx' :rx_vec.flatten(),
                       'ry' :ry_vec.flatten(),
                       'r'      :rr.flatten(),
                       'angle'  :tt.flatten(),
                       'residual':residual})
    #===============================

    return df,line,twiss_init





# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-id"  , "--id_job"        , help = "Job ID"               , default = '0')
    aparser.add_argument("-coll", "--collider"      , help = "Collider file"        , default = './colliders/collider_bbcw.json')
    aparser.add_argument("-m"   , "--multithread"   , help = "Num. Threads"         , default = 'auto')
    aparser.add_argument("-plt" , "--plot"          , help = "Show results as plot" , action  = "store_true")
    aparser.add_argument("-splt" , "--splot"        , help = "Save results as plot" , action  = "store_true")
    aparser.add_argument("-DA"  , "--critical_DA"   , help = "Set crit. DA for plot", default = 6)
    aparser.add_argument("-CRA"  , "--critical_residual"   , help = "Set crit. DA for plot", default = None)
    args = aparser.parse_args()
    
    
    
    # Main function
    df,line,twiss_init = compute_residual(  Jid             = args.id_job,
                                            collider_file   = args.collider,
                                            num_threads     = args.multithread)
    
    # # Exporting to parquet
    # xutils.mkdir('./outputs')
    # df.to_parquet(f'./outputs/OUT_JOB_{str(args.id_job).zfill(4)}.parquet')


    if args.plot or args.splot:

        if args.critical_residual is None:
            # Finding critical residual
            critical_DA  = float(args.critical_DA)
            critical_roi = (np.abs(df.r - critical_DA) < 0.2)
            critical_res = np.max(df.residual[critical_roi])
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
            xutils.mkdir('./figures')
            plt.savefig(f'./figures/OUT_JOB_{str(args.id_job).zfill(4)}.png',dpi=300)
        else:
            plt.show()
    
    #===========================
