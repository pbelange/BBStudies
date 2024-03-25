
import numpy as np
import pandas as pd
from pathlib import Path
import traceback
import gc


# xsuite
import xtrack as xt
import xmask as xm
import xfields as xf
import xpart as xp
import xobjects as xo


# BBStudies
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.Buffers as xBuff
import BBStudies.Tracking.Utils as xutils
import BBStudies.Tracking.Progress as pbar
import BBStudies.Physics.Base as phys




# ==================================================================================================
# --- Functions to load collider with a given context
# ==================================================================================================
def load_collider(collider_path = '../001_configure_collider/zfruits/collider_001.json',user_context = 'CPU',device_id = 0):


    collider = xt.Multiline.from_json(collider_path)

    # Choosing contex, GPU
    #--------------------------------------
    if user_context == 'CPU':
        context = xo.ContextCpu(omp_num_threads='auto')
    elif user_context == 'GPU':
        context = xo.ContextCupy(device = device_id)
    #--------------------------------------

    return collider,context



# ==================================================================================================
# --- Functions to generate particle distribution
# ==================================================================================================
def generate_particles(from_path,line,nemitt_x = None,nemitt_y = None,_context = None):
    # Loading normalized coordinates
    # coord_df = xutils.import_parquet(from_path)
    coord_df = pd.read_parquet(from_path)
    
    # Generating xsuite particles
    particles = xp.build_particles( line        = line,
                                    x_norm      = coord_df.x_sig.values,
                                    px_norm     = coord_df.px_sig.values,
                                    y_norm      = coord_df.y_sig.values,
                                    py_norm     = coord_df.py_sig.values,
                                    zeta        = coord_df.zeta.values,
                                    delta       = coord_df.delta.values,
                                    nemitt_x    = nemitt_x, nemitt_y=nemitt_y,
                                    _context    =_context)
    
    return particles



# ==================================================================================================
# --- Cleaning monitor
# ==================================================================================================

# Adding reset method to monitors:
#==============================
def reset_monitor(self,start_at_turn = None,stop_at_turn = None):
    if start_at_turn is not None:
        self.start_at_turn = start_at_turn
    if stop_at_turn is not None:
        self.stop_at_turn = stop_at_turn
    
    with self.data._bypass_linked_vars():
            for tt, nn in self._ParticlesClass.per_particle_vars:
                getattr(self.data, nn)[:] = 0

xt.ParticlesMonitor.reset = reset_monitor
#==============================


# Adding to_pandas method to monitors:
#==============================
def pandas_monitor(self):
    
    extract_columns = ['at_turn','particle_id','x','px','y','py','zeta','pzeta','state']

    _df_tbt = self.data.to_pandas()

    _df_tbt.insert(list(_df_tbt.columns).index('zeta'),'pzeta',_df_tbt['ptau']/_df_tbt['beta0'])
    _df_tbt = _df_tbt[extract_columns].rename(columns={"at_turn": "turn",'particle_id':'particle'})

    return _df_tbt
xt.ParticlesMonitor.to_pandas = pandas_monitor
#==============================

# ==================================================================================================
# --- Main function
# ==================================================================================================
def particle_dist_and_track(config = None,config_path = 'config.yaml'):
    # Loading config
    #==============================
    if config is None:
        config = xutils.read_YAML(config_path)

    sequence   = config['tracking']['collider']['sequence']
    method     = config['tracking']['collider']['method']
    num_turns  = int(config['tracking']['num_turns'])
    num_particles = len(pd.read_parquet(config['tracking']['particles']['path']))
    ee_at_dict = config['elements'][sequence]

    chunks   = xPlus.split_in_chunks(num_turns, main_chunk  = config['tracking']['size_chunks'],
                                                n_chunks    = config['tracking']['num_chunks'])
    #==============================


    # Preparing output folder
    #==============================
    for _path in [config['analysis']['path']]:
        if _path is not None:
            xutils.mkdir(_path) 
    #==============================



    # Preparing collider
    #==============================
    print('LOADING COLLIDER')
    collider,context = load_collider(   collider_path = config['tracking']['collider']['path'],
                                        user_context  = config['tracking']['context']['type'],
                                        device_id     = config['tracking']['context']['device_id'])

    # Cycling line at_element
    line    = collider[sequence]
    cycle_at= config['tracking']['collider']['cycle_at']
    if line.element_names[0] != ee_at_dict[cycle_at]:
        print('CYCLING LINE') 
        line.cycle(name_first_element=ee_at_dict[cycle_at], inplace=True)


    # Installing monitors
    print('INSTALLING MONITORS') 
    monitors     = {}
    monitor_list = config['tracking']['collider']['monitor_at']
    if not isinstance(monitor_list,list):
        monitor_list = [monitor_list]
    for monitor_at in monitor_list:

        monitors[monitor_at] = {}
        monitors[monitor_at]['main'] = xt.ParticlesMonitor( _context      = context,
                                                            num_particles = num_particles,
                                                            start_at_turn = 0, 
                                                            stop_at_turn  = int(np.max(chunks)))

        ee_name = ee_at_dict[monitor_at]
        line.insert_element(index=ee_name, element=monitors[monitor_at]['main'], name=ee_name + '_monitor')
    line.build_tracker(_context=context)
    #==============================





    # Parsing emittance
    #==============================
    # Extracting emittance from previous config
    config_J001 = collider.metadata['config_J001']

    nemitt_x,nemitt_y = (config_J001['config_collider']['config_beambeam'][f'nemitt_{plane}'] for plane in ['x','y'])
    sigma_z           = config_J001['config_collider']['config_beambeam'][f'sigma_z']

    # Computing RF bucket emittance
    rfbucket    = xPlus.RFBucket(line)
    nemitt_zeta = rfbucket.compute_emittance(sigma_z=sigma_z)
    #==============================

    # Generating particles
    #==============================
    particles = generate_particles(from_path    = config['tracking']['particles']['path'],
                                    line        = line,
                                    nemitt_x    = nemitt_x,
                                    nemitt_y    = nemitt_y,
                                    _context    = context) 
    #==============================




    # Custom monitors and pcsections
    #==============================
    pcsections = {}
    buffers    = {}
    _twiss = line.twiss(method=method.lower())

    for monitor_at in monitors.keys():

        # CPU_monitor to only extract once from GPU
        monitors[monitor_at]['cpu'] = xBuff.CPU_monitor()

        # Storage for analysis
        if config['analysis']['num_turns'] is not None:
            monitors[monitor_at]['storage'] = xBuff.storage_monitor(num_particles,config['analysis']['num_turns'])
        else:
            # Point back to original cpu monitor
            monitors[monitor_at]['storage'] = monitors[monitor_at]['cpu']


        # NAFF storage for long N values
        if config['analysis']['naff']['active']:
            monitors[monitor_at]['storage_naff'] = xBuff.storage_monitor(num_particles,config['analysis']['naff']['num_turns'])
        else:
            monitors[monitor_at]['storage_naff'] = None

        # pcsections
        ee_name = ee_at_dict[monitor_at]
        ee_idx = _twiss.name.tolist().index(ee_name)
        pcsections[monitor_at] = xPlus.Poincare_Section(name           = monitor_at,
                                                        twiss          = _twiss.get_twiss_init(at_element=ee_name), 
                                                        tune_on_co     = [_twiss.mux[-1], _twiss.muy[-1], _twiss.muzeta[-1]],
                                                        nemitt_x       = nemitt_x,       
                                                        nemitt_y       = nemitt_y,       
                                                        nemitt_zeta    = nemitt_zeta)
        
        # Buffers
        buffers[monitor_at] = {'checkpoints':None,'excursion':None,'naff':None}
        if config['analysis']['checkpoints']['active']:
            buffers[monitor_at]['checkpoints'] = xBuff.Checkpoint_Buffer()
        if config['analysis']['excursion']['active']:
            buffers[monitor_at]['excursion'] = xBuff.Excursion_Buffer()
        if config['analysis']['naff']['active']:
            buffers[monitor_at]['naff'] = xBuff.NAFF_Buffer()
            buffers[monitor_at]['naff'].n_harm       = config['analysis']['naff']['num_harmonics']
            buffers[monitor_at]['naff'].window_order = config['analysis']['naff']['window_order']
            buffers[monitor_at]['naff'].window_type  = config['analysis']['naff']['window_type']
            buffers[monitor_at]['naff'].multiprocesses = config['analysis']['naff']['multiprocesses']

            # To be injected manually!
            #=========================
            buffers[monitor_at]['naff'].twiss          = pcsections[monitor_at].twiss
            buffers[monitor_at]['naff'].nemitt_x       = pcsections[monitor_at].nemitt_x       
            buffers[monitor_at]['naff'].nemitt_y       = pcsections[monitor_at].nemitt_y       
            buffers[monitor_at]['naff'].nemitt_zeta    = pcsections[monitor_at].nemitt_zeta    
            #=========================
    #==============================
        


    # Tracking
    #==============================
    print('START TRACKING...')
    # Tracking
    #--------------------------------------------

    PBar = pbar.ProgressBar(message='___  Tracking  ___',color='blue',n_steps=len(chunks),max_visible=3)
    interface = xPlus.Tracking_Interface(   line            = line,
                                            method          = method,
                                            cycle_at        = ee_at_dict[cycle_at],
                                            sequence        = sequence,
                                            context         = context,
                                            config          = config,

                                            num_particles   = num_particles,
                                            num_turns       = num_turns,

                                            nemitt_x        = nemitt_x,       
                                            nemitt_y        = nemitt_y,       
                                            nemitt_zeta     = nemitt_zeta,
                                            sigma_z         = sigma_z,

                                            poincare        = list(pcsections.values()),
                                            
                                            PBar            = PBar,
                                            progress_divide = 100,)

    # Saving Interface data
    #--------------------------
    interface.export_metadata(  path = config['analysis']['path'],
                                collider_name    = config['tracking']['collider']['name'],
                                distribution_name= config['tracking']['particles']['name'])

    # Optimizing for tracking and then starting:
    #-------------------------------
    line.optimize_for_tracking()
    #-------------------------------
    PBar.start()
    try:
        ID_length = len(str(len(chunks)))
        for ID,chunk in enumerate(chunks):

            # Updating progress bar
            PBar.add_subtask(ID,message = f'CHUNK {str(ID).zfill(ID_length)}/{str(len(chunks)-1).zfill(ID_length)}',color = 'red',n_steps = chunk,level=2)

            # Resetting monitors
            #--------------------------------------------
            last_turn    = context.nparray_from_context_array(particles.at_turn).max()

            for key in monitors.keys():
                monitors[key]['main'].reset(start_at_turn = last_turn, 
                                            stop_at_turn  = last_turn + chunk)
            #--------------------------------------------

            # Tracking
            #--------------------------------------------
            interface.run_tracking(line,particles,num_turns = chunk)
            #--------------------------------------------
            
            # Processing all buffers
            for key in monitors.keys():
                new_data = []

                _monitor_main = monitors[key]['main']
                _monitor_cpu = monitors[key]['cpu']
                _monitor_storage      = monitors[key]['storage']
                _monitor_storage_naff = monitors[key]['storage_naff']
                

                _buffer_checkpoints = buffers[key]['checkpoints']
                _buffer_excursion = buffers[key]['excursion']
                _buffer_naff = buffers[key]['naff'] 

                # Making sure the data is on cpu:
                _monitor_cpu.process(monitor=_monitor_main)
                
                # Saving into storage
                if isinstance(_monitor_storage,xBuff.storage_monitor):
                    _monitor_storage.process(monitor=_monitor_cpu)
                # Else: _monitor_storage IS _monitor_cpu anyway    

                # Processing the analysis
                if _monitor_storage.is_full:
                    # Saving Checkpoint if needed
                    #--------------------------
                    if _buffer_checkpoints is not None:
                        _buffer_checkpoints.process(monitor=_monitor_storage)
                        new_data.append('checkpoints')

                    # Saving Excursion if needed
                    #--------------------------
                    if _buffer_excursion is not None:
                        _buffer_excursion.process(monitor=_monitor_storage)
                        new_data.append('excursion')

                    # Cleaning up!
                    _monitor_storage.clean()

                # NAFF computations
                #--------------------------
                if _buffer_naff is not None:
                    # Storing for NAFF
                    _monitor_storage_naff.process(monitor=_monitor_cpu)

                    # Processing (NAFF if window length is reached)
                    if _monitor_storage_naff.is_full:
                        _buffer_naff.process(monitor=_monitor_storage_naff)
                        new_data.append('naff')

                        # Cleaning up!
                        _monitor_storage_naff.clean()
                        
                #--------------------------
                        
                # TURN-BY-TURN
                #--------------------------
                if config['analysis']['turn_by_turn']['active']:
                    new_data.append('tbt')
            

                # Writing to parquet
                #--------------------------
                if len(new_data)>0:
                    for data_key in new_data:
                        if data_key == 'tbt':
                            _tbt = _monitor_main.to_pandas()
                            _tbt.insert(0,'chunk',ID)
                            pcsections[key].data[data_key] = _tbt
                        else:
                            pcsections[key].data[data_key] = buffers[key][data_key].to_pandas()
                            buffers[key][data_key].clean()
                    
                    pcsections[key].to_parquet(path          = config['analysis']['path'],
                                            collider_name    = config['tracking']['collider']['name'],
                                            distribution_name= config['tracking']['particles']['name'],
                                            datakeys         = new_data)
        
        # Updating Interface data excecution time)
        #--------------------------
        interface.export_metadata(  path = config['analysis']['path'],
                                    collider_name    = config['tracking']['collider']['name'],
                                    distribution_name= config['tracking']['particles']['name'])
                
    except Exception as error:
        PBar.close()
        print("An error occurred:", type(error).__name__, " - ", error)
        traceback.print_exc()
    except KeyboardInterrupt:
        PBar.close()
        print("Terminated by user: KeyboardInterrupt")


    return particles,interface


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
    
    particles,interface = particle_dist_and_track(config_path=args.config)
    #===========================