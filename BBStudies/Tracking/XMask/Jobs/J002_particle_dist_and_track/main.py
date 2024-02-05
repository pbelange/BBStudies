
import numpy as np
import pandas as pd
from pathlib import Path
import gc


# xsuite
import xtrack as xt
import xmask as xm
import xfields as xf
import xpart as xp
import xobjects as xo


# BBStudies
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.XMask.Utils as xutils
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
    collider.build_trackers(_context=context)
    #--------------------------------------

    return collider,context



# ==================================================================================================
# --- Functions to generate particle distribution
# ==================================================================================================
def generate_particles(from_path,line,nemitt_x = None,nemitt_y = None,_context = None):
    # Loading normalized coordinates
    coord_df = xPlus.import_parquet(from_path)
    
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
# --- Functions to initialize monitor
# ==================================================================================================
def initialize_monitor(context = None,num_particles = 0,start_at_turn=0,nturns = 1):
        return xt.ParticlesMonitor( _context       = context,
                                    num_particles = num_particles,
                                    start_at_turn = start_at_turn, 
                                    stop_at_turn  = start_at_turn + nturns)



# ==================================================================================================
# --- Main function
# ==================================================================================================
def particle_dist_and_track(config = None,config_path = 'config.yaml'):


    # Loading config
    #==============================
    if config is None:
        config = xutils.read_YAML(config_path)
    #==============================


    # Preparing output folder
    #==============================
    turn_b_turn_path    = config['tracking']['turn_b_turn_path']
    data_path           = config['tracking']['data_path']
    checkpoint_path     = config['tracking']['checkpoint_path']
    particles_path      = config['tracking']['particles_path']

    for _path in [turn_b_turn_path,data_path,checkpoint_path]:
        if _path is not None:
            xutils.mkdir(_path) 
    #==============================


    # Loading collider
    #==============================
    print('LOADING COLLIDER')
    collider,context = load_collider(   collider_path = config['tracking']['collider_path'],
                                        user_context  = config['tracking']['user_context'],
                                        device_id     = config['tracking']['device_id'])
    #==============================


    # Parsing config
    #==============================
    sequence = config['tracking']['sequence']
    line     = collider[sequence]
    n_turns  = int(config['tracking']['n_turns'])
    monitor_at_dict = config['elements'][sequence]
    monitor_at      = config['tracking']['monitor_at']
    if monitor_at in monitor_at_dict.keys():
        monitor_at = monitor_at_dict[monitor_at]
    #==============================

    # Cycling line at_element
    #==============================
    if line.element_names[0] != monitor_at:
        line.cycle(name_first_element=monitor_at, inplace=True)
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
    particles = generate_particles(from_path    = particles_path,
                                    line        = line,
                                    nemitt_x    = nemitt_x,
                                    nemitt_y    = nemitt_y,
                                    _context    = context) 
    n_parts  = len(particles.particle_id)
            
    # Handpicked for lighter t-by-t data
    if config['tracking']['handpick_every'] is not None:
        handpick_particles = particles.particle_id[::config['tracking']['handpick_every']]
    else:
        handpick_particles = None    
    #==============================



    
    # Finding number of chunks:
    #==============================
    n_chunks   = config['tracking']['n_chunks']
    main_chunk = config['tracking']['chunk_size']
    if n_chunks:
        n_chunks = int(n_chunks)
    if main_chunk:
        main_chunk = int(main_chunk)
    

    chunks   = xPlus.split_in_chunks(n_turns,n_chunks=n_chunks,main_chunk=main_chunk)
    #==============================


    

    # Creating data buffer and checkpoint if needed
    #==============================
    data_buffer = None
    if config['tracking']['data_path'] is not None:
        data_buffer = xPlus.naff_Buffer()

    checkpoint_buffer = None
    if config['tracking']['checkpoint_path'] is not None:
        checkpoint_buffer = xPlus.Checkpoint_Buffer()
    #==============================


    # Tracking
    #==============================
    print('START TRACKING...')
    
    ID_length    = len(str(len(chunks)))

    PBar = pbar.ProgressBar(message='___  Tracking  ___',color='blue',n_steps=len(chunks),max_visible=3)
    PBar.start()
    for ID,chunk in enumerate(chunks):

        # Updating progress bar
        PBar.add_subtask(ID,message = f'CHUNK {str(ID).zfill(ID_length)}/{str(len(chunks)-1).zfill(ID_length)}',color = 'red',n_steps = chunk,level=2)

        # Setting monitor to make sure we can overwrite it
        #--------------------------------------------
        last_turn    = context.nparray_from_context_array(particles.at_turn).max()
        main_monitor = initialize_monitor(context       = context,
                                          num_particles = len(particles.particle_id),
                                          start_at_turn = last_turn,
                                          nturns        = chunk)
        #--------------------------------------------


        # Tracking
        #--------------------------------------------
        tracked = xPlus.Tracking_Interface( line      = line,
                                            particles = particles,
                                            _context  = context,
                                            n_turns   = chunk,
                                            monitor   = main_monitor,
                                            monitor_at= monitor_at,
                                            nemitt_x  = nemitt_x,
                                            nemitt_y  = nemitt_y,
                                            nemitt_zeta = nemitt_zeta,
                                            sigma_z     = sigma_z,
                                            Pbar        = PBar,
                                            progress_divide = 100,
                                            config          = config)
        #--------------------------------------------


        # Data Buffer Computation
        #---------------
        if config['tracking']['data_path'] is not None:
            data_buffer.process(monitor=main_monitor)
        if config['tracking']['checkpoint_path'] is not None:
            checkpoint_buffer.process(monitor=main_monitor)
        #---------------

        # Saving Chunk if needed
        #--------------------------
        if config['tracking']['turn_b_turn_path'] is not None:
            if config['tracking']['last_n_turns'] is not None:
                # Keeping only last_n_turns!
                #----------
                turn_idx_min = config['tracking']['n_turns'] - config['tracking']['last_n_turns']
                tracked._df._df = tracked.df[tracked.df.turn>=turn_idx_min]
                #----------

                if (context.nparray_from_context_array(particles.at_turn).max())>turn_idx_min:
                    tracked.to_parquet(config['tracking']['turn_b_turn_path'],partition_name='CHUNK',partition_ID=str(ID).zfill(ID_length),handpick_particles=handpick_particles)
            else:
                tracked.to_parquet(config['tracking']['turn_b_turn_path'],partition_name='CHUNK',partition_ID=str(ID).zfill(ID_length),handpick_particles=handpick_particles)
        #--------------------------

    
    # Saving data buffer if needed
    #--------------------------
    if config['tracking']['data_path'] is not None:
        tracked.exec_time    = PBar.main_task.finished_time
        tracked.parquet_data = '_data'
        tracked._data        = data_buffer.to_pandas()
        tracked.to_parquet(config['tracking']['data_path'],partition_name=config['tracking']['partition_name'],partition_ID=config['tracking']['partition_ID'])

    if config['tracking']['checkpoint_path'] is not None:
        tracked.exec_time    = PBar.main_task.finished_time
        tracked.parquet_data = '_checkpoint'
        tracked._checkpoint  = checkpoint_buffer.to_pandas()
        tracked.to_parquet(config['tracking']['checkpoint_path'],partition_name=config['tracking']['partition_name'],partition_ID=config['tracking']['partition_ID'])
    #--------------------------



    return particles,tracked


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
    
    particles,tracked = particle_dist_and_track(config_path=args.config)
    #===========================