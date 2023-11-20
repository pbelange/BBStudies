
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
def generate_particles(n_part = 1000,force_n_part = False,line = None,_context = None,at_element = None,nemitt_x = None,nemitt_y = None):

    n_part  = int(n_part)
    n_r     = int(np.floor(np.sqrt(n_part)))
    n_theta = int(n_part//n_r + 1)
    coordinates = phys.polar_grid(  r_sig     = np.linspace(0,10,n_r),
                                    theta_sig = np.linspace(0,np.pi/2,n_theta))
    
    momentum  = phys.polar_grid(  r_sig     = np.linspace(0,10,n_r),
                                    theta_sig = np.linspace(0,np.pi/2,n_theta))
    
    # Shuffling momentum
    #-----------------------
    ID = list(momentum.index)
    np.random.seed(0)
    np.random.shuffle(ID)
    momentum = momentum.loc[ID,['x_sig','y_sig']].reset_index(drop=True).rename(columns={'x_sig':'px_sig','y_sig':'py_sig'})
    #-----------------------

    coordinates = pd.concat([coordinates,momentum],axis=1)
    # coordinates.insert(0,'delta',0)
    

    if force_n_part:
        coordinates = coordinates[:n_part]

    if line is not None:
        particles = xp.build_particles(  line   = line,
                                        x_norm  =coordinates.x_sig.values,
                                        px_norm =coordinates.px_sig.values,
                                        y_norm  =coordinates.y_sig.values,
                                        py_norm =coordinates.py_sig.values,
                                        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                        at_element=at_element,
                                        _context=_context)
    else:
        particles = None


    
    return particles,coordinates


def generate_realistic_particles(n_part = 1000,r_sig_max = 3,line = None,_context = None,at_element = None,nemitt_x = None,nemitt_y = None,sigma_z = None):

    generator = phys.polar_grid(r_sig     = [0] + list(np.linspace(0.1,6.5,15)),
                            theta_sig = np.linspace(0,2*np.pi,100))
    generator = generator.rename(columns={'y_sig':'px_sig'})[['x_sig','px_sig']]
    n_part = len(generator)


    # Longitudinal plane: generate gaussian distribution matched to bucket 
    # zeta, delta, matcher = xp.generate_longitudinal_coordinates(num_particles=n_part, distribution='gaussian',sigma_z=sigma_z, line=line,return_matcher=True)
    # nemitt_zeta = matcher._compute_emittance(matcher.rfbucket,matcher.psi)

    if line is not None:
        particles = xp.build_particles( line    = line,
                                        x_norm  = generator.x_sig.values,
                                        px_norm = generator.px_sig.values,
                                        y_norm  = None,
                                        py_norm = None,
                                        zeta    = None,
                                        delta   = None,
                                        nemitt_x   = nemitt_x, nemitt_y=nemitt_y,
                                        at_element = at_element)
    else:
        particles = None

    return particles,0



def initialize_monitor(context = None,num_particles = 0,start_at_turn=0,nturns = 1):
        monitor = xt.ParticlesMonitor( _context       = context,
                                        num_particles = num_particles,
                                        start_at_turn = start_at_turn, 
                                        stop_at_turn  = start_at_turn + nturns)
        return monitor



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
    n_parts  = int(config['tracking']['n_parts'])
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


    # Generating particle distribution
    #==============================
    # Extracting emittance from previous config
    config_J001 = collider.metadata['config_J001']
    
    nemitt_x,nemitt_y = (config_J001['config_collider']['config_beambeam'][f'nemitt_{plane}'] for plane in ['x','y'])
    sigma_z           = config_J001['config_collider']['config_beambeam'][f'sigma_z']
    nemitt_zeta       = 1 # will be computed from matching in generate_particle()
    #-----------------------
    print('GENERATING PARTICLES')
    # particles,coordinates = generate_particles( n_part      = n_parts,
    #                                             force_n_part= False,
    #                                             nemitt_x    = nemitt_x,
    #                                             nemitt_y    = nemitt_y,
    #                                             line        = line,
    #                                             at_element  = monitor_at,
    #                                              _context   = context)

    particles,nemitt_zeta = generate_realistic_particles(   n_part      = n_parts,
                                                            r_sig_max   = 3,
                                                            nemitt_x    = nemitt_x,
                                                            nemitt_y    = nemitt_y,
                                                            sigma_z     = sigma_z,
                                                            line        = line,
                                                            _context    = context)

    n_parts  = len(particles.particle_id)
    #==============================



    
    # Finding number of chunks:
    #==============================
    n_chunks   = config['tracking']['n_chunks']
    main_chunk = config['tracking']['chunk_size']

    chunks   = xPlus.split_in_chunks(n_turns,n_chunks=n_chunks,main_chunk=main_chunk)
    #==============================


    

    # Creating data buffer and checkpoint if needed
    #==============================
    data_buffer = None
    if config['tracking']['data_path'] is not None:
        data_buffer = xPlus.Data_Buffer()

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
            tracked.to_parquet(config['tracking']['turn_b_turn_path'],partition_name='CHUNK',partition_ID=str(ID).zfill(ID_length))
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