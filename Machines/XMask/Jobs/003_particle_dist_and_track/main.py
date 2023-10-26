
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
import sys
sys.path.append('/Users/pbelanger/ABPLocal/BBStudies')
sys.path.append('/home/phbelang/abp/BBStudies')
import BBStudies.Tracking.XsuitePlus as xPlus
import BBStudies.Tracking.Progress as pbar
import BBStudies.Tracking.InteractionPoint as inp
import BBStudies.Physics.Detuning as tune
import BBStudies.Plotting.BBPlots as bbplt
import BBStudies.Physics.Base as phys
import BBStudies.Physics.Constants as cst


# JOB imports
import importlib
sys.path.append('../../')
main_002 = importlib.import_module('Jobs.002_user_specific_tasks.main')
user_specific_tasks = main_002.user_specific_tasks





#xPlus = importlib.reload(xPlus)


# ==================================================================================================
# --- Functions to read configuration files and generate configuration files for orbit correction
# ==================================================================================================
import ruamel.yaml
ryaml = ruamel.yaml.YAML()
def read_configuration(config_path="config.yaml"):
    # Read configuration for simulations
    with open(config_path, "r") as fid:
        config = ryaml.load(fid)


    return config





# ==================================================================================================
# --- Functions to load collider with a given context
# ==================================================================================================
def load_collider(collider_path = '../001_configure_collider/zfruits/collider_001.json',user_context = 'CPU',device_id = 0):


    # Load collider and install collimators
    # collider = user_specific_tasks( config_path       = "../002_user_specific_tasks/config.yaml",
    #                                 collider_path     = "../001_configure_collider/zfruits/collider_001.json",
    #                                 collider_out_path = None,
    #                                 collider          = None)


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




def particle_dist_and_track():

    # Loading setup
    #----------------------------------
    # Loading config
    config = read_configuration('config.yaml')
    assert config['tracking']['partition_turns'] != config['tracking']['process_data'] , 'Cannot partition turns and process data at the same time'

    # Loading collider
    print('LOADING COLLIDER')
    collider,context = load_collider(   collider_path = config['tracking']['collider_path'],
                                        user_context  = config['tracking']['user_context'],
                                        device_id     = config['tracking']['device_id'])

    # Parsing config
    sequence = config['tracking']['sequence']
    line     = collider[sequence]
    n_parts  = int(config['tracking']['n_parts'])
    n_turns  = int(config['tracking']['n_turns'])
    monitor_at_dict = config['elements'][sequence]
    monitor_at      = config['tracking']['monitor_at']
    if monitor_at in monitor_at_dict.keys():
        monitor_at = monitor_at_dict[monitor_at]
    #----------------------------------


    # Generating particle distribution
    #----------------------------------
    # Extracting emittance from previous config
    config_bb    = read_configuration('../001_configure_collider/config.yaml')
    beam = sequence[-2:]
    bunch_number = config_bb['config_collider']['config_beambeam']['mask_with_filling_pattern'][f'i_bunch_{beam}'] 
    nemitt_x,nemitt_y = (config_bb['config_collider']['config_beambeam'][f'nemitt_{plane}'] for plane in ['x','y'])
    #-----------------------
    print('GENERATING PARTICLES')
    particles,coordinates = generate_particles( n_part      = n_parts,
                                                force_n_part= False,
                                                nemitt_x    = nemitt_x,
                                                nemitt_y    = nemitt_y,
                                                line        = line,
                                                at_element  = monitor_at,
                                                 _context   = context)
    n_parts  = len(particles.particle_id)
    #----------------------------------



    
    # Finding number of chunks:
    #==============================
    if config['tracking']['partition_turns']:
        n_chunks   = config['tracking']['partition_n_chunks']
        main_chunk = None
    elif config['tracking']['process_data']:
        n_chunks   = config['tracking']['process_n_chunks']
        main_chunk = config['tracking']['process_turn_chunk']
    else:
        n_chunks   = 1
        main_chunk = None
    chunks   = xPlus.split_in_chunks(n_turns,n_chunks=n_chunks,main_chunk=main_chunk)
    #==============================

    # Preparing output folder
    #==============================
    if not Path('zfruits').exists():
        Path('zfruits').mkdir()
    #==============================


    def initialize_monitor(context = None,num_particles = 0,start_at_turn=0,nturns = 1):
        monitor = xt.ParticlesMonitor( _context       = context,
                                        num_particles = num_particles,
                                        start_at_turn = start_at_turn, 
                                        stop_at_turn  = start_at_turn + nturns)
        return monitor

    # Creating data buffer if needed
    #----------------------------------
    data_buffer = None
    if config['tracking']['process_data']:
        data_buffer = xPlus.Data_Buffer()
    #----------------------------------


    # Tracking
    #----------------------------------
    print('START TRACKING...')
    parquet_path = config['tracking']['parquet_path']
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
                                            nemitt_zeta = 1,
                                            Pbar        = PBar,
                                            progress_divide = 100)
        #--------------------------------------------


        # Data Buffer Computation
        #---------------
        if config['tracking']['process_data']:
            data_buffer.process(monitor=main_monitor)
        #---------------

        # Saving Chunk if needed
        #--------------------------
        if config['tracking']['partition_turns']:
            # print(f'SAVING TO PARQUET... -> {parquet_path}')
            tracked.to_parquet(parquet_path,partition_name='CHUNK',partition_ID=str(ID).zfill(ID_length))
        #--------------------------

    
    # Saving data buffer if needed
    #--------------------------
    if config['tracking']['process_data']:
        tracked.exec_time    = PBar.main_task.finished_time
        tracked.parquet_data = '_data'
        tracked._data        = data_buffer.to_pandas()

        tracked.to_parquet(parquet_path,partition_name='DATA',partition_ID='0001')




    return particles,tracked


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    particles,tracked = particle_dist_and_track()

