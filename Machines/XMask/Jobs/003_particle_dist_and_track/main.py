
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
def load_collider(collider_path = '../001_configure_collider/zfruits/collider_001.json',user_context = 'CPU'):


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
        context = xo.ContextCupy()
    collider.build_trackers(_context=context)
    #--------------------------------------

    return collider,context



# ==================================================================================================
# --- Functions to generate particle distribution
# ==================================================================================================
def generate_particles(n_part = 1000,force_n_part = False,line = None,_context = None,nemitt_x = None,nemitt_y = None):

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
                                        _context=_context)
    else:
        particles = None


    
    return particles,coordinates




def particle_dist_and_track():

    # Loading setup
    #----------------------------------
    # Loading config
    config = read_configuration('config.yaml')

    # Loading collider
    print('LOADING COLLIDER')
    collider,context = load_collider(   collider_path = config['tracking']['collider_path'],
                                        user_context  = config['tracking']['user_context'])

    # Parsing config
    sequence = config['tracking']['sequence']
    line     = collider[sequence]
    n_parts  = int(config['tracking']['n_parts'])
    n_turns  = int(config['tracking']['n_turns'])
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
                                                 _context   = context)
    #----------------------------------


    # Tracking
    #----------------------------------
    print('START TRACKING...')
    tracked = xPlus.Tracking_Interface( line      = line,
                                        particles = particles,
                                        n_turns   = n_turns,
                                        progress  = True,
                                        rebuild   = False,
                                        monitor   = None,
                                        method    ='6D',
                                        _context   = context)

    # Saving emittance:
    tracked.nemitt_x    = nemitt_x
    tracked.nemitt_y    = nemitt_y
    tracked.nemitt_zeta = 1#nemitt_zeta
    #----------------------------------



    # Saving results
    #----------------------------------
    # Preparing output folder
    if not Path('zfruits').exists():
        Path('zfruits').mkdir()
        

    # Setting Bunch number for partitionning
    parquet_path = config['tracking']['tracking_path']
    bunch_ID     = str(bunch_number).zfill(4)

    print(f'SAVING TO PARQUET... -> {parquet_path}')
    tracked.to_parquet(parquet_path,partition_name='BUNCH',partition_ID=bunch_ID)
    #----------------------------------

    return particles,tracked


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    particles,tracked = particle_dist_and_track()

