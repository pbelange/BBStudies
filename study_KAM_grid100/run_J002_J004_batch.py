import numpy as np
import pandas as pd
from pathlib import Path
import gc
import subprocess
import time
import json
import ruamel.yaml

import BBStudies.Tracking.XMask.Utils as xutils
import BBStudies.Tracking.XMask.Jobs as Jobs


def run_jobs(user_context = 'GPU', device_id = 0):

    # Choose collider file from device_id
    # collider_file   = 'BUNCH_0220'
    partition_name  = 'KAM'

    for collider_file in ['BUNCH_0000']:

        for zeta_sigma in [1]:

            particle_file   = [ f'XY_ZETA_{zeta_sigma}_GRID_1o4',
                                f'XY_ZETA_{zeta_sigma}_GRID_2o4',
                                f'XY_ZETA_{zeta_sigma}_GRID_3o4',
                                f'XY_ZETA_{zeta_sigma}_GRID_4o4'][device_id]



        
            # Load Config
            #-------------------------
            config_file = 'configs/config_J002.yaml'
            tmp_file    = 'configs/tmp_config_J002_{user_context}_{device_id}.yaml'
            #-------------------------



            # Update config
            #====================================
            config = xutils.read_YAML(config_file)

            config['tracking']['user_context']      = user_context
            config['tracking']['device_id']         = device_id

            config['tracking']['particles_path']    = f'particles/{particle_file}.parquet'
            config['tracking']['collider_path']     = f'colliders/collider_{collider_file}.json'
            


            config['tracking']['partition_name']    = partition_name
            config['tracking']['partition_ID']      = f'{collider_file}_{particle_file}'
            config['tracking']['data_path']         = 'tracking/DATA'
            config['tracking']['checkpoint_path']   = 'tracking/CHECKPOINTS'
            #-----------------------------------------

            config['tracking']['turn_b_turn_path']  = f'tracking/FULL/{partition_name}_{collider_file}_{particle_file}'
            config['tracking']['last_n_turns']      = None
            config['tracking']['n_turns']           = 1e4
            #====================================




            # Save tmp file
            #-------------------------
            tmp_file = tmp_file.format( user_context = config['tracking']['user_context'],
                                        device_id    = config['tracking']['device_id'])
            xutils.save_YAML(config,file=tmp_file)
            #-------------------------


            # Running Jop
            #====================================
            print(f'RUNNING FILE: {tmp_file}')
            subprocess.run(["python", f"{Jobs.JOBS['J002']}/main.py","-c", f"{tmp_file}"])
            subprocess.run(["python", f"{Jobs.JOBS['J004']}/main.py",   "--data_path"       , f"{config['tracking']['data_path']}",
                                                                        "--checkpoint_path" , f"{config['tracking']['checkpoint_path']}",
                                                                        "--partition_name"  , f"{config['tracking']['partition_name']}",
                                                                        "--partition_ID"    , f"{config['tracking']['partition_ID']}"])
            #====================================




# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-d", "--device"   ,help = "device [CPU | GPU]"    ,default = 'GPU')
    aparser.add_argument("-i", "--id"       ,help = "device ID"             ,default = 0)
    args = aparser.parse_args()
    
    
    run_jobs(user_context = args.device, device_id = int(args.id))
    #===========================