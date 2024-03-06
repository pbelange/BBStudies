import numpy as np
import pandas as pd
from pathlib import Path
import gc
import subprocess
import time
import json
import ruamel.yaml

import BBStudies.Tracking.Utils as xutils
import BBStudies.Tracking.Jobs as Jobs


def run_jobs(user_context = 'GPU', device_id = 0):
    # Load Config
    #-------------------------
    config_file = 'configs/config_J002.yaml'
    tmp_file    = 'configs/tmp_config_J002_{user_context}_{device_id}.yaml'
    #-------------------------

    # Choose collider file from device_id
    collider_name   = 'BUNCH_0220'
    particle_name   = [ 'SERPENT_XY_ZETA_2',
                        'SERPENT_XY_ZETA_3',
                        'SERPENT_XY_ZETA_2',
                        'SERPENT_XY_ZETA_3',][device_id]




    # Update config
    config = xutils.read_YAML(config_file)

    # TRACKING
    #====================================
    config['tracking']['num_turns']                 = int(1e3)
    config['tracking']['size_chunks']               = int(100)
    #====================================

    #====================================
    config['tracking']['context']['type']           = user_context
    config['tracking']['context']['device_id']      = device_id
    #-----------------------
    config['tracking']['collider']['path']          = f'colliders/collider_{collider_name}.json'
    config['tracking']['collider']['name']          = collider_name
    config['tracking']['collider']['sequence']      = 'lhcb1'
    #-----------------------
    config['tracking']['collider']['cycle_at']      = 'IP3'
    config['tracking']['collider']['monitor_at']    = ['TCP_V', 'TCP_H', 'TCP_S'] 
    #====================================

    #====================================
    config['tracking']['particles']['path']         = f'particles/{particle_name}.parquet'
    config['tracking']['particles']['name']         = particle_name
    #====================================



    # ANALYSIS
    #====================================
    config['tracking']['analysis']['path']                      = 'tracking'

    config['tracking']['analysis']['turn_by_turn']['active']    = False
    config['tracking']['analysis']['checkpoints']['active']     = True
    config['tracking']['analysis']['excursion']['active']       = True
    config['tracking']['analysis']['naff']['active']            = True
    #-----------------------
    config['tracking']['analysis']['naff']['num_turns']         = int(500)
    config['tracking']['analysis']['naff']['num_harmonics']     = 10
    config['tracking']['analysis']['naff']['window_order']      = 4
    config['tracking']['analysis']['naff']['window_type']       = 'hann'
    config['tracking']['analysis']['naff']['multiprocesses']    = 4 
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
    # subprocess.run(["python", f"{Jobs.JOBS['J004']}/main.py",   "--data_path"       , f"{config['tracking']['data_path']}",
    #                                                             "--checkpoint_path" , f"{config['tracking']['checkpoint_path']}",
    #                                                             "--partition_name"  , f"{config['tracking']['partition_name']}",
    #                                                             "--partition_ID"    , f"{config['tracking']['partition_ID']}"])
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