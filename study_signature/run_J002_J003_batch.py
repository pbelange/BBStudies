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
    # Load Config
    #-------------------------
    config_file = 'configs/config_J002.yaml'
    tmp_file    = 'configs/tmp_config_J002_{user_context}_{device_id}.yaml'
    #-------------------------

    for i in [0,1,2]:


        # Update config
        #====================================
        config = xutils.read_YAML(config_file)

        config['tracking']['user_context']      = user_context
        config['tracking']['device_id']         = device_id

        # config['tracking']['collider_path']     = 'colliders/collider_BUNCH_0000.json'
        config['tracking']['collider_path']     = 'colliders/collider_BUNCH_0220.json'
        particle_file                           = f'XPLANE_ZETA_{i}'

        config['tracking']['partition_name']    = 'TEST'
        config['tracking']['partition_ID']      = f'BUNCH_0220_{particle_file}'
        config['tracking']['data_path']         = 'tracking/coupling_study/DATA'
        config['tracking']['checkpoint_path']   = 'tracking/coupling_study/CHECKPOINTS'
        #-----------------------------------------

        config['tracking']['turn_b_turn_path']  = f'tracking/coupling_study/FULL/TEST_{config["tracking"]["partition_ID"]}'

        config['tracking']['particles_path']    = f'particles/{particle_file}.parquet'
        #====================================




        # Save tmp file
        #-------------------------
        tmp_file = tmp_file.format( user_context = config['tracking']['user_context'],
                                    device_id    = config['tracking']['device_id'])
        xutils.save_YAML(config,file=tmp_file)
        #-------------------------


        # Running Jop
        #====================================
        subprocess.run(["python", f"{Jobs.JOBS['J002']}/main.py","-c", f"{tmp_file}"])
        subprocess.run(["python", f"{Jobs.JOBS['J003']}/main.py",   "--data_path"       , f"{config['tracking']['data_path']}",
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
    
    

    
    run_jobs(user_context = args.device, device_id = args.id)
    #===========================