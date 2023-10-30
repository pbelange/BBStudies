import numpy as np
import pandas as pd
from pathlib import Path
import gc
import subprocess
import time
import json
import ruamel.yaml

import main as main

# Initialize yaml reader
ryaml = ruamel.yaml.YAML()


config_path = 'config.yaml'
tmp_config  = 'tmp_config.yaml'

collider_file_template = '../001_configure_collider/zfruits/collider_BUNCHED/collider_BUNCH_{bunch_str}.json'
partition_path_template = 'zfruits/BBB_Signature/FULL/BUNCH_{bunch_str}'



bunch_start = 200
bunch_stop  = 247

for bunch in np.arange(bunch_start,bunch_stop+1):
    if bunch < 226:
        continue

    config = main.read_configuration(config_path)

    # Overwrite the bunch number
    config['tracking']['collider_path']  = collider_file_template.format(bunch_str=str(bunch).zfill(4))
    config['tracking']['bunch_number']   = int(bunch)

    if bunch in [202,203,204,210,211,212,218,219,220]:
        config['tracking']['partition_path'] = partition_path_template.format(bunch_str=str(bunch).zfill(4))
    else:
        config['tracking']['partition_path'] = None

    # Drop update configuration
    with open(tmp_config, "w") as fid:
        ryaml.dump(config, fid)


    # Tracking
    subprocess.run(["python", f"main.py","-c", f"{tmp_config}"])
    #------------------------------------------
    gc.collect()
    time.sleep(2)
    gc.collect()
    #------------------------------------------
