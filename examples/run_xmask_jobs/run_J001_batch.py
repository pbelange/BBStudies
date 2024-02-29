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

# Config file
config_file = 'configs/config_J001.yaml'
tmp_file    = 'configs/tmp_config_J001.yaml'

# Update config
# # No beam-beam only
#=================================
# config = xutils.read_YAML(config_file)
# config['save_collider'] = '/home/HPC/phbelang/abp/BBStudies/study_signature/colliders/collider_BUNCH_0000.json'
# config['config_collider']['config_beambeam']['activate_beam_beam'] = False

# # No beam-beam and no octupoles
# =================================
# config = xutils.read_YAML(config_file)
# config['save_collider'] = '/home/HPC/phbelang/abp/BBStudies/study_signature/colliders/collider_NO_OCTU.json'
# config['config_collider']['config_beambeam']['activate_beam_beam'] = False
# config['config_collider']['config_knobs_and_tuning']['knob_settings']['i_oct_b1'] = 0
# config['config_collider']['config_knobs_and_tuning']['knob_settings']['i_oct_b2'] = 0



# Per bunch number
#=================================
bunch_start = 200
bunch_stop  = 247

# for bunch in np.arange(bunch_start,bunch_stop+1):
for bunch in [200,220]:
    bunch_str = str(bunch).zfill(4)
    
    config = xutils.read_YAML(config_file)
    config['save_collider'] = f'/home/HPC/phbelang/abp/BBStudies/study_signature/colliders/collider_BUNCH_{bunch_str}.json'
    config['config_collider']['config_beambeam']['mask_with_filling_pattern']['i_bunch_b1'] = bunch
    config['config_collider']['config_beambeam']['mask_with_filling_pattern']['i_bunch_b2'] = bunch


# Save tmp file
xutils.save_YAML(config,file=tmp_file)

# Running Jop
subprocess.run(["python", f"{Jobs.JOBS['J001']}/main.py","-c", f"{tmp_file}"])
