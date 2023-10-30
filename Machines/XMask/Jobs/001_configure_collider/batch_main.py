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
collider_file_template = 'zfruits/collider_BUNCHED/collider_BUNCH_{bunch_str}.json'


# Preparing output folder
if not Path('zfruits').exists():
    Path('zfruits').mkdir()

# Preparing output folder
if not Path('zfruits/collider_BUNCHED').exists():
    Path('zfruits/collider_BUNCHED').mkdir()


bunch_start = 200
bunch_stop  = 247

for bunch in np.arange(bunch_start,bunch_stop+1):

    with open(config_path, "r") as fid:
        config = ryaml.load(fid)

    # Overwrite the bunch number
    config['config_simulation']['collider_file_out'] = collider_file_template.format(bunch_str=str(bunch).zfill(4))
    config["config_collider"]["config_beambeam"]["mask_with_filling_pattern"]["i_bunch_b1"] = int(bunch)
    config["config_collider"]["config_beambeam"]["mask_with_filling_pattern"]["i_bunch_b2"] = int(bunch)

    # Running mask
    main.configure_collider(config = config)
