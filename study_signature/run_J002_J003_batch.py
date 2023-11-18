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
config_file = 'configs/config_J002.yaml'
tmp_file    = 'configs/tmp_config_J002.yaml'

# Update config
config = xutils.read_YAML(config_file)
#config['save_collider'] = 'nothing'

# Save tmp file
xutils.save_YAML(config,file=tmp_file)

# Running Jop
subprocess.run(["python", f"{Jobs.JOBS['J002']}/main.py","-c", f"{tmp_file}"])
# subprocess.run(["python", f"{Jobs.JOBS['J003']}/main.py","-c", f"{tmp_file}"])
