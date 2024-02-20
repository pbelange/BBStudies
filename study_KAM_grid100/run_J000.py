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
config_file = 'configs/config_J000.yaml'

# Running Jop
subprocess.run(["python", f"{Jobs.JOBS['J000']}/main.py","-c", f"{config_file}"])

