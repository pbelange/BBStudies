
import subprocess

import BBStudies.Tracking.Utils as xutils
import BBStudies.Tracking.Jobs as Jobs


# Config file
config_file = 'configs/config_J000.yaml'

# Running Jop
subprocess.run(["python", f"{Jobs.JOBS['J000']}/main.py","-c", f"{config_file}"])

