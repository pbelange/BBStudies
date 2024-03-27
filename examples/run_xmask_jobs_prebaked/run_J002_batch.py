

import numpy as np
import subprocess

import BBStudies.Tracking.Utils as xutils
import BBStudies.Tracking.Jobs as Jobs


def run_jobs(device_id = 0):


    # Config file
    config_file = 'configs/config_J002.yaml'

    # Running Jop
    #====================================
    print(f'RUNNING FILE: {config_file}')
    subprocess.run(["python", f"{Jobs.JOBS['J002']}/main.py","-c", f"{config_file}"])
    #====================================




# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-i", "--id"       ,help = "device ID"             ,default = 0)
    args = aparser.parse_args()
    
    
    run_jobs(device_id = int(args.id))
    #===========================

