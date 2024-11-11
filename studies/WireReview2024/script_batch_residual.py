import concurrent.futures
import subprocess
import shutil
from tqdm import tqdm 
from pathlib import Path


import BBStudies.Tracking.Utils as xutils

# def main(CONFIG):

# try:
import argparse
# Adding command line parser
aparser = argparse.ArgumentParser()
aparser.add_argument("-c"   , "--config"        , help = "Config file"          , default = 'config.yaml')
args = aparser.parse_args()

CONFIG = args.config


config = xutils.read_YAML(CONFIG)

# Copy config in destination folder
out_path = config['out_path']

assert not Path(out_path).exists(), f"Output folder {out_path} already exists. Please remove it before running the script."


xutils.mkdir(out_path)
shutil.copy(CONFIG, out_path + '/' + Path(CONFIG).name)


# Function to run a script
Jids = range(int(config['n_I']*config['n_d']))
def run_script(Jid):
    result = subprocess.run(["python", "script_compute_residual.py","-c",f"{CONFIG}","-id",f"{Jid}"], capture_output=True)
    return result.stdout.decode(), result.stderr.decode()

# Function to run a script
Rids = range(11)
def run_reference(Rid):
    result = subprocess.run(["python", "script_reference_residual.py","-c",f"{CONFIG}","-id",f"{Rid}"], capture_output=True)
    return result.stdout.decode(), result.stderr.decode()


# Limit the number of processes to 20
assert config['max_workers']*config['num_threads'] <= 120, "Too many workers. Please reduce the number of workers."
max_workers = config['max_workers']

# Use ProcessPoolExecutor to run up to 20 processes concurrently
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all jobs to the executor
    futures =   [executor.submit(run_script, Jid) for Jid in Jids]
    futures +=  [executor.submit(run_reference, Rid) for Rid in Rids]

    # Process the results as they complete
    # for future in concurrent.futures.as_completed(futures):
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(Jids) + len(Rids), desc="Processing jobs", unit="job"):
        stdout, stderr = future.result()
        # print(f"Output: {stdout}")
        if stderr:
            print(f"Error: {stderr}")


print("All scripts completed.")



# # ==================================================================================================
# # --- Script for execution
# # ==================================================================================================
# if __name__ == '__main__':
    