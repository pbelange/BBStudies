

import numpy as np
import subprocess

import BBStudies.Tracking.Utils as xutils
import BBStudies.Tracking.Jobs as Jobs


def run_jobs(device_id = 0):


    # Parameters
    #=================================
    bunch_start = 200
    bunch_stop  = 200

    BBCW_ON = True
    #=================================


    # Prepare filepaths
    #-------------------------
    config_file = 'configs/config_J001.yaml'
    tmp_file    = 'configs/tmp_config_J001_{device_id}.yaml'

    if BBCW_ON:
        bbcw_current    = 350.0
        collider_name   = 'BBCW_BUNCH_{bunch_str}'
    else:
        bbcw_current    = int(0)
        collider_name   = 'BUNCH_{bunch_str}'
    #-------------------------
        

    # Loop over bunches
    # for bunch in np.arange(bunch_start,bunch_stop+1):
    for bunch in [200]:
        bunch_str = str(bunch).zfill(4)
    
        # Update config
        config = xutils.read_YAML(config_file)

        config['save_collider'] = f'colliders/collider_{collider_name.format(bunch_str=bunch_str)}.json'
        config['config_collider']['config_beambeam']['mask_with_filling_pattern']['i_bunch_b1'] = int(bunch)
        config['config_collider']['config_beambeam']['mask_with_filling_pattern']['i_bunch_b2'] = int(bunch)

        # BUNCH_0000 has beam-beam off, and does the powering of Q4
        if bunch == 0 :
            config['config_collider']['config_beambeam']['activate_beam_beam'] = False
            config['config_collider']['config_bbcw']['qff_file'] = None
        else:
            config['config_collider']['config_beambeam']['activate_beam_beam'] = True
            config['config_collider']['config_bbcw']['qff_file'] = 'colliders/collider_BBCW_BUNCH_0000.json'


        # Powering the wire, on or off
        for beam in ['b1','b2']:
            for ip in ['ip1','ip5']:
                config['config_collider']['config_bbcw'][beam][ip]['current'] = bbcw_current

        # Save tmp file
        #-------------------------
        tmp_file = tmp_file.format(device_id = device_id)
        xutils.save_YAML(config,file=tmp_file)
        #-------------------------


        # Running Jop
        #====================================
        print(f'RUNNING FILE: {tmp_file}')
        subprocess.run(["python", f"{Jobs.JOBS['J001']}/main.py","-c", f"{tmp_file}"])
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

