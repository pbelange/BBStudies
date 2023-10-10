"""This script is used to install collimator aperture in the machine"""

# ==================================================================================================
# --- Imports
# ==================================================================================================
import json
import ruamel.yaml
from pathlib import Path
import numpy as np
import pandas as pd
import os
import xtrack as xt
import xmask as xm


# Initialize yaml reader
ryaml = ruamel.yaml.YAML()



# ==================================================================================================
# --- Functions to read configuration files and generate configuration files for orbit correction
# ==================================================================================================
def read_configuration(config_path="config.yaml"):
    # Read configuration for simulations
    with open(config_path, "r") as fid:
        config = ryaml.load(fid)

    return config




# ==================================================================================================
# --- Collimator class 
# ==================================================================================================
class collimator():
    def __init__(self,gap,gap_sigma = None,tilt=0,tilt_units='deg',center = [0,0]):

        self.gap = gap
        self.gap_sigma = gap_sigma
        if tilt_units == 'deg':
            self.tilt = np.deg2rad(tilt)

        self.center = center
        self.nominal_emittance = 3.5e-6 # [m]

        if gap is not None:
            self.element = self.make_element()
        


    def make_element(self,):
        # Creating vertices
        self.infinite_extent = 1 # [m]
        self.x_poly = np.array([-self.gap/2,-self.gap/2,self.gap/2,self.gap/2])
        self.y_poly = np.array([-self.infinite_extent,self.infinite_extent,self.infinite_extent,-self.infinite_extent])

        # Rotating vertices
        self.x_vertices = self.x_poly*np.cos(self.tilt) - self.y_poly*np.sin(self.tilt)
        self.y_vertices = self.x_poly*np.sin(self.tilt) + self.y_poly*np.cos(self.tilt)

        # Translating vertices
        self.x_vertices += self.center[0]
        self.y_vertices += self.center[1]

        return xt.LimitPolygon(x_vertices= self.x_vertices, y_vertices= self.y_vertices) 


# ==================================================================================================
# --- Function to install collimator in the line
# ==================================================================================================






# ==================================================================================================
# --- Main function
# ==================================================================================================
def user_specific_tasks(config_path       = "config.yaml",
                        collider_path     = "../001_configure_collider/zfruits/collider_001.json",
                        collider_out_path = "zfruits/collider_002.json",
                        collider          = None):
    
    # Loading config
    config_collimators = read_configuration(config_path=config_path)


    # Creating collimator objects
    apertures = {}
    for name,info in config_collimators['collimators'].items():
        apertures[name] = collimator(gap = info['gap'],tilt=info['tilt'],center=[0,0])

    # Loading line
    if collider is None:
        print('Loading collider...')
        collider = xt.Multiline.from_json(collider_path)

    # Installing collimators
    print('Installing collimators...')
    for seq in ['lhcb1','lhcb2']:
        beam = seq[-2:]
        for ee,aper in apertures.items():
            if beam not in ee:
                continue
            print(f'Installing {ee} in {seq}...')
            collider[seq].insert_element(element=aper.element, name=ee+'_aper', index=ee)


    # Exporting collider
    if collider_out_path is not None:
        print('Saving collider...')
        # Preparing output folder
        if not Path('zfruits').exists():
            Path('zfruits').mkdir()
        collider.to_json(collider_out_path)
    
    return collider

# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    collider = user_specific_tasks()
