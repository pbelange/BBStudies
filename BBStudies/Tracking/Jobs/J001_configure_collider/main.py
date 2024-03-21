"""This script is used to configure the collider from config file to model realistic conditions."""

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

import BBStudies
from  BBStudies.Tracking.Jobs.J001_configure_collider.job_specific_tools import generate_orbit_correction_setup,luminosity_leveling, luminosity_leveling_ip1_5, compute_PU
import BBStudies.Tracking.Jobs.J001_configure_collider.bbcw_utils as bbcw_utils
import BBStudies.Tracking.Utils as xutils



# ==================================================================================================
# --- Functions to generate configuration files for orbit correction
# ==================================================================================================

def generate_configuration_correction_files(output_folder="_correction"):
    # Generate configuration files for orbit correction
    correction_setup = generate_orbit_correction_setup()
    os.makedirs(output_folder, exist_ok=True)
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"{output_folder}/corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)


# ==================================================================================================
# --- Function to install beam-beam
# ==================================================================================================
def install_beam_beam(collider, config_collider):
    # Load config
    config_bb = config_collider["config_beambeam"]

    # Install beam-beam lenses (inactive and not configured)
    collider.install_beambeam_interactions(
        clockwise_line="lhcb1",
        anticlockwise_line="lhcb2",
        ip_names=["ip1", "ip2", "ip5", "ip8"],
        delay_at_ips_slots=[0, 891, 0, 2670],
        num_long_range_encounters_per_side=config_bb["num_long_range_encounters_per_side"],
        num_slices_head_on=config_bb["num_slices_head_on"],
        harmonic_number=35640,
        bunch_spacing_buckets=config_bb["bunch_spacing_buckets"],
        sigmaz=config_bb["sigma_z"],
    )

    return collider, config_bb


# ==================================================================================================
# --- Function to match knobs and tuning
# ==================================================================================================
def set_knobs(config_collider, collider):
    # Read knobs and tuning settings from config file
    conf_knobs_and_tuning = config_collider["config_knobs_and_tuning"]

    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
        collider.vars[kk] = vv

    return collider, conf_knobs_and_tuning


def match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True):
    # Tunings
    for line_name in ["lhcb1", "lhcb2"]:
        knob_names = conf_knobs_and_tuning["knob_names"][line_name]

        targets = {
            "qx": conf_knobs_and_tuning["qx"][line_name],
            "qy": conf_knobs_and_tuning["qy"][line_name],
            "dqx": conf_knobs_and_tuning["dqx"][line_name],
            "dqy": conf_knobs_and_tuning["dqy"][line_name],
        }

        xm.machine_tuning(
            line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=match_linear_coupling_to_zero,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            line_co_ref=collider[line_name + "_co_ref"],
            co_corr_config=conf_knobs_and_tuning["closed_orbit_correction"][line_name],
        )

    return collider


# ==================================================================================================
# --- Function to compute the number of collisions in the IPs (used for luminosity leveling)
# ==================================================================================================
def compute_collision_from_scheme(pattern_fname):
    # # Get the filling scheme path (in json or csv format)
    # filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]


    # Load the filling scheme
    if pattern_fname.endswith(".json"):
        with open(pattern_fname, "r") as fid:
            filling_scheme = json.load(fid)
    else:
        raise ValueError(
            f"Unknown filling scheme file format: {pattern_fname}. It you provided a csv"
            " file, it should have been automatically convert when running the script"
            " 001_make_folders.py. Something went wrong."
        )

    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Assert that the arrays have the required length, and do the convolution
    assert len(array_b1) == len(array_b2) == 3564
    n_collisions_ip1_and_5 = array_b1 @ array_b2
    n_collisions_ip2 = np.roll(array_b1, 891) @ array_b2
    n_collisions_ip8 = np.roll(array_b1, 2670) @ array_b2

    return n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8


# ==================================================================================================
# --- Function to do the Levelling
# ==================================================================================================
def do_levelling(
    config_collider, config_bb, n_collisions_ip2, n_collisions_ip8, collider, n_collisions_ip1_and_5
):
    # Read knobs and tuning settings from config file (already updated with the number of collisions)
    config_lumi_leveling = config_collider["config_lumi_leveling"]

    # Update the number of bunches in the configuration file
    config_lumi_leveling["ip2"]["num_colliding_bunches"] = int(n_collisions_ip2)
    config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

    # First level luminosity in IP 1/5 changing the intensity
    if "config_lumi_leveling_ip1_5" in config_collider:
        print("Leveling luminosity in IP 1/5 varying the intensity")
        # Update the number of bunches in the configuration file
        config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"] = int(
            n_collisions_ip1_and_5
        )

        # Get crab cavities
        crab = False
        if "on_crab1" in config_collider["config_knobs_and_tuning"]["knob_settings"]:
            crab_val = float(
                config_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"]
            )
            if crab_val > 0:
                crab = True

        # Do the levelling
        try:
            I = luminosity_leveling_ip1_5(
                collider,
                config_collider,
                config_bb,
                crab=crab,
            )
        except ValueError:
            print("There was a problem during the luminosity leveling in IP1/5... Ignoring it.")
            I = config_bb["num_particles_per_bunch"]
            
        initial_I = config_bb["num_particles_per_bunch"]
        config_bb["num_particles_per_bunch"] = float(I)

    # Set up the constraints for lumi optimization in IP8
    additional_targets_lumi = []
    if "constraints" in config_lumi_leveling["ip8"]:
        for constraint in config_lumi_leveling["ip8"]["constraints"]:
            obs, beam, sign, val, at = constraint.split("_")
            target = xt.TargetInequality(obs, sign, float(val), at=at, line=beam, tol=1e-6)
            additional_targets_lumi.append(target)

    # Then level luminosity in IP 2/8 changing the separation
    collider = luminosity_leveling(
        collider,
        config_lumi_leveling=config_lumi_leveling,
        config_beambeam=config_bb,
        additional_targets_lumi=additional_targets_lumi,
        crab=crab,
    )

    # Update configuration
    config_bb["num_particles_per_bunch_before_optimization"] = float(initial_I)
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2h"] = float(
        collider.vars["on_sep2h"]._value
    )
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2v"] = float(
        collider.vars["on_sep2v"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8h"] = float(
        collider.vars["on_sep8h"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8v"] = float(
        collider.vars["on_sep8v"]._value
    )

    return collider, config_collider, crab


# ==================================================================================================
# --- Function to add linear coupling
# ==================================================================================================
def add_linear_coupling(conf_knobs_and_tuning, collider, config_mad):
    # Get the version of the optics
    version_hllhc = config_mad["ver_hllhc_optics"]
    version_run = config_mad["ver_lhc_run"]

    # Add linear coupling as the target in the tuning of the base collider was 0
    # (not possible to set it the target to 0.001 for now)
    if version_run == 3.0:
        collider.vars["cmrs.b1_sq"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["cmrs.b2_sq"] += conf_knobs_and_tuning["delta_cmr"]
    elif version_hllhc == 1.6 or version_hllhc == 1.5:
        collider.vars["c_minus_re_b1"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["c_minus_re_b2"] += conf_knobs_and_tuning["delta_cmr"]
    else:
        raise ValueError(f"Unknown version of the optics/run: {version_hllhc}, {version_run}.")

    return collider


# ==================================================================================================
# --- Function to assert that tune, chromaticity and linear coupling are correct before beam-beam
#     configuration
# ==================================================================================================
def assert_tune_chroma_coupling(collider, conf_knobs_and_tuning):
    for line_name in ["lhcb1", "lhcb2"]:
        tw = collider[line_name].twiss()
        assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], atol=1e-4), (
            f"tune_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qx'][line_name]}, got {tw.qx}"
        )
        assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], atol=1e-4), (
            f"tune_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qy'][line_name]}, got {tw.qy}"
        )
        assert np.isclose(
            tw.dqx,
            conf_knobs_and_tuning["dqx"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqx'][line_name]}, got {tw.dqx}"
        )
        assert np.isclose(
            tw.dqy,
            conf_knobs_and_tuning["dqy"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqy'][line_name]}, got {tw.dqy}"
        )

        assert np.isclose(
            tw.c_minus,
            conf_knobs_and_tuning["delta_cmr"],
            atol=5e-3,
        ), (
            f"linear coupling is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
        )


# ==================================================================================================
# --- Function to configure beam-beam
# ==================================================================================================
def configure_beam_beam(collider, config_bb,pattern_fname = ''):
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    # Configure filling scheme mask and bunch numbers
    if "mask_with_filling_pattern" in config_bb:
        # Initialize filling pattern with empty values
        filling_pattern_cw = None
        filling_pattern_acw = None

        # Initialize bunch numbers with empty values
        i_bunch_cw = None
        i_bunch_acw = None

        if pattern_fname != '' :
            # Fill values if possible
            with open(pattern_fname, "r") as fid:
                filling = json.load(fid)
            filling_pattern_cw = filling["beam1"]
            filling_pattern_acw = filling["beam2"]

            # Only track bunch number if a filling pattern has been provided
            if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
                i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
            if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
                i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

            # Note that a bunch number must be provided if a filling pattern is provided
            # Apply filling pattern
            collider.apply_filling_pattern(
                filling_pattern_cw=filling_pattern_cw,
                filling_pattern_acw=filling_pattern_acw,
                i_bunch_cw=i_bunch_cw,
                i_bunch_acw=i_bunch_acw,
            )
    return collider


# ==================================================================================================
# --- Function to compute luminosity once the collider is configured
# ==================================================================================================
def record_final_luminosity(collider, config_collider, l_n_collisions, crab):
    
    config_bb = config_collider["config_beambeam"]

    # Get the final luminoisty in all IPs
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()
    l_lumi = []
    l_PU = []
    l_ip = ["ip1", "ip2", "ip5", "ip8"]
    for n_col, ip in zip(l_n_collisions, l_ip):
        try:
            L = xt.lumi.luminosity_from_twiss(
                    n_colliding_bunches=n_col,
                    num_particles_per_bunch=config_bb["num_particles_per_bunch"],
                    ip_name=ip,
                    nemitt_x=config_bb["nemitt_x"],
                    nemitt_y=config_bb["nemitt_y"],
                    sigma_z=config_bb["sigma_z"],
                    twiss_b1=twiss_b1,
                    twiss_b2=twiss_b2,
                    crab=crab,
                )
            PU = compute_PU(L, n_col, twiss_b1["T_rev0"])
        except:
            print(f"There was a problem during the luminosity computation in {ip}... Ignoring it.")
            L = 0
            PU = 0
        l_lumi.append(L)
        l_PU.append(PU)

    # Update configuration
    for ip, L, PU in zip(l_ip, l_lumi, l_PU):
        config_collider["config_beambeam"][f"luminosity_{ip}_after_optimization"] = float(L)
        config_collider["config_beambeam"][f"Pile-up_{ip}_after_optimization"] = float(PU)
        
    return config_collider



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_collider(_collider,_config,file_path,include_config=False):
    if include_config:
        collider_dict = _collider.to_dict()
        collider_dict["config_yaml"] = _config

        #  collider.set_metadata(config_dict)
        with open(file_path, "w") as fid:
            json.dump(collider_dict, fid, cls=NpEncoder)
        
    else:
        collider.to_json(file_path)


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(config_file = 'config.yaml'):
    
    
    # Generate configuration files for orbit correction
    #-------------------
    generate_configuration_correction_files()
    #-------------------


    # Loading collider and configs
    #-------------------
    config      = xutils.read_YAML(file=config_file )
    collider    = xt.Multiline.from_json(config['from_collider'])
    config_mad  = collider.metadata['config_J000']
    config_collider = config["config_collider"]
    #-------------------


    # Preparing output folder
    #-------------------
    save_collider = config["save_collider"]

    if save_collider is not None:
        xutils.mkdir(save_collider)
    #-------------------

    # Finding Filling scheme
    #-------------------
    pattern_fname = config_collider['config_beambeam']['mask_with_filling_pattern']['pattern_fname']
    if not Path(pattern_fname).exists():
        pattern_fname = str(Path(BBStudies.__file__).parents[1]/pattern_fname)
    #-------------------
        
    # Install BBCW
    collider = bbcw_utils.install_BBCW(collider,L_phy=1,L_int=2)

    # Install beam-beam
    collider, config_bb = install_beam_beam(collider, config_collider)

    # Build trackers
    collider.build_trackers()

    # Set knobs
    collider, conf_knobs_and_tuning = set_knobs(config_collider, collider)

    # Match tune and chromaticity
    collider = match_tune_and_chroma(
        collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True
    )

    # Compute the number of collisions in the different IPs
    (
        n_collisions_ip1_and_5,
        n_collisions_ip2,
        n_collisions_ip8,
    ) = compute_collision_from_scheme(pattern_fname)

    # Do the leveling if requested
    if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling"]:
        collider, config_collider, crab = do_levelling(
            config_collider,
            config_bb,
            n_collisions_ip2,
            n_collisions_ip8,
            collider,
            n_collisions_ip1_and_5,
        )
    else:
        print(
            "No leveling is done as no configuration has been provided, or skip_leveling"
            " is set to True."
        )

        # Get crab cavities
        crab = False
        if "on_crab1" in config_collider["config_knobs_and_tuning"]["knob_settings"]:
            crab_val = float(
                config_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"]
            )
            if crab_val > 0:
                crab = True

    # Add linear coupling
    collider = add_linear_coupling(conf_knobs_and_tuning, collider, config_mad)

    # Rematch tune and chromaticity
    collider = match_tune_and_chroma(
        collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=False
    )

    # Assert that tune, chromaticity and linear coupling are correct one last time
    assert_tune_chroma_coupling(collider, conf_knobs_and_tuning)



    activate_beam_beam = config_collider['config_beambeam']['activate_beam_beam']
    if activate_beam_beam:
        # Configure beam-beam
        collider = configure_beam_beam(collider, config_bb,pattern_fname=pattern_fname)


    # Power BBCW
    collider,qff_knobs = bbcw_utils.power_BBCW(collider,config)
    collider.metadata['qff_knobs'] = qff_knobs


    # Update configuration with luminosity now that bb is known
    l_n_collisions  = [n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip1_and_5, n_collisions_ip8]
    config_collider = record_final_luminosity(collider,config_collider, l_n_collisions, crab)
    config['config_collider'] = config_collider


    # Saving resulting collider
    if save_collider is not None:
        # Save the final collider before tracking
        print(f'Saving "{save_collider}"')

        # Saving config in metadata
        collider.metadata['config_J001'] = config
        collider.to_json(save_collider)




    # Remove the correction folder, and potential C files remaining
    try:
        os.system("rm -rf _correction")
        os.system("rm -f *.cc")
    except:
        pass

    print('Job Finished!')
    return collider








# ==================================================================================================
# --- Script for execution
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-c", "--config",      help = "Config file"  ,default = 'config.yaml')
    args = aparser.parse_args()
    
    
    assert Path(args.config).exists(), 'Invalid config path'
    
    collider = configure_collider(config_file = args.config)
    #===========================

