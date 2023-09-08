
"""This script is used to build the base collider with Xmask, configuring only the optics. Functions
in this script are called sequentially."""
# ==================================================================================================
# --- Imports
# ==================================================================================================
from cpymad.madx import Madx
import os
import xmask as xm
import xmask.lhc as xlhc
import shutil
import yaml
from pathlib import Path

# Import user-defined optics-specific tools
import optics_specific_tools as ost





# ==================================================================================================
# --- Function to load configuration file
# ==================================================================================================
def load_configuration(config_path="config.yaml"):
    # Load configuration
    with open(config_path, "r") as fid:
        configuration = yaml.safe_load(fid)

    # Get configuration for the particles distribution and the collider separately
    config_mad = configuration["config_mad"]

    return configuration, config_mad



# ==================================================================================================
# --- Function to build collider from mad model
# ==================================================================================================
def build_collider_from_mad(config_mad, sanity_checks=True):
    # Make mad environment
    for kk in config_mad['links'].keys():
        if not Path(kk).parent.exists():
            Path(kk).parent.mkdir()
    xm.make_mad_environment(links=config_mad['links'])


    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")

    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    ost.build_sequence(mad_b1b2, mylhcbeam=1)
    ost.build_sequence(mad_b4, mylhcbeam=4)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    ost.apply_optics(mad_b1b2, optics_file=config_mad["optics_file"])

    if sanity_checks:
        mad_b1b2.use(sequence="lhcb1")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)
        mad_b1b2.use(sequence="lhcb2")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Apply optics (only for b4, just for check)
    ost.apply_optics(mad_b4, optics_file=config_mad["optics_file"])
    if sanity_checks:
        mad_b4.use(sequence="lhcb2")
        mad_b4.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Build xsuite collider
    collider = xlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config=config_mad["beam_config"],
        enable_imperfections=config_mad["enable_imperfections"],
        enable_knob_synthesis=config_mad["enable_knob_synthesis"],
        rename_coupling_knobs=config_mad["rename_coupling_knobs"],
        pars_for_imperfections=config_mad["pars_for_imperfections"],
        ver_lhc_run=config_mad["ver_lhc_run"],
        ver_hllhc_optics=config_mad["ver_hllhc_optics"],
    )
    collider.build_trackers()

    if sanity_checks:
        collider["lhcb1"].twiss(method="4d")
        collider["lhcb2"].twiss(method="4d")
    # Return collider
    return collider


def activate_RF_and_twiss(collider, config_mad, sanity_checks=True):
    # Define a RF system (values are not so immportant as they're defined later)
    print("--- Now Computing Twiss assuming:")
    if config_mad["ver_hllhc_optics"] == 1.6:
        dic_rf = {"vrf400": 16.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.5}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    elif config_mad["ver_lhc_run"] == 3.0:
        dic_rf = {"vrf400": 12.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.0}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    print("---")

    # Rebuild tracker if needed
    try:
        collider.build_trackers()
    except:
        print("Skipping rebuilding tracker")

    for knob, val in dic_rf.items():
        collider.vars[knob] = val

    if sanity_checks:
        for my_line in ["lhcb1", "lhcb2"]:
            ost.check_xsuite_lattices(collider[my_line])

    return collider


def clean():
    # Remove all the temporaty files created in the process of building collider
    os.remove("mad_collider.log")
    os.remove("mad_b4.log")
    os.unlink("errors")
    shutil.rmtree("__pycache__")
    shutil.rmtree("temp")
    shutil.rmtree("acc-models-lhc")


# ==================================================================================================
# --- Main function for building distribution and collider
# ==================================================================================================
def build_collider(config_file="config.yaml"):
    # Get configuration
    configuration, config_mad = load_configuration(config_file)

    # Get sanity checks flag
    sanity_checks = configuration["sanity_checks"]

    # Build collider from mad model
    collider = build_collider_from_mad(config_mad, sanity_checks)

    # Twiss to ensure eveyrthing is ok
    collider = activate_RF_and_twiss(collider, config_mad, sanity_checks)

    # Clean temporary files
    clean()

    # Save to file
    if not Path('zfruits').exists():
        Path('zfruits').mkdir()
    collider.to_json('zfruits/collider_000.json')


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    build_collider()






