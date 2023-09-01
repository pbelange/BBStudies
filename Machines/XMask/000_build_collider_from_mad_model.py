from cpymad.madx import Madx
import xtrack as xt

import xmask as xm
import xmask.lhc as xmlhc

import argparse
from pathlib import Path



#==========================================
# Parsing command line arguments for subprocess
#==========================================
aparser = argparse.ArgumentParser()

aparser.add_argument("-c", "--config"                   , help = "Path to config.yaml file"     ,default = 'config.yaml')
aparser.add_argument("-ost", "--optics_specific_tools"  , help = "Path to ost file"             ,default = 'optics_specific_tools.py')
args = aparser.parse_args()

assert Path(args.config).exists(), 'Invalid config path'
assert Path(args.optics_specific_tools).exists(), 'Invalid ost path'



#==========================================
# Import user-defined optics-specific tools and config
#==========================================
import sys
sys.path.append(str(Path(args.optics_specific_tools).parent))
ost =  __import__(Path(args.optics_specific_tools).stem)

with open(args.config,'r') as fid:
    config = xm.yaml.load(fid)
config_mad_model = config['config_mad']



#==========================================
# Proceed with main code block
#==========================================
# Create mad environment
if 'links' in config_mad_model.keys():
    xm.make_mad_environment(links=config_mad_model['links'])
else:
    xm.make_mad_environment(None)
# Start mad
mad_b1b2 = Madx(command_log="mad_collider.log")
mad_b4 = Madx(command_log="mad_b4.log")

# Build sequences
ost.build_sequence(mad_b1b2, mylhcbeam=1)
ost.build_sequence(mad_b4, mylhcbeam=4)

# Apply optics (only for b1b2, b4 will be generated from b1b2)
ost.apply_optics(mad_b1b2, optics_file=config_mad_model['optics_file'])

# Build xsuite collider
collider = xmlhc.build_xsuite_collider(
    sequence_b1=mad_b1b2.sequence.lhcb1,
    sequence_b2=mad_b1b2.sequence.lhcb2,
    sequence_b4=mad_b4.sequence.lhcb2,
    beam_config=config_mad_model['beam_config'],
    enable_imperfections=config_mad_model['enable_imperfections'],
    enable_knob_synthesis=config_mad_model['enable_knob_synthesis'],
    rename_coupling_knobs=config_mad_model['rename_coupling_knobs'],
    pars_for_imperfections=config_mad_model['pars_for_imperfections'],
    ver_lhc_run=config_mad_model['ver_lhc_run'],
    ver_hllhc_optics=config_mad_model['ver_hllhc_optics'])


# Save to file
collider.to_json('subp_lines/collider_000.json')


