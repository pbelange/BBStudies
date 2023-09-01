import json

import xmask as xm
import xtrack as xt

# Read beam-beam config from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)
config_bb = config['config_beambeam']

# Load collider and build trackers
collider = xt.Multiline.from_json('collider_03_tuned_and_leveled_bb_off.json')
collider.build_trackers()

# Configure beam-beam lenses
print('Configuring beam-beam lenses...')
collider.configure_beambeam_interactions(
    num_particles=config_bb['num_particles_per_bunch'],
    nemitt_x=config_bb['nemitt_x'],
    nemitt_y=config_bb['nemitt_y'])

if 'mask_with_filling_pattern' in config_bb:
    fname = config_bb['mask_with_filling_pattern']['pattern_fname']
    i_bunch_cw = config_bb['mask_with_filling_pattern']['i_bunch_b1']
    i_bunch_acw = config_bb['mask_with_filling_pattern']['i_bunch_b2']
    with open(fname, 'r') as fid:
        filling = json.load(fid)

    collider.apply_filling_pattern(
        filling_pattern_cw=filling['beam1'],
        filling_pattern_acw=filling['beam2'],
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

collider.to_json('collider_04_tuned_and_leveled_bb_on.json')
