import numpy as np

import xtrack as xt
import xmask as xm

# Load collider
collider = xt.Multiline.from_json('collider_00_from_mad.json')

# Read beam-beam config from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)
config_bb = config['config_beambeam']

# Install beam-beam lenses (inactive and not configured)
collider.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
    delay_at_ips_slots=[0, 891, 0, 2670],
    num_long_range_encounters_per_side=
        config_bb['num_long_range_encounters_per_side'],
    num_slices_head_on=config_bb['num_slices_head_on'],
    harmonic_number=35640,
    bunch_spacing_buckets=config_bb['bunch_spacing_buckets'],
    sigmaz=config_bb['sigma_z'])

# Save to file
collider.to_json('collider_01_bb_off.json')
