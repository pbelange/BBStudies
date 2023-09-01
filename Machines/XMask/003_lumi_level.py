from scipy.constants import c as clight

import xtrack as xt
import xmask as xm
import xmask.lhc as xlhc

# Load collider anf build trackers
collider = xt.Multiline.from_json('collider_02_tuned_bb_off.json')
collider.build_trackers()

# Read knobs and tuning settings from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

config_lumi_leveling = config['config_lumi_leveling']
config_beambeam = config['config_beambeam']

xlhc.luminosity_leveling(
    collider, config_lumi_leveling=config_lumi_leveling,
    config_beambeam=config_beambeam)

# Re-match tunes, and chromaticities
conf_knobs_and_tuning = config['config_knobs_and_tuning']

for line_name in ['lhcb1', 'lhcb2']:
    knob_names = conf_knobs_and_tuning['knob_names'][line_name]
    targets = {
        'qx': conf_knobs_and_tuning['qx'][line_name],
        'qy': conf_knobs_and_tuning['qy'][line_name],
        'dqx': conf_knobs_and_tuning['dqx'][line_name],
        'dqy': conf_knobs_and_tuning['dqy'][line_name],
    }
    xm.machine_tuning(line=collider[line_name],
        enable_tune_correction=True, enable_chromaticity_correction=True,
        knob_names=knob_names, targets=targets)

collider.to_json('collider_03_tuned_and_leveled_bb_off.json')
