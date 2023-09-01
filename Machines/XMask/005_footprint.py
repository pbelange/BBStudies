import xtrack as xt


collider = xt.Multiline.from_json('./collider_04_tuned_and_leveled_bb_on.json')
collider.build_trackers()

fp_polar_no_rescale = collider['lhcb1'].get_footprint(nemitt_x=2.5e-6, nemitt_y=2.5e-6)

fp_polar_with_rescale = collider['lhcb1'].get_footprint(
    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
    linear_rescale_on_knobs=[
        xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)]
    )

#!end-doc-part

fp_ua_with_rescale = collider['lhcb1'].get_footprint(
    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
    mode='uniform_action_grid',
    linear_rescale_on_knobs=[
        xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)]
    )

import matplotlib.pyplot as plt

plt.close('all')

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
fp_polar_no_rescale.plot(ax=ax1, label='no rescale bb')
plt.suptitle('Polar mode - no rescale on beambeam (default)')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, sharex=ax1, sharey=ax1)
fp_polar_with_rescale.plot(ax=ax2, label='rescale bb')
plt.suptitle('Polar mode - linear rescale on beambeam')

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, sharex=ax1, sharey=ax1)
fp_ua_with_rescale.plot()
plt.suptitle('Uniform action grid mode - linear rescale on beambeam')

plt.show()
