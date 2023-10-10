# Imports
import json
import logging
from scipy.constants import c as clight
from scipy.optimize import minimize_scalar
import xtrack as xt
import numpy as np


# Function to generate dictionnary containing the orbit correction setup
def generate_orbit_correction_setup():
    correction_setup = {}
    correction_setup["lhcb1"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r8.b1",
            end="e.ds.l1.b1",
            vary=(
                "corr_co_acbh14.l1b1",
                "corr_co_acbh12.l1b1",
                "corr_co_acbv15.l1b1",
                "corr_co_acbv13.l1b1",
            ),
            targets=("e.ds.l1.b1",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b1",
            end="s.ds.l2.b1",
            vary=(
                "corr_co_acbh13.r1b1",
                "corr_co_acbh15.r1b1",
                "corr_co_acbv12.r1b1",
                "corr_co_acbv14.r1b1",
            ),
            targets=("s.ds.l2.b1",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r4.b1",
            end="e.ds.l5.b1",
            vary=(
                "corr_co_acbh14.l5b1",
                "corr_co_acbh12.l5b1",
                "corr_co_acbv15.l5b1",
                "corr_co_acbv13.l5b1",
            ),
            targets=("e.ds.l5.b1",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b1",
            end="s.ds.l6.b1",
            vary=(
                "corr_co_acbh13.r5b1",
                "corr_co_acbh15.r5b1",
                "corr_co_acbv12.r5b1",
                "corr_co_acbv14.r5b1",
            ),
            targets=("s.ds.l6.b1",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b1",
            end="s.ds.r1.b1",
            vary=(
                "corr_co_acbch6.l1b1",
                "corr_co_acbcv5.l1b1",
                "corr_co_acbch5.r1b1",
                "corr_co_acbcv6.r1b1",
                "corr_co_acbyhs4.l1b1",
                "corr_co_acbyhs4.r1b1",
                "corr_co_acbyvs4.l1b1",
                "corr_co_acbyvs4.r1b1",
            ),
            targets=("ip1", "s.ds.r1.b1"),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l2.b1",
            end="s.ds.r2.b1",
            vary=(
                "corr_co_acbyhs5.l2b1",
                "corr_co_acbchs5.r2b1",
                "corr_co_acbyvs5.l2b1",
                "corr_co_acbcvs5.r2b1",
                "corr_co_acbyhs4.l2b1",
                "corr_co_acbyhs4.r2b1",
                "corr_co_acbyvs4.l2b1",
                "corr_co_acbyvs4.r2b1",
            ),
            targets=("ip2", "s.ds.r2.b1"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b1",
            end="s.ds.r5.b1",
            vary=(
                "corr_co_acbch6.l5b1",
                "corr_co_acbcv5.l5b1",
                "corr_co_acbch5.r5b1",
                "corr_co_acbcv6.r5b1",
                "corr_co_acbyhs4.l5b1",
                "corr_co_acbyhs4.r5b1",
                "corr_co_acbyvs4.l5b1",
                "corr_co_acbyvs4.r5b1",
            ),
            targets=("ip5", "s.ds.r5.b1"),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l8.b1",
            end="s.ds.r8.b1",
            vary=(
                "corr_co_acbch5.l8b1",
                "corr_co_acbyhs4.l8b1",
                "corr_co_acbyhs4.r8b1",
                "corr_co_acbyhs5.r8b1",
                "corr_co_acbcvs5.l8b1",
                "corr_co_acbyvs4.l8b1",
                "corr_co_acbyvs4.r8b1",
                "corr_co_acbyvs5.r8b1",
            ),
            targets=("ip8", "s.ds.r8.b1"),
        ),
    }

    correction_setup["lhcb2"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b2",
            end="e.ds.r8.b2",
            vary=(
                "corr_co_acbh13.l1b2",
                "corr_co_acbh15.l1b2",
                "corr_co_acbv12.l1b2",
                "corr_co_acbv14.l1b2",
            ),
            targets=("e.ds.r8.b2",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l2.b2",
            end="s.ds.r1.b2",
            vary=(
                "corr_co_acbh12.r1b2",
                "corr_co_acbh14.r1b2",
                "corr_co_acbv13.r1b2",
                "corr_co_acbv15.r1b2",
            ),
            targets=("s.ds.r1.b2",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b2",
            end="e.ds.r4.b2",
            vary=(
                "corr_co_acbh13.l5b2",
                "corr_co_acbh15.l5b2",
                "corr_co_acbv12.l5b2",
                "corr_co_acbv14.l5b2",
            ),
            targets=("e.ds.r4.b2",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l6.b2",
            end="s.ds.r5.b2",
            vary=(
                "corr_co_acbh12.r5b2",
                "corr_co_acbh14.r5b2",
                "corr_co_acbv13.r5b2",
                "corr_co_acbv15.r5b2",
            ),
            targets=("s.ds.r5.b2",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b2",
            end="e.ds.l1.b2",
            vary=(
                "corr_co_acbch6.r1b2",
                "corr_co_acbcv5.r1b2",
                "corr_co_acbch5.l1b2",
                "corr_co_acbcv6.l1b2",
                "corr_co_acbyhs4.l1b2",
                "corr_co_acbyhs4.r1b2",
                "corr_co_acbyvs4.l1b2",
                "corr_co_acbyvs4.r1b2",
            ),
            targets=(
                "ip1",
                "e.ds.l1.b2",
            ),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r2.b2",
            end="e.ds.l2.b2",
            vary=(
                "corr_co_acbyhs5.l2b2",
                "corr_co_acbchs5.r2b2",
                "corr_co_acbyvs5.l2b2",
                "corr_co_acbcvs5.r2b2",
                "corr_co_acbyhs4.l2b2",
                "corr_co_acbyhs4.r2b2",
                "corr_co_acbyvs4.l2b2",
                "corr_co_acbyvs4.r2b2",
            ),
            targets=("ip2", "e.ds.l2.b2"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b2",
            end="e.ds.l5.b2",
            vary=(
                "corr_co_acbch6.r5b2",
                "corr_co_acbcv5.r5b2",
                "corr_co_acbch5.l5b2",
                "corr_co_acbcv6.l5b2",
                "corr_co_acbyhs4.l5b2",
                "corr_co_acbyhs4.r5b2",
                "corr_co_acbyvs4.l5b2",
                "corr_co_acbyvs4.r5b2",
            ),
            targets=(
                "ip5",
                "e.ds.l5.b2",
            ),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r8.b2",
            end="e.ds.l8.b2",
            vary=(
                "corr_co_acbchs5.l8b2",
                "corr_co_acbyhs5.r8b2",
                "corr_co_acbcvs5.l8b2",
                "corr_co_acbyvs5.r8b2",
                "corr_co_acbyhs4.l8b2",
                "corr_co_acbyhs4.r8b2",
                "corr_co_acbyvs4.l8b2",
                "corr_co_acbyvs4.r8b2",
            ),
            targets=(
                "ip8",
                "e.ds.l8.b2",
            ),
        ),
    }
    return correction_setup


def luminosity_leveling(
    collider,
    config_lumi_leveling,
    config_beambeam,
    additional_targets_lumi=[],
    crab=False,
):
    for ip_name in config_lumi_leveling.keys():
        print(f"\n --- Leveling in {ip_name} ---")

        config_this_ip = config_lumi_leveling[ip_name]
        bump_range = config_this_ip["bump_range"]

        assert config_this_ip[
            "preserve_angles_at_ip"
        ], "Only preserve_angles_at_ip=True is supported for now"
        assert config_this_ip[
            "preserve_bump_closure"
        ], "Only preserve_bump_closure=True is supported for now"

        beta0_b1 = collider.lhcb1.particle_ref.beta0[0]
        f_rev = 1 / (collider.lhcb1.get_length() / (beta0_b1 * clight))

        targets = []
        vary = []

        if "luminosity" in config_this_ip.keys():
            targets.append(
                xt.TargetLuminosity(
                    ip_name=ip_name,
                    luminosity=config_this_ip["luminosity"],
                    crab=crab,
                    tol=1e30,  # 0.01 * config_this_ip["luminosity"],
                    f_rev=f_rev,
                    num_colliding_bunches=config_this_ip["num_colliding_bunches"],
                    num_particles_per_bunch=config_beambeam["num_particles_per_bunch"],
                    sigma_z=config_beambeam["sigma_z"],
                    nemitt_x=config_beambeam["nemitt_x"],
                    nemitt_y=config_beambeam["nemitt_y"],
                    log=True,
                )
            )

            # Added this line for constraints
            targets.extend(additional_targets_lumi)
        elif "separation_in_sigmas" in config_this_ip.keys():
            targets.append(
                xt.TargetSeparation(
                    ip_name=ip_name,
                    separation_norm=config_this_ip["separation_in_sigmas"],
                    tol=1e-4,  # in sigmas
                    plane=config_this_ip["plane"],
                    nemitt_x=config_beambeam["nemitt_x"],
                    nemitt_y=config_beambeam["nemitt_y"],
                )
            )
        else:
            raise ValueError("Either `luminosity` or `separation_in_sigmas` must be specified")

        if config_this_ip["impose_separation_orthogonal_to_crossing"]:
            targets.append(xt.TargetSeparationOrthogonalToCrossing(ip_name="ip8"))
        vary.append(xt.VaryList(config_this_ip["knobs"], step=1e-4))

        # Target and knobs to rematch the crossing angles and close the bumps
        for line_name in ["lhcb1", "lhcb2"]:
            targets += [
                # Preserve crossing angle
                xt.TargetList(
                    ["px", "py"], at=ip_name, line=line_name, value="preserve", tol=1e-7, scale=1e3
                ),
                # Close the bumps
                xt.TargetList(
                    ["x", "y"],
                    at=bump_range[line_name][-1],
                    line=line_name,
                    value="preserve",
                    tol=1e-5,
                    scale=1,
                ),
                xt.TargetList(
                    ["px", "py"],
                    at=bump_range[line_name][-1],
                    line=line_name,
                    value="preserve",
                    tol=1e-5,
                    scale=1e3,
                ),
            ]

        vary.append(xt.VaryList(config_this_ip["corrector_knob_names"], step=1e-7))

        # Match
        collider.match(
            lines=["lhcb1", "lhcb2"],
            ele_start=[bump_range["lhcb1"][0], bump_range["lhcb2"][0]],
            ele_stop=[bump_range["lhcb1"][-1], bump_range["lhcb2"][-1]],
            twiss_init="preserve",
            targets=targets,
            vary=vary,
        )
        
    return collider


def compute_PU(luminosity, num_colliding_bunches, T_rev0, cross_section = 81e-27):
    return (
            luminosity
            / num_colliding_bunches
            * cross_section
            * T_rev0
        )

def luminosity_leveling_ip1_5(
    collider,
    config_collider,
    config_bb,
    crab=False,
):
    # Get Twiss
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()

    def compute_lumi(I):
        luminosity = xt.lumi.luminosity_from_twiss(
            n_colliding_bunches=config_collider["config_lumi_leveling_ip1_5"][
                "num_colliding_bunches"
            ],
            num_particles_per_bunch=I,
            ip_name="ip1",
            nemitt_x=config_bb["nemitt_x"],
            nemitt_y=config_bb["nemitt_y"],
            sigma_z=config_bb["sigma_z"],
            twiss_b1=twiss_b1,
            twiss_b2=twiss_b2,
            crab=crab,
        )
        return luminosity

    def f(I):
        luminosity = compute_lumi(I)

        PU = compute_PU(luminosity, config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"], twiss_b1["T_rev0"])
        penalty_PU = max(
            0,
            (PU - config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_PU"]) * 1e35,
        )

        return (
            abs(luminosity - config_collider["config_lumi_leveling_ip1_5"]["luminosity"])
            + penalty_PU
        )

    # Do the optimization
    res = minimize_scalar(
        f,
        bounds=(
            1e10,
            float(config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_intensity"]),
        ),
        method="bounded",
        options={"xatol": 1e7},
    )
    if not res.success:
        logging.warning("Optimization for leveling in IP 1/5 failed. Please check the constraints.")
    else:
        print(
            f"Optimization for leveling in IP 1/5 succeeded with I={res.x:.2e} particles per bunch"
        )
    return res.x


if __name__ == "__main__":
    correction_setup = generate_orbit_correction_setup()
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)
