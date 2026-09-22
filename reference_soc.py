"""
Ampere-Hour (AH) counting reference SOC, following:

Zeng, Y., Li, Y., Yang, T. (2023), Equation (2):
SOC_t = SOC_0 - (1/C_N) * integral( eta * I dt )

This is the same reference method the paper uses as its "true" y_i
for computing MAE/RMSE -- they state explicitly that y_i is
calculated based on the AH method.

Nothing here depends on which file was loaded -- it just needs I, dt,
a starting SOC, and the cell's capacity.
"""

import numpy as np


def compute_ah_reference_soc(I, dt, soc0, capacity_ah, coulomb_eff=1.0,
                              clip=False):
    """
    I : np.array, amps, discharge positive
    dt: seconds between samples
    soc0: initial SOC, 0-100 (%)
    capacity_ah: battery capacity in Amp-hours
    coulomb_eff: eta (coulombic efficiency), ~1.0
    clip: if True, clamp result to [0,100]. Left off by default -- if
        the cell delivers slightly more than the assumed capacity_ah,
        letting the reference go a little negative is more honest
        than silently flattening it at 0%, which would distort the
        error at the end of the run.

    Returns: soc_ref, np.array, same length as I, in percent (0-100)
    """
    capacity_as = capacity_ah * 3600.0  # convert Ah to A-s (amp-seconds)

    # cumulative integral of I*dt -- i.e. running total charge removed
    charge_removed_as = np.cumsum(coulomb_eff * I * dt)

    soc_ref = soc0 - (charge_removed_as / capacity_as) * 100.0
    if clip:
        soc_ref = np.clip(soc_ref, 0.0, 100.0)

    return soc_ref


def measured_capacity_ah(I, dt, soc0_pct):
    """
    Calibrates the cell's actual delivered capacity from the run
    itself, rather than trusting the nameplate rating. Assuming the
    run ends at 0% SOC, the total charge removed corresponds to
    soc0_pct worth of capacity:
        capacity = total_charge_removed / (soc0_pct / 100)
    This keeps the AH reference anchored to what this specific run
    actually delivered.
    """
    total_removed_as = np.sum(I * dt)
    return (total_removed_as / 3600.0) / (soc0_pct / 100.0)


if __name__ == "__main__":
    # quick synthetic sanity check: 1A constant discharge, 1.1Ah battery
    dt = 1.0
    I_test = np.full(3600, 1.0)  # 1 hour at 1A
    soc = compute_ah_reference_soc(I_test, dt, soc0=100.0, capacity_ah=1.1)
    print(f"After 1 hour at 1A on a 1.1Ah battery, SOC dropped from 100% to {soc[-1]:.2f}%")
    # Expect ~9.1% -- 1Ah removed out of 1.1Ah capacity -> 100 - (1/1.1)*100 ~= 9.1%