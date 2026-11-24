"""
Loads a drive-cycle .mat file from the Panasonic 18650PF Li-ion
Battery Data set (P. Kollmeyer, Mendeley Data, DOI 10.17632/wykht8y7tg.1),
the benchmark dataset used in Guo, Li, Xue & Luo (2026), "State of
Charge Estimation Method for Lithium-Ion Batteries Based on Online
Parameter Identification and QPSO-AUKF," Batteries 12(3), 84.

File format: each .mat file is a single MATLAB struct named "meas"
with fields TimeStamp, Voltage, Current, Ah, Wh, Power,
Battery_Temp_degC, Time, Chamber_Temp_degC -- confirmed by directly
inspecting the file with scipy.io.loadmat(squeeze_me=True,
struct_as_record=False). Current is logged discharge-negative (same
convention as the CALCE Arbin files in data_loader.py), so the sign
is flipped here to match the discharge-positive convention used
throughout the rest of this project (models_panasonic.py,
models_real_data.py).

25degC/Drive cycles/ contains one file per drive cycle:
  Cycle_1 .. Cycle_4, US06, HWFTa, HWFTb, UDDS, LA92, NN
all run back-to-back from a full charge (SOC~100%) without an
intervening recharge, which is why the benchmark paper's Table 4
reports HWFET/LA92/UDDS as three separate single-cycle runs rather
than one long discharge, and why SOC0=100% is used below rather than
being backed out from a prior step the way SOC0=80% was for the
CALCE file (there is no equivalent "step 5" here).

Native sampling is ~10 Hz (dt ~ 0.1 s, confirmed by inspecting
np.diff(Time) directly), not 1 s -- resample_to_fixed_dt() is reused
from data_loader.py to bring it onto a uniform 1 s grid, both to
match the project's existing UKF timestep convention and to keep the
run length comparable to the CALCE DST segment.
"""

import numpy as np
import scipy.io as sio

from data_loader import resample_to_fixed_dt  # reused unchanged


def load_drive_cycle(filepath):
    """
    Returns:
      t  : np.array, seconds, re-zeroed to start at 0
      I  : np.array, amps, discharge positive (matches models_panasonic.py)
      V  : np.array, volts
      Ah : np.array, amp-hours, discharge-positive convention (i.e.
           negated from the raw file so it DECREASES during discharge,
           matching how Ah is used in reference_soc.py's
           compute_ah_reference_soc -- it is not used by that function
           directly, but is returned here in case a result needs to be
           cross-checked against the tester's own coulomb count).
    """
    raw = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    meas = raw["meas"]

    t = np.asarray(meas.Time, dtype=float)
    t = t - t[0]

    I_raw = np.asarray(meas.Current, dtype=float)
    I = -I_raw  # flip sign: file uses discharge-negative, we use discharge-positive

    V = np.asarray(meas.Voltage, dtype=float)

    Ah_raw = np.asarray(meas.Ah, dtype=float)
    Ah = -Ah_raw

    return t, I, V, Ah


if __name__ == "__main__":
    path = ("data/Panasonic_18650PF_Data/25degC/Drive cycles/"
             "03-21-17_00.29 25degC_UDDS_Pan18650PF.mat")
    t, I, V, Ah = load_drive_cycle(path)
    print(f"Raw UDDS cycle: {len(t)} samples, {t[-1]:.1f} s "
          f"({t[-1]/60:.1f} min), native sample rate ~{len(t)/t[-1]:.1f} Hz")
    print(f"Current range (discharge positive): {I.min():.3f} to {I.max():.3f} A")
    print(f"Voltage range: {V.min():.3f} to {V.max():.3f} V "
          f"(V[0]={V[0]:.3f} -> V[-1]={V[-1]:.3f})")
    print(f"Ah removed over this cycle (discharge positive): {Ah[-1] - Ah[0]:.4f} Ah")

    t_u, I_u, V_u = resample_to_fixed_dt(t, I, V, dt=1.0)
    print(f"Resampled to {len(t_u)} uniform 1-second samples")