"""
Loads a real CALCE INR-18650-20R DST Arbin (.xls) file and extracts
just the DST segment.

Inspected the file by grouping on Step_Index and checking each
group's current range, voltage range, sample count, and cumulative
charge/discharge capacity. That gives this structure:

  Sheet name : 'Channel_1-008'

  Step_Index breakdown:
    1 : single resting point at the very start
    2 : CC charge at 1A (0.5C) up toward 4.2V
    3 : CV taper at 4.2V (current falls off, still charging)
    4 : rest, cell now at ~100% SOC
    5 : constant -1A discharge for a fixed duration -- CALCE
        deliberately dropping the cell to a known starting SOC before
        the real DST test begins. In the 80SOC file this step removes
        0.4001 Ah (20.0% of the 2.0Ah rated capacity), landing the
        cell at 80% SOC -- this is where "80SOC" in the filename
        comes from.
    6 : rest
    7 : the actual DST profile. Current alternates between about
        -4.0A and +2.0A -- a real dynamic stress test, unlike the
        fixed-rate steps around it. This is the segment used here.
    8 : rest at the end

Step 5 (an unambiguous plain discharge) has negative current in the
raw file, so this file logs discharge as negative current; the sign
is flipped here so current > 0 means discharge everywhere else in the
project (matching models_real_data.py).
"""

import pandas as pd
import numpy as np

SHEET_NAME = "Channel_1-008"
STEP_INDEX_DST = 7


def load_dst_segment(filepath):
    """
    Returns:
      t : np.array, seconds, re-zeroed to start at 0
      I : np.array, amps, discharge positive (matches models_real_data.py)
      V : np.array, volts
    """
    df = pd.read_excel(filepath, sheet_name=SHEET_NAME)

    seg = df[df["Step_Index"] == STEP_INDEX_DST].reset_index(drop=True)

    t = seg["Test_Time(s)"].to_numpy()
    t = t - t[0]

    I_raw = seg["Current(A)"].to_numpy()
    I = -I_raw  # flip sign: file uses discharge-negative, we use discharge-positive

    V = seg["Voltage(V)"].to_numpy()

    return t, I, V


def resample_to_fixed_dt(t, I, V, dt=1.0):
    """
    Real Arbin logging isn't perfectly uniform, so this resamples
    onto a uniform dt-second grid by linear interpolation.
    """
    t_uniform = np.arange(t[0], t[-1], dt)
    I_uniform = np.interp(t_uniform, t, I)
    V_uniform = np.interp(t_uniform, t, V)
    return t_uniform, I_uniform, V_uniform


if __name__ == "__main__":
    t, I, V = load_dst_segment("data/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls")
    print(f"Raw DST segment: {len(t)} samples, {t[-1]:.1f} s "
          f"({t[-1]/60:.1f} min)")
    print(f"Current range (discharge positive): {I.min():.3f} to {I.max():.3f} A")
    print(f"Voltage range: {V.min():.3f} to {V.max():.3f} V "
          f"(V[0]={V[0]:.3f} -> V[-1]={V[-1]:.3f})")

    t_u, I_u, V_u = resample_to_fixed_dt(t, I, V, dt=1.0)
    print(f"Resampled to {len(t_u)} uniform 1-second samples")