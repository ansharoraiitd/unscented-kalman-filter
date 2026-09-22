"""
Battery model for the INR-18650-20R cell, using the Zeng et al. (2023)
paper's own published numbers directly -- no fitting or identification
of my own. This gives a clean, apples-to-apples comparison against
their Table 3 results, since every number here comes straight from
the paper.

Source of every number below:
  - Uocv(SOC): Eq. (15), fit to their own INR-18650-20R OCV test.
  - R0, R1 (=Rp), C1 (=Cp): Table 2, identified via Recursive Least
    Squares on their own DST data at 25 degC. These are fixed values
    (not functions of SOC) -- a deliberate simplification the paper
    itself makes to reduce computation for real-time use.
  - Capacity: Table 1, Samsung INR-18650-20R, 2 Ah nominal.
"""

import numpy as np

NOMINAL_CAPACITY_AH = 2.0   # Table 1
CAPACITY_AS = 3600.0 * NOMINAL_CAPACITY_AH

# Fixed RC-branch parameters, Table 2 of the paper (DST, 25 degC)
R0 = 0.0715   # ohm
R1 = 0.0223   # ohm
C1 = 996.2    # farad


def voc_from_soc(soc):
    """
    Open-circuit voltage as a function of SOC, Eq. (15) of the paper.
    soc: percent, 0-100. Converted internally to fraction (0-1), since
    Eq. (15) is written in terms of SOC as a fraction.
    """
    z = np.clip(soc, 0.0, 100.0) / 100.0
    return (7.708 * z**6 - 18.26 * z**5 + 9.985 * z**4
            + 6.409 * z**3 - 7.569 * z**2 + 2.636 * z + 3.271)


def soc_dynamics(soc, current, dt):
    """Coulomb counting -- same logic as models.py, only the capacity differs."""
    soc_new = soc - (current * dt) / CAPACITY_AS * 100.0
    return np.clip(soc_new, 0.0, 100.0)


def v1_dynamics(v1, soc, current, dt):
    """Fixed R1/C1 (Table 2), discrete RC solution."""
    tau = R1 * C1
    alpha = np.exp(-dt / tau)
    return v1 * alpha + current * R1 * (1.0 - alpha)


def terminal_voltage(soc, v1, current):
    """Vt = Voc(SOC) - I*R0 - V1, fixed R0 (Table 2)."""
    return voc_from_soc(soc) - current * R0 - v1