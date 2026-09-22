"""
Battery model equations used for the synthetic convergence experiment.

All equations and numerical parameters below are taken directly from:

  Chen, M. and Rincon-Mora, G.A. (2006). "Accurate Electrical Battery
  Model Capable of Predicting Runtime and I-V Performance."
  IEEE Transactions on Energy Conversion, 21(2), pp. 504-511.

Specifically, Equations (2)-(5) (SOC in FRACTION form, 0 to 1, as used
in the source paper), fitted from real pulse-discharge test data on a
polymer Li-ion battery (TCL PL-383562, 850 mAh nominal capacity).

This uses their Thevenin-style RSeries + one RC branch (RTransient_S,
CTransient_S), i.e. the SHORT time-constant branch only, giving a
2-state model: state = [SOC, V1], where V1 is the voltage across the
RC branch. The long time-constant branch (RTransient_L, CTransient_L,
Eqs. 6-7) is not modeled here.
"""

import numpy as np

NOMINAL_CAPACITY_AH = 0.85          # 850 mAh, Chen & Rincon-Mora test cell
CAPACITY_AS = 3600.0 * NOMINAL_CAPACITY_AH   # Eq. (1): C_Capacity = 3600*Capacity (A-s)


def voc_from_soc(soc):
    """
    Open-circuit voltage as a function of SOC.
    Chen & Rincon-Mora (2006), Equation (2).
    soc: percent, 0-100. Converted internally to fraction (0-1).
    """
    z = np.clip(soc, 0.0, 100.0) / 100.0
    return (-1.031 * np.exp(-35.0 * z) + 3.685 + 0.2156 * z
            - 0.1178 * z**2 + 0.3201 * z**3)


def r_series_from_soc(soc):
    """
    Series resistance R_Series(SOC).
    Chen & Rincon-Mora (2006), Equation (3).
    """
    z = np.clip(soc, 0.0, 100.0) / 100.0
    return 0.1562 * np.exp(-24.37 * z) + 0.07446


def r_transient_from_soc(soc):
    """
    RC branch resistance R_Transient_S(SOC).
    Chen & Rincon-Mora (2006), Equation (4).
    """
    z = np.clip(soc, 0.0, 100.0) / 100.0
    return 0.3208 * np.exp(-29.14 * z) + 0.04669


def c_transient_from_soc(soc):
    """
    RC branch capacitance C_Transient_S(SOC).
    Chen & Rincon-Mora (2006), Equation (5).
    """
    z = np.clip(soc, 0.0, 100.0) / 100.0
    return -752.9 * np.exp(-13.51 * z) + 703.6


def soc_dynamics(soc, current, dt):
    """
    Process model for SOC: coulomb counting.
    Sign convention: current > 0 = discharge (SOC decreases).
    """
    soc_new = soc - (current * dt) / CAPACITY_AS * 100.0
    return np.clip(soc_new, 0.0, 100.0)


def v1_dynamics(v1, soc, current, dt):
    """
    Process model for V1: RC branch voltage, discrete-time solution
    of the RC charge/discharge ODE. R and C are evaluated at the
    current SOC via Eqs. (4)-(5), so they vary with SOC rather than
    being fixed constants.

    tau = R1*C1;  V1[k+1] = V1[k]*exp(-dt/tau) + I*R1*(1 - exp(-dt/tau))
    """
    r1 = r_transient_from_soc(soc)
    c1 = c_transient_from_soc(soc)
    tau = r1 * c1
    alpha = np.exp(-dt / tau)
    v1_new = v1 * alpha + current * r1 * (1.0 - alpha)
    return v1_new


def terminal_voltage(soc, v1, current):
    """
    Measurement model: Vt = Voc(SOC) - I*RSeries(SOC) - V1
    """
    return voc_from_soc(soc) - current * r_series_from_soc(soc) - v1