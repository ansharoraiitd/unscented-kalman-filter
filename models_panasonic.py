"""
Battery model for the Panasonic 18650PF cell, the benchmark cell used
in Guo, Li, Xue & Luo (2026), "State of Charge Estimation Method for
Lithium-Ion Batteries Based on Online Parameter Identification and
QPSO-AUKF," Batteries 12(3), 84 (Section 4.4, "Validation on an
Independent Wisconsin Dataset," their Table 4).

Unlike models_real_data.py, the benchmark paper does not publish R0,
R1, C1 or an OCV polynomial for THIS cell (their own fixed parameters
are for the Samsung INR-18650-20R dataset in Section 4.1-4.3, not the
Panasonic cell used for the Wisconsin-dataset comparison in Section
4.4). So every number below was identified directly from the raw
Panasonic 18650PF Li-ion Battery Data (P. Kollmeyer, Mendeley Data,
DOI 10.17632/wykht8y7tg.1) using standard, citable methods -- not
fitted to this project's own filter output, and not taken from the
benchmark paper.

OCV(SOC): degree-12 polynomial, fit to the dataset's own C/20
(0.145 A, ~C/20 for this 2.9 Ah cell) slow DISCHARGE leg specifically
("C20 OCV Test_C20_25dC.mat"), not averaged with the charge leg. Every
drive cycle in this benchmark (including UDDS) is overwhelmingly a
discharge profile, so the cell's open-circuit behaviour during it
tracks the discharge-leg curve more closely than a charge/discharge
average would (charge/discharge averaging, the standard way to cancel
polarization and open-circuit hysteresis when the direction of use is
mixed or unknown -- Plett, G.L. (2015), "Battery Management Systems,
Volume I: Battery Modeling," Artech House, Ch. 2 -- was checked
directly via an open-loop voltage simulation against the true
reference SOC and found to leave a larger residual bias here). The
discharge leg was converted to SOC via Ah integration, resampled onto
a grid densified below 10% SOC (where the curve is steepest), and fit
with a degree-12 polynomial: RMS residual 7.4 mV, max 100 mV,
confirmed strictly monotonically increasing over SOC in [0,1].

R0(SOC), R1(SOC), C1(SOC): identified from all fourteen 1C (2.899 A)
pulses in the dataset's own 5-pulse HPPC file
("25degC_5Pulse_HPPC_Pan18650PF.mat"), spanning ~100% down to ~8%
SOC. At each pulse, R0 is the average of the instantaneous voltage
step at pulse onset and at pulse release (divided by the pulse
current); R1 and the time constant tau=R1*C1 come from fitting a
single exponential, V(t) = V_inf - dV*exp(-t/tau), to the 20-minute
voltage relaxation immediately following the pulse -- the standard
HPPC parameter-extraction procedure (Chen & Rincon-Mora (2006),
already cited in models.py, use the same onset-step / relaxation-fit
logic for their own R_Series and R_Transient_S). R1 varies by close
to 18x across this SOC range (0.0067 ohm at 99.9% SOC vs. 0.119 ohm
at 7.9% SOC), so each of R0, R1, C1 is represented as a
piecewise-linear lookup table over the 14 identified SOC checkpoints
(_SOC_GRID below), interpolated with np.interp() -- the lookup-table
parameterization Plett (2015, Ch. 4) recommends when enough HPPC
checkpoints are available to resolve a parameter's SOC-dependence
directly, rather than assuming a parametric (e.g. exponential) form
for it the way models.py does for the Chen & Rincon-Mora cell.

Capacity: 2.997 Ah, measured from the cell's own C/20 discharge leg
(Ah removed from start-of-discharge to the 2.5 V cutoff), rather than
the 2.9 Ah nameplate rating -- the same "measure it from the data,
don't trust the nameplate" principle already used in
reference_soc.py's measured_capacity_ah() for the CALCE cell.
"""

import numpy as np

NOMINAL_CAPACITY_AH = 2.997  # measured from the cell's own C/20 discharge leg
CAPACITY_AS = 3600.0 * NOMINAL_CAPACITY_AH

# Fourteen SOC checkpoints (ascending) and the R0/R1/C1 identified at
# each one's 1C HPPC pulse + relaxation fit (see module docstring).
_SOC_GRID = np.array([
    7.94, 12.78, 17.61, 22.45, 27.29, 32.13, 41.80, 51.48,
    61.16, 70.83, 80.51, 90.19, 95.02, 99.86,
])
_R0_GRID = np.array([
    0.0257, 0.0279, 0.0258, 0.0214, 0.0207, 0.0189, 0.0198, 0.0189,
    0.0197, 0.0184, 0.0199, 0.0207, 0.0218, 0.0236,
])  # ohm
_R1_GRID = np.array([
    0.1193, 0.0254, 0.0074, 0.0063, 0.0058, 0.0058, 0.0056, 0.0055,
    0.0078, 0.0087, 0.0085, 0.0076, 0.0069, 0.0067,
])  # ohm
_C1_GRID = np.array([
    24.8, 318.7, 3419.8, 4739.7, 5777.9, 6146.5, 5988.1, 5413.7,
    6446.9, 4003.5, 3328.3, 2959.0, 3025.3, 3747.8,
])  # farad

# Degree-12 OCV(SOC) polynomial, SOC as a fraction (0-1), fit to the
# C/20 discharge leg (see module docstring).
_OCV_COEFFS = np.array([
    -80878.548,    # z^12
    509912.408,    # z^11
    -1416259.11,   # z^10
    2280645.45,    # z^9
    -2358474.81,   # z^8
    1639354.69,    # z^7
    -779287.274,   # z^6
    252844.144,    # z^5
    -54909.1978,   # z^4
    7667.33653,    # z^3
    -643.841256,   # z^2
    30.2418704,    # z^1
    2.67423574,    # z^0
])


def voc_from_soc(soc):
    """Open-circuit voltage as a function of SOC. soc: percent, 0-100."""
    z = np.clip(soc, 0.0, 100.0) / 100.0
    return np.polyval(_OCV_COEFFS, z)


def r0_from_soc(soc):
    """Series (ohmic) resistance, piecewise-linear lookup over _SOC_GRID."""
    return np.interp(soc, _SOC_GRID, _R0_GRID)


def r1_from_soc(soc):
    """RC-branch resistance, piecewise-linear lookup over _SOC_GRID."""
    return np.interp(soc, _SOC_GRID, _R1_GRID)


def c1_from_soc(soc):
    """RC-branch capacitance, piecewise-linear lookup over _SOC_GRID."""
    return np.interp(soc, _SOC_GRID, _C1_GRID)


def soc_dynamics(soc, current, dt):
    """Coulomb counting -- identical logic to models_real_data.py, only the capacity differs."""
    soc_new = soc - (current * dt) / CAPACITY_AS * 100.0
    return np.clip(soc_new, 0.0, 100.0)


def v1_dynamics(v1, soc, current, dt):
    """
    RC branch voltage, discrete solution of the RC ODE, with R1 and
    C1 evaluated at the CURRENT soc (same convention as models.py's
    SOC-dependent v1_dynamics for the Chen & Rincon-Mora cell).
    """
    r1 = r1_from_soc(soc)
    c1 = c1_from_soc(soc)
    tau = r1 * c1
    alpha = np.exp(-dt / tau)
    return v1 * alpha + current * r1 * (1.0 - alpha)


def terminal_voltage(soc, v1, current):
    """Vt = Voc(SOC) - I*R0(SOC) - V1, R0 evaluated at the current soc."""
    return voc_from_soc(soc) - current * r0_from_soc(soc) - v1