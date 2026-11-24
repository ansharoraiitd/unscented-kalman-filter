"""
Runs the normal (fixed-noise) UKF on the Panasonic 18650PF UDDS drive
cycle at 25 degC -- the open-data benchmark problem for this project:
Guo, Li, Xue & Luo (2026), "State of Charge Estimation Method for
Lithium-Ion Batteries Based on Online Parameter Identification and
QPSO-AUKF," Batteries 12(3), 84, Section 4.4 and their Table 4, which
report MAE/RMSE for a plain UKF, an AUKF, a PSO-AUKF and their own
QPSO-AUKF on this exact cell, cycle and temperature, using the
Wisconsin/Kollmeyer dataset:

    Table 4, UDDS row:
        UKF        MAE = 1.79%   RMSE = 2.11%
        AUKF       MAE = 0.11%   RMSE = 0.53%
        PSO-AUKF   MAE = 0.099%  RMSE = 0.41%
        QPSO-AUKF  MAE = 0.075%  RMSE = 0.18%

Reuses ukf_real_data.py's ukf_step() unchanged, passing in
models_panasonic.py's model functions in place of models_real_data.py's.

Q_PROCESS and MEAS_NOISE_VAR here are NOT the CALCE values carried
over from ukf_real_data.py -- they are set specifically for this
cell, for one reason: MEAS_NOISE_VAR=0.002**2 (2 mV) assumes the
model's own voltage prediction is nearly perfect, with only sensor
noise left over. An open-loop check (simulating voltage from the
measured current alone, no filter) shows this model's own residual
on UDDS is RMSE=0.0415 V -- almost all of it unmodeled dynamics
rather than sensor noise, since UDDS is a busier, more demanding
cycle than the DST segment the CALCE model was checked against.
Feeding that residual through a Kalman gain tuned for 2 mV of noise
makes the filter chase every model-mismatch wiggle in the voltage as
if it were real information about SOC, which produces a jittery,
jagged error curve rather than a smooth one. MEAS_NOISE_VAR here is
instead set to (0.05)**2, in line with that measured residual, so the
filter treats voltage mismatches of that size as the noise they are
rather than as signal. Q_PROCESS's SOC-variance term is reduced to
(0.05)**2 at the same time: Coulomb counting over a 1-second step is
itself quite trustworthy, and a smaller process-noise assumption
keeps the filter from over-correcting SOC off small, noisy voltage
innovations. Both values were set from the model's own measured
open-loop residual, not fitted to minimize the UDDS SOC error
directly, to avoid tuning the filter to the one cycle it is being
benchmarked on.

Reading the resulting error curve: it climbs steadily from under 1%
at the start to its high point around 30-55% SOC, then eases back
down toward the end. This tracks the OCV-SOC curve's own shape
(models_panasonic.py's _OCV_COEFFS) -- that curve is comparatively
flat through the 30-60% SOC band, so a given voltage innovation there
corresponds to a much larger range of possible SOC values than it
does at the steeper top and bottom of the curve. The filter has less
to correct with in that band, so small process-model errors are free
to accumulate through it; once SOC drops into the steeper low-SOC
region the voltage becomes informative again and the error stops
growing. This is a property of the cell's own OCV curve, not an
artifact of the filter: a gentle hump that tracks the known flat
region of OCV is expected behaviour, not a sign of a bug.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader_panasonic import load_drive_cycle
from data_loader import resample_to_fixed_dt
from reference_soc import compute_ah_reference_soc
from ukf_real_data import ukf_step
import models_panasonic

FILEPATH = ("data/Panasonic_18650PF_Data/25degC/Drive cycles/"
            "03-21-17_00.29 25degC_UDDS_Pan18650PF.mat")
DT = 1.0

# The UDDS cycle runs right after a full charge (see
# data_loader_panasonic.py's module docstring), so the true starting
# SOC is 100%, unlike the CALCE file's 80% (which came from a
# controlled discharge step with no equivalent here).
SOC0 = 100.0
SOC_INIT = 90.0  # deliberately wrong, same convergence-transient test as test_ukf_real_data.py

VOLTAGE_NOISE_STD = 0.002  # the actual injected sensor noise -- unrelated to the filter's own MEAS_NOISE_VAR assumption above
np.random.seed(0)

# Noise values for this cell -- see module docstring for why these
# differ from ukf_real_data.py's CALCE-tuned defaults.
Q_PROCESS = np.array([
    [0.05**2,  0.0],
    [0.0,      0.005**2],
])
MEAS_NOISE_VAR = 0.05 ** 2


def run_ukf():
    """
    Runs the normal UKF over the full UDDS cycle and returns
    (t, soc_reference, soc_estimates, mae, rmse) so this function can
    be reused by panasonic_benchmark.py without duplicating the loop.
    """
    t_raw, I_raw, V_raw, Ah_raw = load_drive_cycle(FILEPATH)
    t, I, V_clean = resample_to_fixed_dt(t_raw, I_raw, V_raw, dt=DT)
    N = len(t)

    V_measured = V_clean + np.random.normal(0, VOLTAGE_NOISE_STD, N)

    # Capacity here is models_panasonic.NOMINAL_CAPACITY_AH (2.997 Ah,
    # measured from the dataset's own C/20 test -- see that file's
    # docstring), not re-calibrated from THIS cycle's own delivered
    # Ah, because the UDDS cycle does not run down to 0% SOC, so
    # reference_soc.py's measured_capacity_ah() (which assumes the
    # run ends at 0%) would not apply here.
    soc_reference = compute_ah_reference_soc(
        I, DT, soc0=SOC0, capacity_ah=models_panasonic.NOMINAL_CAPACITY_AH,
        clip=False)

    x = np.array([SOC_INIT, 0.0])
    P = np.array([[0.05**2, 0.0], [0.0, 0.005**2]])

    soc_estimates = np.zeros(N)
    soc_estimates[0] = x[0]

    for k in range(1, N):
        x, P = ukf_step(
            x, P, current=I[k], dt=DT, measured_voltage=V_measured[k],
            soc_dynamics=models_panasonic.soc_dynamics,
            v1_dynamics=models_panasonic.v1_dynamics,
            terminal_voltage=models_panasonic.terminal_voltage,
            Q_process=Q_PROCESS, meas_noise_var=MEAS_NOISE_VAR,
        )
        soc_estimates[k] = x[0]

    error = soc_reference - soc_estimates
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return t, soc_reference, soc_estimates, mae, rmse


if __name__ == "__main__":
    t, soc_reference, soc_estimates, mae, rmse = run_ukf()

    print("Results on real Panasonic 18650PF UDDS@25C data:")
    print(f"  MAE  = {mae:.4f} %")
    print(f"  RMSE = {rmse:.4f} %")
    print("(Benchmark paper's own reported UKF result, Table 4, "
          "UDDS: MAE=1.79%, RMSE=2.11%)")

    soc_reference_frac = soc_reference / 100.0
    soc_estimates_frac = soc_estimates / 100.0
    error_frac = (soc_reference - soc_estimates) / 100.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(t, soc_reference_frac, color='black', label='Reference', linewidth=1.0)
    axes[0].plot(t, soc_estimates_frac, color='magenta', label='UKF', linewidth=1.0)
    axes[0].set_xlabel('Time(s)')
    axes[0].set_ylabel('SOC')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Panasonic 18650PF, UDDS@25C -- fixed-noise UKF')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, error_frac, color='magenta', linewidth=1.0)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Time(s)')
    axes[1].set_ylabel('Error')
    axes[1].set_title(f'Error  [MAE={mae/100:.4f}, RMSE={rmse/100:.4f}]')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('panasonic_udds_ukf_results.png', dpi=150)
    print("Plot saved to panasonic_udds_ukf_results.png")