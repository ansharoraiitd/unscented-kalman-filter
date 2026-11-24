"""
Runs the Sage-Husa Adaptive UKF on the same real CALCE INR-18650-20R
DST data as test_ukf_real_data.py, using the same model
(models_real_data.py), same noise injection, same starting
conditions -- the ONLY thing that changes is that Q and/or R are now
re-estimated online instead of held fixed at Q_PROCESS/MEAS_NOISE_VAR.
This isolates the adaptive mechanism as the one variable, the same
way test_ukf_real_data.py isolated the filter implementation against
Zeng et al. (2023).

Runs four configurations in one go, for the ablation table:
  1. fixed      -- baseline, reproduces test_ukf_real_data.py exactly
  2. adapt_Q    -- Sage-Husa on Q only, R stays fixed
  3. adapt_R    -- Sage-Husa on R only, Q stays fixed
  4. adapt_both -- Sage-Husa on both (headline result)

Prints a comparison table (MAE/RMSE for all four, plus the paper's
own published number) and saves one plot, real_data_results_adaptive.png
(SOC tracking + error, same layout as test_ukf_real_data.py's plot),
for the adapt_both run.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_dst_segment, resample_to_fixed_dt
from reference_soc import compute_ah_reference_soc, measured_capacity_ah
from ukf_real_data import ukf_step
from ukf_adaptive import ukf_step_adaptive, Q_PROCESS_INIT, MEAS_NOISE_VAR_INIT
import models_real_data

FILEPATH = "data/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls"
RATED_CAPACITY_AH = 2.0
DT = 1.0

SOC0 = 80.0
SOC_INIT = 90.0   # same deliberately-wrong start as test_ukf_real_data.py

VOLTAGE_NOISE_STD = 0.002
FORGETTING_FACTOR_B = 0.98

np.random.seed(0)   # same seed as test_ukf_real_data.py -> same noise realization

t_raw, I_raw, V_raw = load_dst_segment(FILEPATH)
t, I, V_clean = resample_to_fixed_dt(t_raw, I_raw, V_raw, dt=DT)
N = len(t)

cap_actual = measured_capacity_ah(I, DT, SOC0)
V_measured = V_clean + np.random.normal(0, VOLTAGE_NOISE_STD, N)
soc_reference = compute_ah_reference_soc(
    I, DT, soc0=SOC0, capacity_ah=cap_actual, clip=False)


def run_fixed():
    """Reproduces test_ukf_real_data.py exactly, for the baseline row."""
    x = np.array([SOC_INIT, 0.0])
    P = np.array([[0.05**2, 0.0], [0.0, 0.005**2]])
    soc_estimates = np.zeros(N)
    soc_estimates[0] = x[0]
    for k in range(1, N):
        x, P = ukf_step(
            x, P, current=I[k], dt=DT, measured_voltage=V_measured[k],
            soc_dynamics=models_real_data.soc_dynamics,
            v1_dynamics=models_real_data.v1_dynamics,
            terminal_voltage=models_real_data.terminal_voltage,
        )
        soc_estimates[k] = x[0]
    return soc_estimates


def run_adaptive(adapt_Q, adapt_R, b=FORGETTING_FACTOR_B):
    """
    One Sage-Husa AUKF run. adapt_Q/adapt_R select which ablation
    this is (Q-only, R-only, or both).
    Returns soc_estimates.
    """
    x = np.array([SOC_INIT, 0.0])
    P = np.array([[0.05**2, 0.0], [0.0, 0.005**2]])
    Q = Q_PROCESS_INIT.copy()
    R = MEAS_NOISE_VAR_INIT

    soc_estimates = np.zeros(N)
    soc_estimates[0] = x[0]

    for k in range(1, N):
        x, P, Q, R, innovation = ukf_step_adaptive(
            x, P, Q, R, current=I[k], dt=DT, measured_voltage=V_measured[k],
            k=k, b=b, adapt_Q=adapt_Q, adapt_R=adapt_R,
            soc_dynamics=models_real_data.soc_dynamics,
            v1_dynamics=models_real_data.v1_dynamics,
            terminal_voltage=models_real_data.terminal_voltage,
        )
        soc_estimates[k] = x[0]

    return soc_estimates


def mae_rmse(soc_est):
    error = soc_reference - soc_est
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    return mae, rmse, error


# ---------------- run all four configurations ----------------
soc_fixed = run_fixed()
soc_Q_only = run_adaptive(adapt_Q=True, adapt_R=False)
soc_R_only = run_adaptive(adapt_Q=False, adapt_R=True)
soc_both = run_adaptive(adapt_Q=True, adapt_R=True)

mae_fixed, rmse_fixed, _ = mae_rmse(soc_fixed)
mae_Q, rmse_Q, _ = mae_rmse(soc_Q_only)
mae_R, rmse_R, _ = mae_rmse(soc_R_only)
mae_both, rmse_both, error_both = mae_rmse(soc_both)

print("Results on real DST data (INR-18650-20R cell, paper's own model parameters):")
print(f"  {'Filter':<30s} {'MAE (%)':>10s} {'RMSE (%)':>10s}")
print(f"  {'Fixed UKF (baseline)':<30s} {mae_fixed:>10.4f} {rmse_fixed:>10.4f}")
print(f"  {'Sage-Husa AUKF (Q only)':<30s} {mae_Q:>10.4f} {rmse_Q:>10.4f}")
print(f"  {'Sage-Husa AUKF (R only)':<30s} {mae_R:>10.4f} {rmse_R:>10.4f}")
print(f"  {'Sage-Husa AUKF (Q and R)':<30s} {mae_both:>10.4f} {rmse_both:>10.4f}")
print(f"  {'Zeng et al. (2023), Table 3':<30s} {0.93:>10.4f} {1.22:>10.4f}")

# ---------------- SOC tracking + error plot, adapt_both run ----------------
# Same layout/conventions as test_ukf_real_data.py's plot, so the two
# are directly comparable figure-for-figure.
soc_reference_frac = soc_reference / 100.0
soc_both_frac = soc_both / 100.0
error_both_frac = error_both / 100.0

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].plot(t, soc_reference_frac, label='AH', linewidth=1.0)
axes[0].plot(t, soc_both_frac, label='Adaptive UKF', linewidth=1.0, alpha=0.85)
axes[0].set_xlabel('Time/s')
axes[0].set_ylabel('SOC')
axes[0].set_ylim(-0.1, 0.9)
axes[0].set_title('DST@25C -- Sage-Husa Adaptive UKF (Q and R)')
axes[0].legend()
axes[0].grid(alpha=0.3)

inset = axes[0].inset_axes([0.12, 0.12, 0.42, 0.30])
mask = (t >= 3650) & (t <= 3820)
inset.plot(t[mask], soc_reference_frac[mask], linewidth=1.0)
inset.plot(t[mask], soc_both_frac[mask], linewidth=1.0, alpha=0.85)
inset.grid(alpha=0.3)
inset.tick_params(labelsize=7)

axes[1].plot(t, error_both_frac, linewidth=0.5)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_xlabel('Time/s')
axes[1].set_ylabel('Error')
axes[1].set_ylim(-0.2, 0.25)
axes[1].set_title(f'Error  [MAE={mae_both/100:.4f}, RMSE={rmse_both/100:.4f}]')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('real_data_results_adaptive.png', dpi=150)
print("Plot saved to real_data_results_adaptive.png")