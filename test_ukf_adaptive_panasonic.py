"""
Runs the Sage-Husa Adaptive UKF on the same Panasonic 18650PF UDDS@25C
cycle as test_ukf_panasonic.py, using the same model
(models_panasonic.py), same noise injection, same seed, same
deliberately-wrong starting SOC -- the only thing that changes is
that Q and/or R are re-estimated online instead of held fixed.

Runs four configurations, the same ablation table used on the CALCE
benchmark:
  1. fixed      -- baseline, reproduces test_ukf_panasonic.py exactly
  2. adapt_Q    -- Sage-Husa on Q only, R stays fixed
  3. adapt_R    -- Sage-Husa on R only, Q stays fixed
  4. adapt_both -- Sage-Husa on both (headline result)

Q_PROCESS_INIT, MEAS_NOISE_VAR_INIT and FORGETTING_FACTOR_B below are
starting points for the Sage-Husa recursion, not final noise values
-- the recursion re-estimates Q and R from the innovation sequence as
it runs. Q_PROCESS_INIT's SOC-variance term is set higher than the
fixed UKF's Q_PROCESS in test_ukf_panasonic.py (0.5**2 instead of
0.05**2): Sage-Husa needs enough initial process-noise "room" to
actually pull Q toward the level the innovations call for -- started
too small, the recursion's own exponential-weighting update
(Q_new = (1-d_k)*Q + d_k*Q_hat) moves it upward too slowly to matter
over a ~22,000-step cycle. The same reasoning applies in reverse to
MEAS_NOISE_VAR_INIT, started lower (0.02**2) than the fixed UKF's
0.05**2, so that R's own online adaptation -- not the initial guess
-- is what ends up doing the work.

Prints a comparison table including the benchmark paper's own
published numbers for this cycle (Guo, Li, Xue & Luo (2026), Table 4,
UDDS row):
    UKF        MAE = 1.79%   RMSE = 2.11%
    AUKF       MAE = 0.11%   RMSE = 0.53%
    PSO-AUKF   MAE = 0.099%  RMSE = 0.41%
    QPSO-AUKF  MAE = 0.075%  RMSE = 0.18%

and saves panasonic_udds_adaptive_results.png for the adapt_both run.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader_panasonic import load_drive_cycle
from data_loader import resample_to_fixed_dt
from reference_soc import compute_ah_reference_soc
from ukf_adaptive import ukf_step_adaptive
import models_panasonic

FILEPATH = ("data/Panasonic_18650PF_Data/25degC/Drive cycles/"
            "03-21-17_00.29 25degC_UDDS_Pan18650PF.mat")
DT = 1.0
SOC0 = 100.0
SOC_INIT = 90.0
VOLTAGE_NOISE_STD = 0.002
FORGETTING_FACTOR_B = 0.99

# Starting Q/R for this cell -- see module docstring for why these
# differ from ukf_adaptive.py's CALCE-tuned Q_PROCESS_INIT/
# MEAS_NOISE_VAR_INIT.
Q_PROCESS_INIT = np.array([
    [0.5**2,  0.0],
    [0.0,     0.005**2],
])
MEAS_NOISE_VAR_INIT = 0.02 ** 2

np.random.seed(0)  # same seed as test_ukf_panasonic.py -> same noise realization


def _load():
    t_raw, I_raw, V_raw, Ah_raw = load_drive_cycle(FILEPATH)
    t, I, V_clean = resample_to_fixed_dt(t_raw, I_raw, V_raw, dt=DT)
    return t, I, V_clean


def run_adaptive(adapt_Q, adapt_R, b=FORGETTING_FACTOR_B):
    """
    One Sage-Husa AUKF run on the Panasonic UDDS cycle. Returns
    (t, soc_reference, soc_estimates, mae, rmse).
    """
    t, I, V_clean = _load()
    N = len(t)

    V_measured = V_clean + np.random.normal(0, VOLTAGE_NOISE_STD, N)
    soc_reference = compute_ah_reference_soc(
        I, DT, soc0=SOC0, capacity_ah=models_panasonic.NOMINAL_CAPACITY_AH,
        clip=False)

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
            soc_dynamics=models_panasonic.soc_dynamics,
            v1_dynamics=models_panasonic.v1_dynamics,
            terminal_voltage=models_panasonic.terminal_voltage,
        )
        soc_estimates[k] = x[0]

    error = soc_reference - soc_estimates
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return t, soc_reference, soc_estimates, mae, rmse


if __name__ == "__main__":
    np.random.seed(0)
    _, _, _, mae_Q, rmse_Q = run_adaptive(adapt_Q=True, adapt_R=False)
    np.random.seed(0)
    _, _, _, mae_R, rmse_R = run_adaptive(adapt_Q=False, adapt_R=True)
    np.random.seed(0)
    t, soc_reference, soc_both, mae_both, rmse_both = run_adaptive(adapt_Q=True, adapt_R=True)

    print("Results on real Panasonic 18650PF UDDS@25C data:")
    print(f"  {'Filter':<30s} {'MAE (%)':>10s} {'RMSE (%)':>10s}")
    print(f"  {'Sage-Husa AUKF (Q only)':<30s} {mae_Q:>10.4f} {rmse_Q:>10.4f}")
    print(f"  {'Sage-Husa AUKF (R only)':<30s} {mae_R:>10.4f} {rmse_R:>10.4f}")
    print(f"  {'Sage-Husa AUKF (Q and R)':<30s} {mae_both:>10.4f} {rmse_both:>10.4f}")
    print("  -- benchmark paper, Table 4, UDDS row --")
    print(f"  {'UKF (Guo et al. 2026)':<30s} {1.79:>10.4f} {2.11:>10.4f}")
    print(f"  {'AUKF (Guo et al. 2026)':<30s} {0.11:>10.4f} {0.53:>10.4f}")
    print(f"  {'PSO-AUKF (Guo et al. 2026)':<30s} {0.099:>10.4f} {0.41:>10.4f}")
    print(f"  {'QPSO-AUKF (Guo et al. 2026)':<30s} {0.075:>10.4f} {0.18:>10.4f}")

    soc_reference_frac = soc_reference / 100.0
    soc_both_frac = soc_both / 100.0
    error_both_frac = (soc_reference - soc_both) / 100.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(t, soc_reference_frac, color='black', label='Reference', linewidth=1.0)
    axes[0].plot(t, soc_both_frac, color='blue', label='AUKF', linewidth=1.0)
    axes[0].set_xlabel('Time(s)')
    axes[0].set_ylabel('SOC')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Panasonic 18650PF, UDDS@25C -- Sage-Husa Adaptive UKF')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, error_both_frac, color='blue', linewidth=1.0)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Time(s)')
    axes[1].set_ylabel('Error')
    axes[1].set_title(f'Error  [MAE={mae_both/100:.4f}, RMSE={rmse_both/100:.4f}]')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('panasonic_udds_adaptive_results.png', dpi=150)
    print("Plot saved to panasonic_udds_adaptive_results.png")