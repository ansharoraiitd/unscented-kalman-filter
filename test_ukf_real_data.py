"""
Runs the UKF on real CALCE INR-18650-20R DST data, using the paper's
own published model (models_real_data.py -- Table 2 + Eq. 15, no
fitting or identification of our own), and evaluates it against an
Ampere-Hour-counting reference SOC using MAE/RMSE, exactly as in
Zeng, Y., Li, Y., Yang, T. (2023), Eqs. (16)-(17).

Since every model parameter here comes directly from the paper rather
than from any fitting step, this is a clean check of the filter
implementation itself against the paper's own reported result
(Table 3, DST/25C: MAE=0.93%, RMSE=1.22%).

Gaussian measurement noise is added to the voltage before it reaches
the filter, matching the paper's own stated methodology of adding
Gaussian white noise to measurements. The filter is also deliberately
initialized away from the true starting SOC (90% vs the actual 80%),
mirroring the initial convergence transient visible in the paper's
own Figure 6d.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_dst_segment, resample_to_fixed_dt
from reference_soc import compute_ah_reference_soc, measured_capacity_ah
from ukf_real_data import ukf_step
import models_real_data

FILEPATH = "data/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls"
RATED_CAPACITY_AH = 2.0     # INR-18650-20R, Table 1 of the Zeng paper
DT = 1.0

# Step_Index==5 (the controlled discharge right before the DST
# profile starts) removes exactly 20.0% of the 2.0Ah rated capacity
# in this file -- that's where "80SOC" in the filename comes from,
# and it sets the true starting point below.
SOC0 = 80.0
SOC_INIT = 90.0  # deliberately wrong, to reproduce the initial convergence transient

VOLTAGE_NOISE_STD = 0.002   # volts; the paper doesn't publish its own value
np.random.seed(0)          # reproducible

t_raw, I_raw, V_raw = load_dst_segment(FILEPATH)
t, I, V_clean = resample_to_fixed_dt(t_raw, I_raw, V_raw, dt=DT)
N = len(t)

# Capacity calibrated from what this run actually delivered
cap_actual = measured_capacity_ah(I, DT, SOC0)

# Simulated noisy voltage measurement, matching the paper's methodology
V_measured = V_clean + np.random.normal(0, VOLTAGE_NOISE_STD, N)

# Ampere-Hour counting reference SOC (this is the "true" y_i)
soc_reference = compute_ah_reference_soc(
    I, DT, soc0=SOC0, capacity_ah=cap_actual, clip=False)

# Run the filter over the real data
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

# MAE / RMSE, exactly Eqs. (16)-(17) of the paper
error = soc_reference - soc_estimates
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error ** 2))

print(f"Results on real DST data (INR-18650-20R cell, paper's own parameters):")
print(f"  MAE  = {mae:.4f} %")
print(f"  RMSE = {rmse:.4f} %")
print(f"(Paper's own reported UKF result, DST/25C, Table 3: MAE=0.93%, RMSE=1.22%)")

# Plot, in fractional SOC (0-1), matching the paper's own axis convention
soc_reference_frac = soc_reference / 100.0
soc_estimates_frac = soc_estimates / 100.0
error_frac = error / 100.0

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].plot(t, soc_reference_frac, label='AH', linewidth=1.0)
axes[0].plot(t, soc_estimates_frac, label='UKF', linewidth=1.0, alpha=0.85)
axes[0].set_xlabel('Time/s')
axes[0].set_ylabel('SOC')
axes[0].set_ylim(-0.1, 0.9)
axes[0].set_title('DST@25C')
axes[0].legend()
axes[0].grid(alpha=0.3)

# inset zoom, mirroring the paper's Figure 6c
inset = axes[0].inset_axes([0.12, 0.12, 0.42, 0.30])
mask = (t >= 3650) & (t <= 3820)
inset.plot(t[mask], soc_reference_frac[mask], linewidth=1.0)
inset.plot(t[mask], soc_estimates_frac[mask], linewidth=1.0, alpha=0.85)
inset.grid(alpha=0.3)
inset.tick_params(labelsize=7)

axes[1].plot(t, error_frac, linewidth=0.5)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_xlabel('Time/s')
axes[1].set_ylabel('Error')
axes[1].set_ylim(-0.2, 0.25)
axes[1].set_title(f'Error  [MAE={mae/100:.4f}, RMSE={rmse/100:.4f}]')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('real_data_results.png', dpi=150)
print("Plot saved to real_data_results.png")