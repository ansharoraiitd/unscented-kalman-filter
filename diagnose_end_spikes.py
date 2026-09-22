"""
diagnose_end_spikes.py -- checks whether the sharp spikes at the end
of the error plot line up with specific current pulses, or occur
independent of current activity. If they're current-synced, they're
most likely the model's known OCV-steepness weakness at low SOC
being triggered by this specific file's ending pulses -- not a bug.
If they occur where current is flat/resting, that points elsewhere.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_dst_segment, resample_to_fixed_dt
from reference_soc import compute_ah_reference_soc, measured_capacity_ah
from ukf_real_data import ukf_step
import models_real_data

FILEPATH = "data/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls"
DT = 1.0
SOC0 = 80.0
SOC_INIT = 90.0
VOLTAGE_NOISE_STD = 0.004
np.random.seed(0)

t_raw, I_raw, V_raw = load_dst_segment(FILEPATH)
t, I, V_clean = resample_to_fixed_dt(t_raw, I_raw, V_raw, dt=DT)
N = len(t)

cap = measured_capacity_ah(I, DT, SOC0)
soc_ref = compute_ah_reference_soc(I, DT, soc0=SOC0, capacity_ah=cap, clip=False)
V_meas = V_clean + np.random.normal(0, VOLTAGE_NOISE_STD, N)

x = np.array([SOC_INIT, 0.0])
P = np.array([[0.05**2, 0.0], [0.0, 0.005**2]])
est = np.zeros(N)
est[0] = x[0]

for k in range(1, N):
    x, P = ukf_step(
        x, P, current=I[k], dt=DT, measured_voltage=V_meas[k],
        soc_dynamics=models_real_data.soc_dynamics,
        v1_dynamics=models_real_data.v1_dynamics,
        terminal_voltage=models_real_data.terminal_voltage,
    )
    est[k] = x[0]

err = soc_ref - est

window = t >= (t[-1] - 1000)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes[0].plot(t[window], I[window], color='tab:green')
axes[0].set_ylabel('Current (A)')
axes[0].grid(alpha=0.3)

axes[1].plot(t[window], soc_ref[window], label='AH')
axes[1].plot(t[window], est[window], label='UKF', alpha=0.85)
axes[1].set_ylabel('SOC (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(t[window], err[window], color='tab:red')
axes[2].axhline(0, color='black', linewidth=0.5)
axes[2].set_ylabel('Error (%)')
axes[2].set_xlabel('Time/s')
axes[2].grid(alpha=0.3)

plt.suptitle('Last 1000s: current, SOC estimates, and error')
plt.tight_layout()
plt.savefig('end_spike_diagnosis.png', dpi=150)
print("Saved end_spike_diagnosis.png")
print(f"Final AH SOC: {soc_ref[-1]:.2f}%,  Final UKF SOC: {est[-1]:.2f}%")