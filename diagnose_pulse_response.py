"""
diagnose_pulse_response.py -- checks whether the filter's innovation
(voltage mismatch) is actually synced to the DST current pulses, the
way it should be for a correctly-responding RC model. If the paper's
oscillatory error comes from pulse-synced model mismatch (not random
noise), our innovation should visibly spike at every current
transition. If it doesn't, the filter isn't reacting to pulses at
all, which points at a real bug -- not a tuning issue.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_dst_segment, resample_to_fixed_dt
import models_real_data
from sigma_points import generate_sigma_points
import ukf_real_data

FILEPATH = "data/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls"
DT = 1.0
SOC0 = 80.0

t_raw, I_raw, V_raw = load_dst_segment(FILEPATH)
t, I, V_clean = resample_to_fixed_dt(t_raw, I_raw, V_raw, dt=DT)
N = len(t)

# run WITHOUT added noise, so any pulse-sync we see is purely model
# mismatch, not injected randomness
x = np.array([SOC0, 0.0])
P = np.array([[5.0**2, 0.0], [0.0, 0.01**2]])

innovations = np.zeros(N)
for k in range(1, N):
    points, weights = generate_sigma_points(x, P, lam=2.0)
    pred = np.array([
        [models_real_data.soc_dynamics(p[0], I[k], DT),
         models_real_data.v1_dynamics(p[1], p[0], I[k], DT)]
        for p in points
    ])
    x_pred = np.sum(weights[:, None] * pred, axis=0)

    v_pred = models_real_data.terminal_voltage(x_pred[0], x_pred[1], I[k])
    innovations[k] = V_clean[k] - v_pred

    x, P = ukf_real_data.ukf_step(
        x, P, current=I[k], dt=DT, measured_voltage=V_clean[k],
        soc_dynamics=models_real_data.soc_dynamics,
        v1_dynamics=models_real_data.v1_dynamics,
        terminal_voltage=models_real_data.terminal_voltage,
    )

# zoom into one representative window covering a few DST pulses
window = (t >= 3600) & (t <= 4200)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t[window], I[window], color='tab:green')
axes[0].set_ylabel('Current (A)')
axes[0].set_title('Current profile vs. innovation, zoomed to 3600-4200s')
axes[0].grid(alpha=0.3)

axes[1].plot(t[window], innovations[window] * 1000, color='tab:red')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_ylabel('Innovation (mV)')
axes[1].set_xlabel('Time/s')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pulse_response_check.png', dpi=150)
print("Saved pulse_response_check.png")
print(f"Innovation range in this window: {innovations[window].min()*1000:.2f} "
      f"to {innovations[window].max()*1000:.2f} mV")