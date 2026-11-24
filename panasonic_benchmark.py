"""
Final benchmarking script for the BTP.

Benchmark problem: Guo, H., Li, Z., Xue, H. & Luo, J. (2026), "State
of Charge Estimation Method for Lithium-Ion Batteries Based on Online
Parameter Identification and QPSO-AUKF," Batteries 12(3), 84,
DOI 10.3390/batteries12030084. Section 4.4, "Validation on an
Independent Wisconsin Dataset," benchmarks four algorithms -- a plain
UKF, an AUKF, a PSO-AUKF and their own QPSO-AUKF -- on the publicly
available Panasonic 18650PF Li-ion Battery Data (P. Kollmeyer,
Mendeley Data, DOI 10.17632/wykht8y7tg.1), reporting MAE and RMSE for
three drive cycles (HWFET, LA92, UDDS) at 25 degC in their Table 4.
This project's own implementations of the normal UKF
(ukf_real_data.py) and the Sage-Husa Adaptive UKF (ukf_adaptive.py)
are run on the same dataset, same cycle (UDDS) and same temperature,
and compared directly against that paper's own reported numbers.

This script does not re-implement the filter loops; it imports and
reuses run_ukf() from test_ukf_panasonic.py and run_adaptive() from
test_ukf_adaptive_panasonic.py, which already load the data, run the
filter, and return the results. This file's only job is to run both,
build the final side-by-side comparison table, and produce one
combined figure overlaying Reference / UKF / Adaptive UKF, matched to
the benchmark paper's own Figure 12 (UDDS condition) layout, axis
scale and color convention: magenta for UKF, blue for AUKF, black for
the reference -- the same colors and roles the paper itself uses for
those two series in Figure 12, so the two figures sit side by side
for a direct visual comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from test_ukf_panasonic import run_ukf
from test_ukf_adaptive_panasonic import run_adaptive

# Benchmark paper's own Table 4, UDDS row -- see module docstring for citation
PAPER_RESULTS = {
    "UKF":        (1.79, 2.11),
    "AUKF":       (0.11, 0.53),
    "PSO-AUKF":   (0.099, 0.41),
    "QPSO-AUKF":  (0.075, 0.18),
}

UKF_COLOR = "magenta"
AUKF_COLOR = "blue"
REF_COLOR = "black"

np.random.seed(0)
t, soc_ref, soc_ukf, mae_ukf, rmse_ukf = run_ukf()

np.random.seed(0)  # identical noise realization for a fair side-by-side comparison
_, _, soc_aukf, mae_aukf, rmse_aukf = run_adaptive(adapt_Q=True, adapt_R=True)

print("=" * 72)
print("FINAL BENCHMARK COMPARISON -- Panasonic 18650PF, UDDS@25C")
print("Benchmark source: Guo, Li, Xue & Luo (2026), Batteries 12(3), 84, Table 4")
print("=" * 72)
print(f"  {'Method':<30s} {'MAE (%)':>10s} {'RMSE (%)':>10s}")
print(f"  {'This project -- UKF':<30s} {mae_ukf:>10.4f} {rmse_ukf:>10.4f}")
print(f"  {'This project -- Sage-Husa AUKF':<30s} {mae_aukf:>10.4f} {rmse_aukf:>10.4f}")
print("  " + "-" * 52)
for name, (mae, rmse) in PAPER_RESULTS.items():
    print(f"  {'Paper -- ' + name:<30s} {mae:>10.4f} {rmse:>10.4f}")
print("=" * 72)
print(f"  Adaptive/fixed MAE ratio:  {mae_aukf/mae_ukf:.3f}")
print(f"  Adaptive/fixed RMSE ratio: {rmse_aukf/rmse_ukf:.3f}")

with open("panasonic_benchmark_comparison.txt", "w") as f:
    f.write("Final benchmark comparison -- Panasonic 18650PF, UDDS@25C\n")
    f.write("Benchmark source: Guo, Li, Xue & Luo (2026), Batteries 12(3), 84, Table 4\n\n")
    f.write(f"{'Method':<30s}{'MAE (%)':>12s}{'RMSE (%)':>12s}\n")
    f.write(f"{'This project -- UKF':<30s}{mae_ukf:>12.4f}{rmse_ukf:>12.4f}\n")
    f.write(f"{'This project -- Sage-Husa AUKF':<30s}{mae_aukf:>12.4f}{rmse_aukf:>12.4f}\n")
    f.write("-" * 54 + "\n")
    for name, (mae, rmse) in PAPER_RESULTS.items():
        f.write(f"{'Paper -- ' + name:<30s}{mae:>12.4f}{rmse:>12.4f}\n")
    f.write("\n")
    f.write(f"Adaptive/fixed MAE ratio:  {mae_aukf/mae_ukf:.3f}\n")
    f.write(f"Adaptive/fixed RMSE ratio: {rmse_aukf/rmse_ukf:.3f}\n")
print("Comparison table saved to panasonic_benchmark_comparison.txt")

# ---------------- combined figure, matched to the paper's Figure 12 ----------------
soc_ref_frac = soc_ref / 100.0
soc_ukf_frac = soc_ukf / 100.0
soc_aukf_frac = soc_aukf / 100.0
err_ukf_frac = (soc_ref - soc_ukf) / 100.0
err_aukf_frac = (soc_ref - soc_aukf) / 100.0

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(t, soc_ref_frac, color=REF_COLOR, linewidth=1.0, label='Reference')
ax.plot(t, soc_ukf_frac, color=UKF_COLOR, linewidth=1.0, label='UKF')
ax.plot(t, soc_aukf_frac, color=AUKF_COLOR, linewidth=1.0, label='AUKF')
ax.set_xlabel('Time(s)')
ax.set_ylabel('SOC')
ax.set_xlim(0, 25000)
ax.set_ylim(0, 1)
ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.legend(fontsize=8, loc='upper right')
ax.set_title('(a) SOC Estimation Curve under UDDS', fontsize=10)

# inset zoom, same role as the benchmark paper's own inset in Figure 12(a):
# a close-up on the early convergence transient
inset = ax.inset_axes([0.12, 0.12, 0.32, 0.30])
mask = (t >= 700) & (t <= 1000)
inset.plot(t[mask], soc_ref_frac[mask], color=REF_COLOR, linewidth=1.0)
inset.plot(t[mask], soc_ukf_frac[mask], color=UKF_COLOR, linewidth=1.0)
inset.plot(t[mask], soc_aukf_frac[mask], color=AUKF_COLOR, linewidth=1.0)
inset.set_xticks([700, 800, 900, 1000])
inset.tick_params(labelsize=6)
inset.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(t, err_ukf_frac, color=UKF_COLOR, linewidth=1.0, label='UKF')
ax2.plot(t, err_aukf_frac, color=AUKF_COLOR, linewidth=1.0, label='AUKF')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel('Cycle')
ax2.set_ylabel('Error')
ax2.set_xlim(0, 25000)
ax2.set_ylim(-0.05, 0.2)
ax2.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax2.set_yticks([-0.05, 0, 0.05, 0.1, 0.15, 0.2])
ax2.legend(fontsize=8, loc='upper right')
ax2.set_title('(b) SOC Estimation Error Curve under UDDS', fontsize=10)

plt.tight_layout()
plt.savefig('panasonic_final_comparison.png', dpi=150)
print("Combined comparison plot saved to panasonic_final_comparison.png")