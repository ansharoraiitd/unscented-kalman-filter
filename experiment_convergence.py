"""
Synthetic convergence test for the 2-state UKF.

Starts the filter deliberately wrong (SoC=80%) against a known true
state (SoC=90%, resting), and tracks how many iterations it takes to
converge below a 1% error threshold. This is a controlled sanity
check of the filter's basic behavior, independent of any real data.
"""

import numpy as np
import matplotlib.pyplot as plt
from ukf import ukf_step
from models import terminal_voltage

np.random.seed(42)

TRUE_SOC = 90.0
TRUE_V1 = 0.0
INIT_X = np.array([80.0, 0.0])
INIT_P = np.array([
    [5.0**2, 0.0],
    [0.0,    0.1**2]
])   # start uncorrelated; the filter can learn correlation over time
CURRENT = 0.0
DT = 1.0
SENSOR_NOISE_STD = 0.002
MAX_ITERATIONS = 200
ERROR_THRESHOLD = 1.0


def run_convergence_experiment(verbose=True):
    x, P = INIT_X.copy(), INIT_P.copy()
    errors = []
    converged_at = None

    for i in range(1, MAX_ITERATIONS + 1):
        true_v = terminal_voltage(TRUE_SOC, TRUE_V1, CURRENT)
        noisy_measurement = true_v + np.random.normal(0, SENSOR_NOISE_STD)

        x, P = ukf_step(x, P, CURRENT, DT, noisy_measurement)

        error = abs(x[0] - TRUE_SOC)
        errors.append(error)

        if converged_at is None and error < ERROR_THRESHOLD:
            converged_at = i

        if verbose and i <= 10:
            print(f"Iter {i:3d}: SoC={x[0]:6.3f}  V1={x[1]:7.4f}  "
                  f"sigma_SoC={np.sqrt(P[0,0]):.4f}  correlation={P[0,1]:.6f}  "
                  f"error={error:.3f}%")

    return errors, converged_at


if __name__ == "__main__":
    errors, converged_at = run_convergence_experiment()

    print()
    if converged_at:
        print(f"Converged to <1% error after {converged_at} iterations.")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', markersize=3, linewidth=1.2)
    plt.axhline(y=ERROR_THRESHOLD, color='r', linestyle='--', label='1% error threshold')
    if converged_at:
        plt.axvline(x=converged_at, color='g', linestyle='--', label=f'Converged at iter {converged_at}')
    plt.xlabel('Iteration')
    plt.ylabel('SoC error (%)')
    plt.title('2-state UKF Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=150)
    print("Plot saved to convergence_plot.png")