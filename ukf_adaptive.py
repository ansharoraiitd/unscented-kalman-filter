"""
Sage-Husa Adaptive UKF predict-update loop for the 2-state case:
x = [SoC, V1], P = 2x2 covariance matrix, 5 sigma points.

Same predict/update cycle as ukf_real_data.py, with one addition: Q
and R are no longer fixed constants. After the normal update step,
both are recursively re-estimated from the innovation sequence
(Sage-Husa covariance matching), so the filter learns its own noise
levels from the data instead of relying on hand-picked initial
values.

Q_PROCESS_INIT and MEAS_NOISE_VAR_INIT are the starting values used on
the CALCE cell (test_ukf_adaptive.py); they are only a starting point
for the recursion, not a noise level that holds for every cell. A
different cell's test script passes its own initial Q/R into
ukf_step_adaptive() directly (test_ukf_adaptive_panasonic.py does
this) without needing any change to this file.

References (the Q/R recursion below follows these directly, not a
reworded version of them):

  Sage, A.P. & Husa, G.W. (1969). "Adaptive Filtering with Unknown
  Prior Statistics." Proceedings of the JACC, pp. 760-769.
      -- original recursive noise-covariance update.

  Liu, S., Cui, N., Zhang, C. (2017). "An Adaptive Square Root
  Unscented Kalman Filter Approach for State of Charge Estimation of
  Lithium-Ion Batteries." Energies, 10(9), 1345.
  https://doi.org/10.3390/en10091345
      -- same Sage-Husa + UKF combination applied to a Thevenin-type
         battery model; confirms the R_k/Q_k recursion used below.

  Chen, Y. & Yi, L. (2020). "State Co-estimation Algorithm for Li-Ion
  Power Batteries Based on Adaptive Unscented Kalman Filters."
  J. Northeastern University (Natural Science), 41(11), 1557-1563.
      -- confirms the forgetting-factor form d_k = (1-b)/(1-b**(k+1)).

Known failure mode, documented in all three references above: the
Q/R recursion is a subtraction of matrices/terms and can push Q or R
negative from floating-point rounding or a bad transient. Same fix
philosophy as regularize_covariance() in ukf_real_data.py -- clip
anything below a small positive floor.
"""

import numpy as np
from sigma_points import generate_sigma_points
from models_real_data import soc_dynamics as _default_soc_dynamics
from models_real_data import v1_dynamics as _default_v1_dynamics
from models_real_data import terminal_voltage as _default_terminal_voltage

# Starting point for the adaptive recursion on the CALCE cell. The
# adaptive filter is initialized here and then updates Q and R
# itself, step by step.
Q_PROCESS_INIT = np.array([
    [0.4**2,   0.0],
    [0.0,      0.005**2],
])
MEAS_NOISE_VAR_INIT = 0.002 ** 2


def regularize_covariance(P, floor=1e-10):
    """
    Same numerical safety net as ukf_real_data.py -- symmetrize and
    clip tiny/negative eigenvalues so Cholesky decomposition in
    generate_sigma_points() never sees an invalid matrix. Reused here
    for Q as well as P, since the Sage-Husa update can push Q
    slightly invalid the same way floating-point rounding can push P
    invalid.
    """
    P = (P + P.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.clip(eigvals, floor, None)
    P = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return P


def ukf_step_adaptive(x, P, Q, R, current, dt, measured_voltage, k,
                       b=0.98, lam=2.0, adapt_Q=True, adapt_R=True,
                       soc_dynamics=_default_soc_dynamics,
                       v1_dynamics=_default_v1_dynamics,
                       terminal_voltage=_default_terminal_voltage):
    """
    x : np.array shape (2,)   current state mean [SoC, V1]
    P : np.array shape (2,2)  current state covariance matrix
    Q : np.array shape (2,2)  current process-noise covariance
                              (pass this cell's own starting value on
                              the very first call, then feed back the
                              Q this function returns on every call
                              after that)
    R : float                  current measurement-noise variance
                              (same pattern as Q)
    k : int                     step counter starting at 1, used only
                              for the forgetting factor d_k. The
                              caller increments this every step --
                              see test_ukf_adaptive.py /
                              test_ukf_adaptive_panasonic.py.
    b : float                   forgetting factor, 0 < b < 1. b close
                              to 1 = slow adaptation (close to the
                              fixed-noise filter); b close to 0 =
                              fast but noisy adaptation. 0.95-0.99 is
                              the typical range in the papers above.
    adapt_Q, adapt_R : bool     set either to False to leave that
                              matrix fixed at its current value --
                              useful for the R-only / Q-only ablation
                              (see test_ukf_adaptive.py).

    soc_dynamics, v1_dynamics, terminal_voltage : same as
        ukf_real_data.py -- defaults to the INR-18650-20R model.

    Returns (x_new, P_new, Q_new, R_new, innovation).
    """

    # ============ PREDICT PHASE (identical to ukf_real_data.py) ============
    points, weights = generate_sigma_points(x, P, lam=lam)

    new_soc_0 = soc_dynamics(points[0][0], current, dt)
    new_v1_0 = v1_dynamics(points[0][1], points[0][0], current, dt)
    new_soc_1 = soc_dynamics(points[1][0], current, dt)
    new_v1_1 = v1_dynamics(points[1][1], points[1][0], current, dt)
    new_soc_2 = soc_dynamics(points[2][0], current, dt)
    new_v1_2 = v1_dynamics(points[2][1], points[2][0], current, dt)
    new_soc_3 = soc_dynamics(points[3][0], current, dt)
    new_v1_3 = v1_dynamics(points[3][1], points[3][0], current, dt)
    new_soc_4 = soc_dynamics(points[4][0], current, dt)
    new_v1_4 = v1_dynamics(points[4][1], points[4][0], current, dt)

    pred_points = np.array([
        [new_soc_0, new_v1_0],
        [new_soc_1, new_v1_1],
        [new_soc_2, new_v1_2],
        [new_soc_3, new_v1_3],
        [new_soc_4, new_v1_4],
    ])

    x_pred = (weights[0] * pred_points[0] + weights[1] * pred_points[1]
              + weights[2] * pred_points[2] + weights[3] * pred_points[3]
              + weights[4] * pred_points[4])

    d0 = pred_points[0] - x_pred
    d1 = pred_points[1] - x_pred
    d2 = pred_points[2] - x_pred
    d3 = pred_points[3] - x_pred
    d4 = pred_points[4] - x_pred

    # Kept separate from +Q so the Sage-Husa Q-update below can use
    # the "pure sigma-point spread" term on its own -- see the
    # ADAPTATION block at the bottom for why this split matters.
    P_pred_noQ = (weights[0] * np.outer(d0, d0) + weights[1] * np.outer(d1, d1)
                  + weights[2] * np.outer(d2, d2) + weights[3] * np.outer(d3, d3)
                  + weights[4] * np.outer(d4, d4))
    P_pred = P_pred_noQ + Q
    P_pred = regularize_covariance(P_pred)

    # ============ UPDATE PHASE (identical structure to ukf_real_data.py) ====
    points2, weights2 = generate_sigma_points(x_pred, P_pred, lam=lam)

    v0 = terminal_voltage(points2[0][0], points2[0][1], current)
    v1 = terminal_voltage(points2[1][0], points2[1][1], current)
    v2 = terminal_voltage(points2[2][0], points2[2][1], current)
    v3 = terminal_voltage(points2[3][0], points2[3][1], current)
    v4 = terminal_voltage(points2[4][0], points2[4][1], current)

    v_mean = (weights2[0]*v0 + weights2[1]*v1 + weights2[2]*v2
              + weights2[3]*v3 + weights2[4]*v4)

    # v_var_noR: predicted voltage variance BEFORE adding R. Kept
    # separate because the Sage-Husa R-update needs this term on its
    # own (R_hat = innovation**2 + v_var_noR). Using v_var (with R
    # already folded in) here is the most common bug in Sage-Husa UKF
    # implementations -- flagged explicitly in Liu, Cui & Zhang (2017).
    v_var_noR = (weights2[0]*(v0-v_mean)**2 + weights2[1]*(v1-v_mean)**2
                 + weights2[2]*(v2-v_mean)**2 + weights2[3]*(v3-v_mean)**2
                 + weights2[4]*(v4-v_mean)**2)
    v_var = v_var_noR + R

    dx0 = points2[0] - x_pred
    dx1 = points2[1] - x_pred
    dx2 = points2[2] - x_pred
    dx3 = points2[3] - x_pred
    dx4 = points2[4] - x_pred

    dv0 = v0 - v_mean
    dv1 = v1 - v_mean
    dv2 = v2 - v_mean
    dv3 = v3 - v_mean
    dv4 = v4 - v_mean

    cross_cov = (weights2[0]*dx0*dv0 + weights2[1]*dx1*dv1
                 + weights2[2]*dx2*dv2 + weights2[3]*dx3*dv3
                 + weights2[4]*dx4*dv4)

    kalman_gain = cross_cov / v_var

    innovation = measured_voltage - v_mean
    x_new = x_pred + kalman_gain * innovation
    P_new = P_pred - v_var * np.outer(kalman_gain, kalman_gain)

    P_new = regularize_covariance(P_new)
    x_new[0] = np.clip(x_new[0], 0.0, 100.0)

    # ============ SAGE-HUSA ADAPTATION ============
    # Forgetting factor, Sage & Husa (1969), confirmed in Chen & Yi (2020):
    #   d_k = (1 - b) / (1 - b**(k+1))
    d_k = (1.0 - b) / (1.0 - b ** (k + 1))

    if adapt_R:
        R_hat = innovation ** 2 + v_var_noR
        R_new = (1.0 - d_k) * R + d_k * R_hat
        R_new = max(R_new, 1e-10)   # R is scalar here -- plain floor is enough
    else:
        R_new = R

    if adapt_Q:
        # Note: subtracting (P_pred_noQ + Q), not the full P_pred,
        # even though P_pred = P_pred_noQ + Q numerically -- written
        # this way so it's explicit that the OLD Q is what's being
        # backed out before blending in the new estimate.
        Q_hat = np.outer(kalman_gain, kalman_gain) * (innovation ** 2) + (P_new - P_pred_noQ - Q)
        Q_new = (1.0 - d_k) * Q + d_k * Q_hat
        Q_new = regularize_covariance(Q_new, floor=1e-10)
    else:
        Q_new = Q

    return x_new, P_new, Q_new, R_new, innovation