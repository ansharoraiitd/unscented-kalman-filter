"""
UKF predict-update loop for the 2-state case: x = [SoC, V1],
P = 2x2 covariance matrix, 5 sigma points.

Runs the standard predict/update cycle: generate sigma points, push
them through the process model to predict the next state, regenerate
sigma points around that prediction, push them through the
measurement model to predict voltage, then correct the state using
the mismatch between predicted and measured voltage (the Kalman
gain). A process-noise term Q is added to the predicted covariance
each step, reflecting that the process model isn't perfect between
measurements -- and a numerical safety net keeps the covariance
matrix well-behaved (symmetric, positive-definite) even when real
data has sharp current transitions that would otherwise push it
slightly invalid due to floating-point rounding.

Q_PROCESS and MEAS_NOISE_VAR are the noise values used on the CALCE
cell (test_ukf_real_data.py). ukf_step() also accepts Q_process and
meas_noise_var as optional arguments so a different cell's test
script can pass in its own noise assumptions, appropriate to that
cell's own model and drive cycle, without modifying this file --
test_ukf_panasonic.py does exactly this.
"""

import numpy as np
from sigma_points import generate_sigma_points
from models_real_data import soc_dynamics as _default_soc_dynamics
from models_real_data import v1_dynamics as _default_v1_dynamics
from models_real_data import terminal_voltage as _default_terminal_voltage

MEAS_NOISE_VAR = 0.002 ** 2  # assumed voltage sensor noise variance

# Assumed process noise -- how much the model's prediction step is
# trusted between measurements. The paper doesn't publish its own Q;
# these values were chosen so the filter responds meaningfully to
# real voltage measurements rather than being over-damped.
Q_PROCESS = np.array([
    [0.4**2,   0.0],
    [0.0,      0.005**2],
])


def regularize_covariance(P):
    """
    Numerical safety net for a covariance matrix -- not physics, just
    protection against floating point rounding errors that could make
    P slightly invalid (asymmetric, or a tiny negative "variance,"
    which Cholesky decomposition cannot handle).
    """
    P = (P + P.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.clip(eigvals, 1e-10, None)
    P = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return P


def ukf_step(x, P, current, dt, measured_voltage, lam=2.0,
             soc_dynamics=_default_soc_dynamics,
             v1_dynamics=_default_v1_dynamics,
             terminal_voltage=_default_terminal_voltage,
             Q_process=None, meas_noise_var=None):
    """
    x : np.array shape (2,) current state mean [SoC, V1]
    P : np.array shape (2,2) current state covariance matrix

    soc_dynamics, v1_dynamics, terminal_voltage : the three model
        functions, defaulting to the INR-18650-20R model (Table 2,
        Eq. 15). Passing in a different set of model functions lets
        this same filter loop run against a different cell's model
        without modifying this file.

    Q_process, meas_noise_var : override the module-level Q_PROCESS /
        MEAS_NOISE_VAR defaults, for a cell whose model and drive
        cycle call for different noise assumptions. Left as None, the
        defaults above are used unchanged.

    Returns updated (x, P).
    """
    Q = Q_PROCESS if Q_process is None else Q_process
    R = MEAS_NOISE_VAR if meas_noise_var is None else meas_noise_var

    # ============ PREDICT PHASE ============
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

    P_pred = (weights[0] * np.outer(d0, d0) + weights[1] * np.outer(d1, d1)
              + weights[2] * np.outer(d2, d2) + weights[3] * np.outer(d3, d3)
              + weights[4] * np.outer(d4, d4))

    P_pred = P_pred + Q
    P_pred = regularize_covariance(P_pred)

    # ============ UPDATE PHASE ============
    points2, weights2 = generate_sigma_points(x_pred, P_pred, lam=lam)

    v0 = terminal_voltage(points2[0][0], points2[0][1], current)
    v1 = terminal_voltage(points2[1][0], points2[1][1], current)
    v2 = terminal_voltage(points2[2][0], points2[2][1], current)
    v3 = terminal_voltage(points2[3][0], points2[3][1], current)
    v4 = terminal_voltage(points2[4][0], points2[4][1], current)

    v_mean = (weights2[0]*v0 + weights2[1]*v1 + weights2[2]*v2
              + weights2[3]*v3 + weights2[4]*v4)

    v_var = (weights2[0]*(v0-v_mean)**2 + weights2[1]*(v1-v_mean)**2
             + weights2[2]*(v2-v_mean)**2 + weights2[3]*(v3-v_mean)**2
             + weights2[4]*(v4-v_mean)**2) + R

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

    return x_new, P_new