"""
Generates the sigma points and weights used by the Unscented Kalman
Filter. The state here is [SoC, V1] -- two variables -- so per the
unscented transform, n=2 gives 2n+1 = 5 sigma points: the mean itself,
plus two points placed symmetrically along each state dimension.
"""

import numpy as np

N = 2   # state dimension: [SoC, V1]


def generate_sigma_points(mu, P, lam=2.0):
    """
    mu : np.array shape (2,)   -- [mean_SoC, mean_V1]
    P  : np.array shape (2,2)  -- covariance matrix:
             [ Var(SoC)        Cov(SoC,V1) ]
             [ Cov(SoC,V1)     Var(V1)     ]
    lam: spacing parameter (default 2.0)

    Returns:
      points  : np.array shape (5, 2)  -- 5 rows, each a [soc, v1] pair
      weights : np.array shape (5,)
    """

    # Scale P by (N + lam). This controls how far apart the 5 points
    # spread out -- bigger P (more uncertain) or bigger lam -> points
    # spread further from the mean.
    scaled_P = (N + lam) * P

    # Cholesky decomposition of the 2x2 matrix: finds S such that
    # S @ S.T = scaled_P. For a 2x2 matrix, S has this triangular form:
    #     S = [ s11   0  ]
    #         [ s21  s22 ]
    # with s11 = sqrt(scaled_P[0,0]), s21 = scaled_P[1,0]/s11,
    # s22 = sqrt(scaled_P[1,1] - s21**2).
    S = np.linalg.cholesky(scaled_P)

    # S[:, 0] = first column = [s11, s21]  -> shift direction 1
    # S[:, 1] = second column = [0, s22]   -> shift direction 2
    shift1 = S[:, 0]
    shift2 = S[:, 1]

    # Build all 5 points: the mean, and two symmetric pairs.
    chi0 = mu                  # the mean itself
    chi1 = mu + shift1         # shifted along direction 1, positive
    chi2 = mu + shift2         # shifted along direction 2, positive
    chi3 = mu - shift1         # mirror of chi1
    chi4 = mu - shift2         # mirror of chi2

    points = np.array([chi0, chi1, chi2, chi3, chi4])

    # Weights: the center point gets one weight, all 4 side points
    # share another.
    w0 = lam / (N + lam)
    w_side = 1.0 / (2 * (N + lam))

    weights = np.array([w0, w_side, w_side, w_side, w_side])

    return points, weights