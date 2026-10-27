"""Non-parametric KNN conditional-moment estimator (paper Sec. 6, Algorithm 2).

At each out-of-sample day t, the K nearest neighbors of the current feature
vector x_t are retrieved from the expanding history of feature vectors
(`sklearn.neighbors.NearestNeighbors`, Context7: scikit-learn NearestNeighbors
API), and the local conditional mean/covariance of *raw asset returns* on
the neighbor days are estimated:

    mu_t    = (1/K) * sum_{s in N_t} r_s
    Sigma_t = LedoitWolfCov({r_s : s in N_t})

using `sklearn.covariance.LedoitWolf` for the shrinkage covariance
estimator, exactly as specified in Algorithm 2 (Sec. 7.2), steps 8-10.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


def knn_conditional_moments(
    x_t: np.ndarray,
    X_train: np.ndarray,
    returns_train: np.ndarray,
    n_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Local conditional moments from the K nearest feature-space neighbors
    of x_t (Algorithm 2, steps 7-10).

    Parameters
    ----------
    x_t : (D,) current feature vector (D = 3N), built from data up to t-1.
    X_train : (M, D) prior feature vectors (the expanding history X_{train}).
    returns_train : (M, N) raw asset log returns aligned 1:1 with the rows
        of X_train (r_s for each history index s).
    n_neighbors : K, the number of neighbors.
    """
    if X_train.shape[0] != returns_train.shape[0]:
        raise ValueError("X_train and returns_train must have the same number of rows.")
    k_eff = min(n_neighbors, X_train.shape[0])
    if k_eff < 2:
        raise ValueError("Need at least 2 neighbors to estimate a covariance matrix.")

    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(X_train)
    _, neighbor_idx = nn.kneighbors(x_t.reshape(1, -1))
    neighbor_idx = neighbor_idx.flatten()

    neighbor_returns = returns_train[neighbor_idx]  # (K, N)
    mu_t = neighbor_returns.mean(axis=0)

    lw = LedoitWolf()
    lw.fit(neighbor_returns)
    sigma_t = lw.covariance_

    return mu_t, sigma_t
