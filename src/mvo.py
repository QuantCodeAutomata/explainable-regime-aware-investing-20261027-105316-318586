"""Transaction-cost-aware mean-variance optimization (paper Sec. 5.6 / Sec. 6).

Both the parametric (Wasserstein HMM) and non-parametric (KNN) strategies
solve the same daily portfolio problem given conditional moments (mu_t,
Sigma_t) and the previous weights w_{t-1}:

    max_w  mu_t' w - gamma * w' Sigma_t w - tau * ||w - w_{t-1}||_1
    s.t.   sum(w) = 1, w >= 0, ||w||_inf <= w_max

Solved with `cvxpy` (Context7: cvxpy DCP/DPP atoms `cp.quad_form`,
`cp.norm1`), which is a convex quadratic program with an L1 penalty — a
disciplined convex program handled directly by `cvxpy`, so no part of this
optimizer is implemented from scratch.
"""

from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np

from src.config import MVO_GAMMA, MVO_TAU, MVO_W_MAX


def solve_mvo(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: Optional[np.ndarray],
    gamma: float = MVO_GAMMA,
    tau: float = MVO_TAU,
    w_max: float = MVO_W_MAX,
    solver: str = cp.SCS,
) -> np.ndarray:
    """Solve the transaction-cost-aware MVO problem (Sec. 5.6) for one day.

    Parameters
    ----------
    mu : (N,) expected asset log-return vector for the day.
    sigma : (N, N) covariance matrix (must be positive semi-definite).
    w_prev : (N,) previous day's weights; defaults to equal weight when
        None (Algorithm 1, line 4: "Initialize w_{t0} <- equal-weight").
    gamma, tau, w_max : risk aversion, L1 transaction-cost penalty, and
        per-asset weight cap (Sec. 5.6).
    """
    n = mu.shape[0]
    if sigma.shape != (n, n):
        raise ValueError(f"sigma shape {sigma.shape} incompatible with mu shape {mu.shape}")
    if w_prev is None:
        w_prev = np.full(n, 1.0 / n)

    # Project sigma onto the PSD cone (small numerical negative eigenvalues
    # can appear after mixing template covariances; not part of the paper's
    # formula, a numerical safeguard for the quadratic-form constraint).
    eigvals, eigvecs = np.linalg.eigh((sigma + sigma.T) / 2.0)
    eigvals_clipped = np.clip(eigvals, 1e-10, None)
    sigma_psd = (eigvecs * eigvals_clipped) @ eigvecs.T

    w = cp.Variable(n)
    objective = cp.Maximize(
        mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(sigma_psd)) - tau * cp.norm1(w - w_prev)
    )
    constraints = [cp.sum(w) == 1, w >= 0, w <= w_max]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    if w.value is None:
        # Fall back to a different solver before giving up (documented
        # numerical fallback, not a methodology change).
        problem.solve(solver=cp.SCS)
    if w.value is None:
        raise RuntimeError(f"MVO problem failed to solve (status={problem.status}).")

    w_sol = np.clip(np.asarray(w.value).flatten(), 0.0, w_max)
    total = w_sol.sum()
    if total <= 0:
        raise RuntimeError("MVO solution has non-positive total weight.")
    return w_sol / total


def turnover(w_t: np.ndarray, w_prev: np.ndarray) -> float:
    """Daily turnover TO_t = 0.5 * ||w_t - w_{t-1}||_1 (paper diagnostics, Sec. 7.1)."""
    return 0.5 * float(np.sum(np.abs(w_t - w_prev)))


def effective_n(w_t: np.ndarray) -> float:
    """Portfolio concentration N_eff,t = (sum w_i^2)^-1 (paper diagnostics, Sec. 7.1)."""
    ssq = float(np.sum(w_t ** 2))
    if ssq <= 0:
        return float("nan")
    return 1.0 / ssq
