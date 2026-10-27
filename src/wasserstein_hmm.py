"""Wasserstein HMM regime inference (paper Sec. 5, Algorithm 1 / Sec. 7.1).

Implements, in order:

1. The closed-form 2-Wasserstein distance between multivariate Gaussians
   (Sec. 5.4, "Template-Based Regime Identity via Wasserstein Distance").
2. Predictive model-order selection (Sec. 5.2): for each candidate order K,
   fit a K-state Gaussian HMM on the expanding history H_t and score it by
   the one-step-ahead predictive log-likelihood on a recent validation
   slice, penalized by a complexity term.
3. Template tracking (Sec. 5.4-5.5): HMM components are mapped to a fixed
   set of G persistent templates by nearest 2-Wasserstein distance, their
   probabilities aggregated, and the templates updated by exponential
   smoothing.

`hmmlearn.hmm.GaussianHMM` (Context7: hmmlearn GaussianHMM API) is used for
the Gaussian-HMM fitting and the E-step (filtered posteriors); the
predictive-likelihood decomposition (Sec. 5.2) and the Wasserstein-distance
template bookkeeping (Sec. 5.4) are not provided by hmmlearn and are
implemented here directly from the paper's equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import sqrtm
from scipy.special import logsumexp

from src.config import (
    HMM_COMPLEXITY_LAMBDA,
    HMM_COV_REG,
    HMM_K_MAX,
    HMM_K_MIN,
    HMM_MIN_TRAIN_OBS,
    HMM_ORDER_SELECTION_FREQ,
    HMM_VALIDATION_SLICE,
    N_TEMPLATES,
    TEMPLATE_CALIBRATION_FRACTION,
    TEMPLATE_SMOOTHING_ETA,
)


# ---------------------------------------------------------------------------
# 1. Closed-form 2-Wasserstein distance between Gaussians (Sec. 5.4)
# ---------------------------------------------------------------------------
def wasserstein2_gaussian(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """2-Wasserstein distance between N(mu1, sigma1) and N(mu2, sigma2).

    Implements the closed-form expression (paper Sec. 5.4):

        W2^2 = ||mu1 - mu2||_2^2
               + Tr(sigma1 + sigma2 - 2 * (sigma2^{1/2} sigma1 sigma2^{1/2})^{1/2})

    using `scipy.linalg.sqrtm` for the matrix square roots, as suggested by
    the paper's implementation notes.
    """
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    sqrt_sigma2 = sqrtm(sigma2)
    sqrt_sigma2 = np.real_if_close(sqrt_sigma2, tol=1000)
    inner = sqrt_sigma2 @ sigma1 @ sqrt_sigma2
    inner = (inner + inner.T) / 2.0  # enforce symmetry against numerical drift
    sqrt_inner = sqrtm(inner)
    sqrt_inner = np.real_if_close(sqrt_inner, tol=1000)
    cov_term = float(np.trace(sigma1 + sigma2 - 2.0 * sqrt_inner))
    w2_sq = max(mean_term + cov_term, 0.0)
    return float(np.sqrt(w2_sq))


# ---------------------------------------------------------------------------
# 2. Predictive model-order selection (Sec. 5.2)
# ---------------------------------------------------------------------------
def _regularize_covars(model: GaussianHMM) -> None:
    """Add a small diagonal jitter to fitted full covariances for numerical
    stability (not specified in the paper; standard regularization for small
    expanding-window Gaussian HMM fits)."""
    covars = model.covars_
    n_features = covars.shape[-1]
    jitter = HMM_COV_REG * np.eye(n_features)
    model.covars_ = covars + jitter[None, :, :]


def fit_gaussian_hmm(
    X: np.ndarray,
    n_states: int,
    n_iter: int,
    tol: float,
    random_state: int = 0,
    init_model: Optional[GaussianHMM] = None,
) -> GaussianHMM:
    """Fit a full-covariance Gaussian HMM on X (Sec. 5.3).

    When `init_model` is given (same n_states) and already fitted, its
    parameters seed the new fit ("warm start"): since the expanding history
    H_t grows by one observation per day, continuing EM from yesterday's
    converged parameters needs only a handful of iterations to re-converge.
    This is a numerical-optimization choice (not a modeling parameter
    defined by the paper) used to keep the daily expanding-window refit of
    Algorithm 1 (line 16) computationally tractable.
    """
    if (
        init_model is not None
        and init_model.n_components == n_states
        and init_model.covariance_type == "full"
    ):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params="",
            params="stmc",
        )
        model.startprob_ = init_model.startprob_.copy()
        model.transmat_ = init_model.transmat_.copy()
        model.means_ = init_model.means_.copy()
        model.covars_ = init_model.covars_.copy()
    else:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params="stmc",
            params="stmc",
        )
    model.fit(X)
    _regularize_covars(model)
    return model


def forward_log_predictive(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Per-step one-step-ahead log predictive density log p(x_s | F_{s-1})
    under the fitted model (paper Sec. 5.2), via the scaled forward
    recursion. `log_c[s] = log p(x_s | x_0, ..., x_{s-1})` for s=0..T-1
    (with the convention `log_c[0] = log p(x_0)` under the model's start
    distribution). Custom implementation of the standard HMM forward-filter
    identity log P(x_0:T-1) = sum_s log c_s; hmmlearn does not expose the
    per-step predictive terms directly.
    """
    log_frameprob = model._compute_log_likelihood(X)  # (T, K) log B_s(k)
    T = log_frameprob.shape[0]
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))

    log_c = np.empty(T)
    log_alpha = log_startprob + log_frameprob[0]
    log_c[0] = logsumexp(log_alpha)
    log_alpha = log_alpha - log_c[0]
    for s in range(1, T):
        log_pred = logsumexp(log_alpha[:, None] + log_transmat, axis=0)
        log_alpha = log_pred + log_frameprob[s]
        log_c[s] = logsumexp(log_alpha)
        log_alpha = log_alpha - log_c[s]
    return log_c


def filtered_state_probs_last(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Filtered state-posterior vector p(z_T = k | x_0:T) at the last index T
    of X (paper Sec. 5.3: "filtered probabilities p_{t,k}"), obtained from
    the same forward recursion as `forward_log_predictive`. Because the
    window ends exactly at T (the current day), the filtered and smoothed
    posteriors coincide there, so this avoids a second forward-backward
    pass through `hmmlearn.predict_proba` for a quantity already produced
    by the strictly-causal forward filter.
    """
    log_frameprob = model._compute_log_likelihood(X)
    T = log_frameprob.shape[0]
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))

    log_alpha = log_startprob + log_frameprob[0]
    log_alpha = log_alpha - logsumexp(log_alpha)
    for s in range(1, T):
        log_pred = logsumexp(log_alpha[:, None] + log_transmat, axis=0)
        log_alpha = log_pred + log_frameprob[s]
        log_alpha = log_alpha - logsumexp(log_alpha)
    return np.exp(log_alpha)


def complexity_penalty(n_states: int, n_features: int) -> float:
    """Complexity(K): number of free parameters of a full-covariance
    Gaussian HMM with K states and D features (paper Sec. 5.2: "proportional
    to K or to the number of free parameters"; the free-parameter count is
    used here).

    Free parameters = (K-1) start probs + K*(K-1) transition probs
                       + K*D means + K*D*(D+1)/2 covariance entries.
    """
    n_start = n_states - 1
    n_trans = n_states * (n_states - 1)
    n_means = n_states * n_features
    n_covars = n_states * n_features * (n_features + 1) // 2
    return float(n_start + n_trans + n_means + n_covars)


def select_hmm_order(
    X_hist: np.ndarray,
    k_min: int = HMM_K_MIN,
    k_max: int = HMM_K_MAX,
    validation_len: int = HMM_VALIDATION_SLICE,
    complexity_lambda: float = HMM_COMPLEXITY_LAMBDA,
    n_iter: int = 40,
    tol: float = 1e-3,
    random_state: int = 0,
    warm_models: Optional[dict] = None,
    warm_iter: int = 10,
) -> Tuple[int, dict]:
    """Predictive model-order selection (Sec. 5.2, Algorithm 1 lines 7-14).

    For K in [k_min, k_max]: fit a K-state Gaussian HMM on the *entire*
    expanding history X_hist = X(H_t) (Algorithm 1, line 10), then compute
    the predictive log score over the last `validation_len` points of
    X_hist using the fitted model's own forward recursion (line 11). Select
    the K maximizing PredLL(K;t) - lambda_K * Complexity(K).

    `warm_models`, if given, is a ``{K: GaussianHMM}`` cache mutated in
    place: each candidate order is warm-started (see `fit_gaussian_hmm`)
    from its own previous converged fit whenever one is cached, needing
    only `warm_iter` EM iterations instead of a full `n_iter` cold restart.
    Because the expanding history H_t changes by a handful of rows between
    consecutive order-reselection days, this reaches essentially the same
    EM optimum as a cold restart while keeping Algorithm 1's periodic
    re-fit over the full expanding history computationally tractable; it is
    a numerical-optimization choice, not a modeling parameter of the paper.
    """
    n_features = X_hist.shape[1]
    scores = {}
    for k in range(k_min, k_max + 1):
        if X_hist.shape[0] < max(validation_len + 1, 5 * k):
            continue
        try:
            warm = warm_models.get(k) if warm_models is not None else None
            fit_iter = warm_iter if warm is not None else n_iter
            model = fit_gaussian_hmm(
                X_hist, k, n_iter=fit_iter, tol=tol, random_state=random_state, init_model=warm
            )
            log_c = forward_log_predictive(model, X_hist)
            pred_ll = float(np.sum(log_c[-validation_len:]))
        except Exception:
            continue
        if warm_models is not None:
            warm_models[k] = model
        penalty = complexity_lambda * complexity_penalty(k, n_features)
        scores[k] = pred_ll - penalty
    if not scores:
        raise RuntimeError("No candidate HMM order could be fit on the given history.")
    best_k = max(scores, key=scores.get)
    return best_k, scores


# ---------------------------------------------------------------------------
# 3. Template tracking (Sec. 5.4-5.5)
# ---------------------------------------------------------------------------
@dataclass
class Template:
    mean: np.ndarray
    cov: np.ndarray


@dataclass
class TemplateTracker:
    """Maintains G persistent templates Theta_g = (mu_g, Sigma_g) (Sec. 5.4)."""

    templates: List[Template]
    eta: float = TEMPLATE_SMOOTHING_ETA

    @classmethod
    def initialize(cls, X_calib: np.ndarray, n_templates: int = N_TEMPLATES, random_state: int = 0) -> "TemplateTracker":
        """Initialize templates on an initial calibration window (Algorithm
        1, line 3: "Initialize templates ... using an initial calibration
        window"). The paper does not specify the initialization procedure;
        a Gaussian Mixture Model with `n_templates` full-covariance
        components fit on the calibration window is used, giving each
        template an initial (mean, covariance) pair in the same feature
        space as the HMM components.
        """
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_templates,
            covariance_type="full",
            random_state=random_state,
            reg_covar=HMM_COV_REG,
            n_init=3,
        )
        gmm.fit(X_calib)
        templates = [Template(mean=gmm.means_[g].copy(), cov=gmm.covariances_[g].copy()) for g in range(n_templates)]
        return cls(templates=templates)

    def map_components(self, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
        """g(k) = argmin_g W2(Theta_g, (mu_{t,k}, Sigma_{t,k})) (Sec. 5.4)."""
        k_states = means.shape[0]
        mapping = np.empty(k_states, dtype=int)
        for k in range(k_states):
            dists = [
                wasserstein2_gaussian(t.mean, t.cov, means[k], covars[k]) for t in self.templates
            ]
            mapping[k] = int(np.argmin(dists))
        return mapping

    def aggregate(self, p_state: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        """p_{t,g} = sum_{k: g(k)=g} p_{t,k} (Sec. 5.4)."""
        p_template = np.zeros(len(self.templates))
        for k, g in enumerate(mapping):
            p_template[g] += p_state[k]
        return p_template

    def update(
        self,
        means: np.ndarray,
        covars: np.ndarray,
        p_state: np.ndarray,
        mapping: np.ndarray,
    ) -> None:
        """Exponential-smoothing template update (Sec. 5.4):

            mu_g   <- (1-eta) mu_g   + eta * mean of components mapped to g
            Sigma_g <- (1-eta) Sigma_g + eta * mean of covariances mapped to g

        where the "mean of components mapped to g" is the filtered-
        probability-weighted average component mapped to that template
        (bar-mu_{t,g}, bar-Sigma_{t,g} in the paper's notation), falling
        back to leaving the template unchanged when no component maps to
        it on a given day.
        """
        for g, template in enumerate(self.templates):
            idx = np.where(mapping == g)[0]
            if idx.size == 0:
                continue
            w = p_state[idx]
            w_sum = w.sum()
            if w_sum <= 1e-12:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w_sum
            mu_bar = np.sum(means[idx] * w[:, None], axis=0)
            sigma_bar = np.sum(covars[idx] * w[:, None, None], axis=0)
            template.mean = (1 - self.eta) * template.mean + self.eta * mu_bar
            template.cov = (1 - self.eta) * template.cov + self.eta * sigma_bar

    def mixture_moments(self, p_template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """mu_t = sum_g p_{t,g} mu_g, Sigma_t = sum_g p_{t,g} Sigma_g (Sec. 5.5),
        evaluated in the full feature space (returns, volatility, momentum
        jointly); the asset-return sub-block used for portfolio
        optimization is extracted by the caller (see
        `extract_return_submoments`)."""
        mu_t = sum(p_template[g] * self.templates[g].mean for g in range(len(self.templates)))
        sigma_t = sum(p_template[g] * self.templates[g].cov for g in range(len(self.templates)))
        return mu_t, sigma_t


def extract_return_submoments(
    mean_vec: np.ndarray, cov_mat: np.ndarray, n_assets: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the asset-return marginal (mu_r, Sigma_r) from a Gaussian
    defined jointly over x = [r; sigma; m] in R^{3N} (Sec. 5.1).

    The paper defines template/portfolio moments mu_t, Sigma_t (Sec. 5.5)
    directly as template parameters without stating explicitly that the
    mean-variance optimizer (Sec. 5.6) operates on an N-dimensional asset
    return vector while the HMM/template Gaussians live in R^{3N}. We take
    this as the marginal distribution of the return block r under the
    jointly Gaussian feature vector, i.e. the sub-vector of the mean and
    the corresponding principal sub-matrix of the covariance that
    correspond to the `r_<asset>` coordinates (the first coordinate of
    every asset's [r, sigma, m] triple, following the fixed column order
    used in `src/features.py:feature_columns`). This is an interpretation
    needed to connect Sec. 5.1's R^{3N} feature space to Sec. 5.6's
    N-dimensional MVO, recorded in results/RESULTS.md.
    """
    r_idx = np.arange(0, 3 * n_assets, 3)
    mu_r = mean_vec[r_idx]
    sigma_r = cov_mat[np.ix_(r_idx, r_idx)]
    return mu_r, sigma_r
