import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CovarianceResult:
    """Result from a covariance estimation method."""
    covariance: np.ndarray
    method: str
    shrinkage: float


class LedoitWolfCovariance:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Shrinks the sample covariance matrix toward a structured estimator
    (constant-correlation model) to reduce estimation error.

    Reference: Ledoit & Wolf (2004),
    "A well-conditioned estimator for large-dimensional covariance matrices"
    """

    def __init__(self) -> None:
        self.shrinkage_: float = 0.0
        self.covariance_: Optional[np.ndarray] = None

    def fit(self, returns: np.ndarray) -> CovarianceResult:
        """
        Compute Ledoit-Wolf shrunk covariance matrix.

        Parameters
        ----------
        returns : np.ndarray
            Shape (n_periods, n_assets). Historical returns.

        Returns
        -------
        CovarianceResult
            Estimated covariance and shrinkage intensity.
        """
        n, p = returns.shape
        if n < 2 or p < 1:
            raise ValueError("Returns must have at least 2 rows and 1 column")

        sample_cov = np.cov(returns, rowvar=False)
        mean_returns = np.mean(returns, axis=0)

        var = np.diag(sample_cov).reshape(-1, 1)
        sqrt_var = np.sqrt(var)
        corr = sample_cov / (sqrt_var @ sqrt_var.T)
        np.fill_diagonal(corr, 1.0)

        avg_corr = self._average_correlation(corr, p)
        prior = avg_corr * (sqrt_var @ sqrt_var.T)
        np.fill_diagonal(prior, var.flatten())

        shrinkage = self._compute_shrinkage(returns, sample_cov, prior, n, p)

        shrunk = shrinkage * prior + (1 - shrinkage) * sample_cov

        self.shrinkage_ = shrinkage
        self.covariance_ = shrunk
        return CovarianceResult(covariance=shrunk, method="ledoit_wolf", shrinkage=shrinkage)

    @staticmethod
    def _average_correlation(corr: np.ndarray, p: int) -> float:
        """Compute average off-diagonal correlation."""
        if p <= 1:
            return 0.0
        upper = corr[np.triu_indices(p, k=1)]
        if len(upper) == 0:
            return 0.0
        return float(np.mean(upper))

    @staticmethod
    def _compute_shrinkage(
        returns: np.ndarray, sample_cov: np.ndarray, prior: np.ndarray, n: int, p: int
    ) -> float:
        """Compute optimal shrinkage intensity (phi/T)."""
        if n < 4 or p < 1:
            return 0.5

        demeaned = returns - np.mean(returns, axis=0)
        phi = 0.0
        for i in range(n):
            xi = demeaned[i:i+1].T @ demeaned[i:i+1]
            diff = xi - sample_cov
            phi += np.sum(diff ** 2)
        phi /= n

        gamma = np.sum((sample_cov - prior) ** 2)
        if gamma == 0:
            return 1.0

        kappa = (phi - gamma) / n if phi > gamma else 0.0
        shrinkage = max(0.0, min(1.0, kappa / gamma))
        return shrinkage


class OASCovariance:
    """
    Oracle Approximating Shrinkage (OAS) estimator.

    Similar to Ledoit-Wolf but with a different shrinkage target
    that has lower MSE for Gaussian data.

    Reference: Chen et al. (2010),
    "Shrinkage algorithms for MMSE covariance estimation"
    """

    def __init__(self) -> None:
        self.shrinkage_: float = 0.0
        self.covariance_: Optional[np.ndarray] = None

    def fit(self, returns: np.ndarray) -> CovarianceResult:
        """
        Compute OAS shrunk covariance matrix.

        Parameters
        ----------
        returns : np.ndarray
            Shape (n_periods, n_assets).

        Returns
        -------
        CovarianceResult
            Estimated covariance and shrinkage intensity.
        """
        n, p = returns.shape
        if n < 2 or p < 1:
            raise ValueError("Returns must have at least 2 rows and 1 column")

        sample_cov = np.cov(returns, rowvar=False)

        trace_sample = np.trace(sample_cov)
        if p > 1:
            rho = (trace_sample / p) * np.eye(p)
        else:
            rho = np.array([[trace_sample]])

        mu = np.trace(sample_cov @ sample_cov) / p

        rho_norm = np.sum(rho ** 2)
        sample_norm = np.sum(sample_cov ** 2)

        shrinkage = ((1 - 2 / p) * trace_sample ** 2 + (n + 1) * (mu - trace_sample ** 2 / p))
        denom = (n + 1) * (sample_norm - trace_sample ** 2 / p)

        if denom != 0:
            shrinkage = max(0.0, min(1.0, shrinkage / denom))
        else:
            shrinkage = 0.5

        shrunk = shrinkage * rho + (1 - shrinkage) * sample_cov

        self.shrinkage_ = shrinkage
        self.covariance_ = shrunk
        return CovarianceResult(covariance=shrunk, method="oas", shrinkage=shrinkage)


def auto_select_covariance(
    returns: np.ndarray,
    methods: Optional[List[str]] = None,
) -> CovarianceResult:
    """
    Automatically select the best covariance estimation method.

    Compares methods using a simple loss metric (Frobenius norm
    against a rolling-window validation set). Falls back to OAS.

    Parameters
    ----------
    returns : np.ndarray
        Shape (n_periods, n_assets).
    methods : list of str, optional
        Methods to try: 'sample', 'ledoit_wolf', 'oas'.

    Returns
    -------
    CovarianceResult
        Best estimated covariance.
    """
    if methods is None:
        methods = ["oas", "ledoit_wolf"]

    if returns.shape[0] < 20 or returns.shape[1] < 2:
        return CovarianceResult(
            covariance=np.cov(returns, rowvar=False) if returns.shape[0] > 1 else np.eye(returns.shape[1]),
            method="sample",
            shrinkage=0.0,
        )

    n = returns.shape[0]
    split = max(n // 3, 10)
    train = returns[:split]
    test = returns[split:]

    if test.shape[0] < 2:
        train = returns[:max(n // 2, 5)]
        test = returns[max(n // 2, 5):]

    best_result: Optional[CovarianceResult] = None
    best_error = float("inf")

    for method in methods:
        try:
            if method == "sample":
                cov = np.cov(train, rowvar=False)
                result = CovarianceResult(covariance=cov, method="sample", shrinkage=0.0)
            elif method == "ledoit_wolf":
                lw = LedoitWolfCovariance()
                result = lw.fit(train)
            elif method == "oas":
                oas = OASCovariance()
                result = oas.fit(train)
            else:
                continue

            test_cov = np.cov(test, rowvar=False)
            error = np.linalg.norm(result.covariance - test_cov, ord="fro")
            if error < best_error:
                best_error = error
                best_result = result
        except Exception:
            continue

    return best_result or CovarianceResult(
        covariance=np.cov(returns, rowvar=False) if returns.shape[0] > 1 else np.eye(returns.shape[1]),
        method="sample",
        shrinkage=0.0,
    )
