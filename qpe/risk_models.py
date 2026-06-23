import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class TrackingErrorConstraint:
    """
    Constrain portfolio tracking error relative to a benchmark.

    TE = sqrt((w - w_b)' Σ (w - w_b))

    Enforces TE <= max_te by projecting weights back into
    the feasible set.
    """

    def __init__(self, max_te: float = 0.08) -> None:
        """
        Parameters
        ----------
        max_te : float, default=0.08
            Maximum annualized tracking error (8%).
        """
        self.max_te = max_te

    def project(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        covariance: np.ndarray,
        max_iter: int = 50,
        step: float = 0.5,
    ) -> np.ndarray:
        """
        Project weights to satisfy tracking error constraint.

        Uses iterative scaling toward benchmark.

        Parameters
        ----------
        weights : np.ndarray
            Candidate portfolio weights.
        benchmark_weights : np.ndarray
            Benchmark weights (e.g., equal weight or cap weight).
        covariance : np.ndarray
            Covariance matrix.
        max_iter : int, default=50
            Maximum iterations.
        step : float, default=0.5
            Scaling step toward benchmark.

        Returns
        -------
        np.ndarray
            Adjusted weights satisfying TE constraint.
        """
        w = weights.copy()
        diff = w - benchmark_weights
        te = np.sqrt(diff @ covariance @ diff) * np.sqrt(252)

        for _ in range(max_iter):
            if te <= self.max_te:
                break
            w = benchmark_weights + (w - benchmark_weights) * step
            w = w / w.sum()
            diff = w - benchmark_weights
            te = np.sqrt(diff @ covariance @ diff) * np.sqrt(252)

        return w


class ConcentrationConstraint:
    """
    Portfolio concentration constraints.

    - Max weight per asset
    - Max weight per sector
    - Min/max number of positions
    """

    def __init__(
        self,
        max_asset_weight: float = 0.08,
        max_sector_weight: float = 0.20,
        min_positions: int = 15,
        max_positions: int = 30,
    ) -> None:
        self.max_asset_weight = max_asset_weight
        self.max_sector_weight = max_sector_weight
        self.min_positions = min_positions
        self.max_positions = max_positions

    def apply_asset_cap(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Cap individual asset weights."""
        return np.clip(weights, 0, self.max_asset_weight)

    def apply_sector_cap(
        self,
        weights: np.ndarray,
        sector_map: Dict[int, str],
        sector_limit: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Cap sector weights and redistribute.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector (n_assets,).
        sector_map : dict
            Asset index -> sector name.
        sector_limit : dict, optional
            Sector -> max weight. Defaults to max_sector_weight for all.

        Returns
        -------
        np.ndarray
            Weights with sector constraints.
        """
        w = weights.copy()
        sectors: Dict[str, List[int]] = {}
        for idx, sector in sector_map.items():
            sectors.setdefault(sector, []).append(idx)

        limit = sector_limit or {}
        for _ in range(20):
            any_capped = False
            for sector, indices in sectors.items():
                max_w = limit.get(sector, self.max_sector_weight)
                sector_w = w[indices].sum()
                if sector_w > max_w:
                    scale = max_w / sector_w
                    w[indices] *= scale
                    any_capped = True
            if any_capped:
                w = w / w.sum()
            else:
                break

        return w


class RiskBudget:
    """
    Risk budgeting / equal risk contribution.

    Computes weights such that each asset contributes equally
    to total portfolio risk (for Risk Parity portfolios).
    """

    def __init__(self, tolerance: float = 1e-6, max_iter: int = 1000) -> None:
        self.tolerance = tolerance
        self.max_iter = max_iter

    def equal_risk_contribution(
        self,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Compute equal risk contribution (ERC) weights.

        Solves: w_i ∝ sqrt(diag(Σ)_ii) normalized, then iteratively
        adjusts for equal marginal risk contribution.

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix.

        Returns
        -------
        np.ndarray
            ERC weights.
        """
        n = covariance.shape[0]
        w = np.ones(n) / n

        for iteration in range(self.max_iter):
            sigma = np.sqrt(w @ covariance @ w)
            mrc = covariance @ w / sigma
            rc = w * mrc

            target_rc = np.ones(n) * rc.mean()
            diff = rc - target_rc

            if np.sqrt(diff @ diff) < self.tolerance:
                break

            step = 0.1
            w = w - step * diff * w
            w = np.maximum(w, 1e-8)
            w = w / w.sum()

        return w
