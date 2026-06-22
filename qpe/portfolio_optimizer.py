import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


class PortfolioOptimizer:
    """
    Calculate optimal portfolio weights based on asset scores.

    Replaces equal-weight allocation with score-weighted allocation,
    subject to minimum and maximum weight constraints.
    """

    def __init__(
        self,
        peso_min: float = 0.02,
        peso_max: float = 0.10,
    ) -> None:
        """
        Parameters
        ----------
        peso_min : float, default=0.02
            Minimum weight per asset (2%).
        peso_max : float, default=0.10
            Maximum weight per asset (10%).
        """
        self.peso_min = peso_min
        self.peso_max = peso_max

    def optimize(
        self,
        scores: List[float],
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate optimal weights from score values.

        Weight = score / sum(scores), then clamped to [peso_min, peso_max].
        Uses iterative proportional fitting to respect constraints.

        Parameters
        ----------
        scores : list of float
            Score values for each asset.
        tickers : list of str, optional
            Ticker symbols for each asset.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, score, weight_raw, weight_pct.
        """
        arr = np.array(scores, dtype=float)
        n = len(arr)
        total = arr.sum()
        if total <= 0:
            raw = np.ones(n) / n
        else:
            raw = arr / total

        weights = raw.copy()
        for _ in range(50):
            clamped = np.clip(weights, self.peso_min, self.peso_max)
            if np.allclose(weights, clamped, atol=1e-6):
                weights = clamped
                break
            weights = np.clip(weights, self.peso_min, self.peso_max)
            wsum = weights.sum()
            if wsum > 0:
                weights = weights / wsum
        else:
            weights = np.clip(weights, self.peso_min, self.peso_max)
            weights = weights / weights.sum()

        result = pd.DataFrame({
            "ticker": tickers if tickers else [f"Ativo_{i}" for i in range(n)],
            "score": scores,
            "weight_raw": raw.round(4),
            "weight_pct": (weights * 100).round(2),
        })

        return result.sort_values("weight_pct", ascending=False).reset_index(drop=True)

    @staticmethod
    def allocate_by_profile(
        tickers: List[str],
        scores: List[float],
        profile_allocation: Dict[str, float],
        profile_tickers: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Allocate assets by profile categories.

        Parameters
        ----------
        tickers : list of str
            All tickers.
        scores : list of float
            All scores.
        profile_allocation : dict
            Category -> target percentage allocation.
        profile_tickers : dict
            Category -> list of tickers belonging to that category.

        Returns
        -------
        dict
            Portfolio with ticker-level weights.
        """
        opt = PortfolioOptimizer()
        all_weights: Dict[str, float] = {}

        for category, cat_pct in profile_allocation.items():
            cat_tickers = profile_tickers.get(category, [])
            if not cat_tickers:
                continue

            cat_scores = [
                scores[tickers.index(t)] if t in tickers else 0
                for t in cat_tickers
            ]
            cat_result = opt.optimize(cat_scores, cat_tickers)
            for _, row in cat_result.iterrows():
                all_weights[row["ticker"]] = row["weight_pct"] * cat_pct / 100.0

        total = sum(all_weights.values())
        if total > 0:
            for t in all_weights:
                all_weights[t] = round(all_weights[t] / total * 100, 2)

        return {
            "weights": all_weights,
            "total_assets": len(all_weights),
            "weight_sum": round(sum(all_weights.values()), 2),
        }
