import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from qpe.regime_detector import RegimeDetector


class AlphaEngine:
    """
    Alpha generation engine with regime-aware factor allocation.

    Adjusts QPE factor weights dynamically based on detected market regime,
    then computes a composite alpha score for each asset.

    Regime adjustments:
    - Bull:   growth +0.10, quality +0.05
    - Bear:   safety +0.10, quality +0.05, growth -0.10
    - Crisis: safety +0.15, quality +0.10, growth -0.15, valuation -0.05
    - Recovery: growth +0.10, valuation +0.05, safety -0.10
    - High Rates: dividends +0.10, safety +0.05, growth -0.10
    - Low Rates:  growth +0.10, valuation +0.05, dividends -0.10
    """

    REGIME_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
        "bull": {"growth": 0.10, "quality": 0.05, "dividends": -0.05, "safety": -0.05, "valuation": -0.05},
        "bear": {"safety": 0.10, "quality": 0.05, "growth": -0.10, "valuation": -0.05},
        "crisis": {"safety": 0.15, "quality": 0.10, "growth": -0.15, "valuation": -0.05, "dividends": -0.05},
        "recovery": {"growth": 0.10, "valuation": 0.05, "safety": -0.10, "dividends": -0.05},
        "high_rates": {"dividends": 0.10, "safety": 0.05, "growth": -0.10, "valuation": -0.05},
        "low_rates": {"growth": 0.10, "valuation": 0.05, "dividends": -0.10, "safety": -0.05},
    }

    BASE_WEIGHTS = {
        "quality": 0.25,
        "valuation": 0.20,
        "dividends": 0.20,
        "growth": 0.20,
        "safety": 0.15,
    }

    def __init__(self) -> None:
        self.regime_detector = RegimeDetector()
        self.current_regime: str = "unknown"
        self.factor_weights: Dict[str, float] = dict(self.BASE_WEIGHTS)

    def detect_regime(self, benchmark_returns: pd.Series, cdi_rate: Optional[float] = None) -> str:
        """
        Detect current market regime.

        Parameters
        ----------
        benchmark_returns : pd.Series
            Benchmark daily returns.
        cdi_rate : float, optional
            Current CDI rate.

        Returns
        -------
        str
            Detected regime key.
        """
        import pandas as pd
        result = self.regime_detector.detect(benchmark_returns, cdi_rate=cdi_rate)
        self.current_regime = result.get("regime", "unknown")
        return self.current_regime

    def get_factor_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get factor weights adjusted for a given regime.

        Parameters
        ----------
        regime : str, optional
            Market regime. Uses current regime if None.

        Returns
        -------
        dict
            Adjusted factor weights.
        """
        r = regime or self.current_regime
        weights = dict(self.BASE_WEIGHTS)

        adj = self.REGIME_ADJUSTMENTS.get(r, {})
        for factor, delta in adj.items():
            if factor in weights:
                weights[factor] = max(0.05, min(0.50, weights[factor] + delta))

        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] = round(weights[k] / total, 4)

        self.factor_weights = weights
        return weights

    def compute_alpha(
        self,
        factor_scores: Dict[str, float],
        regime: Optional[str] = None,
    ) -> float:
        """
        Compute regime-adjusted alpha score for a single asset.

        alpha = Σ (w_i * factor_i)

        Parameters
        ----------
        factor_scores : dict
            Factor scores for the asset: quality, valuation,
            dividends, growth, safety.
        regime : str, optional
            Current market regime.

        Returns
        -------
        float
            Regime-adjusted alpha score (0-100).
        """
        weights = self.get_factor_weights(regime)
        alpha = 0.0
        for factor, w in weights.items():
            score = factor_scores.get(factor, 50)
            alpha += w * score
        return round(alpha, 1)

    def compute_alpha_batch(
        self,
        assets: List[Dict[str, Any]],
        regime: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute alpha scores for a batch of assets.

        Parameters
        ----------
        assets : list of dict
            Each dict must have 'ticker' and factor scores.
        regime : str, optional
            Current market regime.

        Returns
        -------
        list of dict
            Assets with added 'alpha_score' field.
        """
        weights = self.get_factor_weights(regime)

        results = []
        for a in assets:
            alpha = 0.0
            for factor, w in weights.items():
                score = a.get(factor, 50)
                alpha += w * score
            enriched = dict(a)
            enriched["alpha_score"] = round(alpha, 1)
            enriched["regime"] = regime or self.current_regime
            enriched["factor_weights"] = weights
            results.append(enriched)

        return sorted(results, key=lambda x: x["alpha_score"], reverse=True)
