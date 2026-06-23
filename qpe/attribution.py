import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from qpe.performance_metrics import PerformanceMetrics


@dataclass
class FactorContribution:
    """Contribution of a single factor to portfolio returns."""
    factor: str
    weight: float
    contribution_return: float
    contribution_sharpe: float
    contribution_alpha: float
    t_stat: float
    significant: bool


class AlphaAttributionEngine:
    """
    Decompose portfolio alpha into factor contributions.

    For each QPE factor (Quality, Valuation, Dividends, Growth, Safety),
    computes the contribution to:
    - Total return
    - Sharpe ratio
    - Alpha (CAPM)
    """

    FACTORS = ["quality", "valuation", "dividends", "growth", "safety"]

    def __init__(self, risk_free_rate: float = 0.1325) -> None:
        self.risk_free_rate = risk_free_rate
        self.pm = PerformanceMetrics(risk_free_rate=risk_free_rate)

    def attribute(
        self,
        factor_weights: Dict[str, float],
        factor_scores: pd.DataFrame,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        portfolio_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Perform full alpha attribution.

        For each factor, constructs a sub-portfolio of assets
        sorted by that factor and measures its performance.

        Parameters
        ----------
        factor_weights : dict
            Current factor weights (from AlphaEngine).
        factor_scores : pd.DataFrame
            Asset factor scores (index=ticker, columns=factors).
        asset_returns : pd.DataFrame
            Daily returns for each asset.
        benchmark_returns : pd.Series
            Benchmark daily returns.
        portfolio_weights : dict
            Current portfolio allocation (ticker -> weight).

        Returns
        -------
        dict
            Factor contributions and portfolio decomposition.
        """
        tickers = list(portfolio_weights.keys())
        available = [t for t in tickers if t in asset_returns.columns]

        contributions: List[FactorContribution] = []
        total_return = 0.0
        total_alpha = 0.0

        for factor in self.FACTORS:
            w = factor_weights.get(factor, 0)
            if w <= 0 or not available:
                contributions.append(FactorContribution(
                    factor=factor, weight=w,
                    contribution_return=0.0, contribution_sharpe=0.0,
                    contribution_alpha=0.0, t_stat=0.0, significant=False,
                ))
                continue

            sorted_assets = sorted(
                [(t, factor_scores.loc[t, factor]) for t in available if t in factor_scores.index],
                key=lambda x: x[1],
                reverse=True,
            )
            top_n = max(len(sorted_assets) // 3, 1)
            factor_tickers = [t for t, _ in sorted_assets[:top_n]]

            if not factor_tickers:
                contributions.append(FactorContribution(
                    factor=factor, weight=w,
                    contribution_return=0.0, contribution_sharpe=0.0,
                    contribution_alpha=0.0, t_stat=0.0, significant=False,
                ))
                continue

            factor_ret = self._compute_factor_return(asset_returns, factor_tickers)
            aligned = pd.concat(
                {"factor": factor_ret, "benchmark": benchmark_returns},
                axis=1,
            ).dropna()

            ret = self.pm.cumulative_return(aligned["factor"])
            metrics = self.pm.all_metrics(aligned["factor"], aligned["benchmark"])

            t_stat, significant = self._test_significance(
                aligned["factor"], aligned["benchmark"]
            )

            contrib = FactorContribution(
                factor=factor,
                weight=w,
                contribution_return=float(ret) * w,
                contribution_sharpe=float(metrics.get("sharpe_ratio", 0)) * w,
                contribution_alpha=float(metrics.get("alpha", 0)) * w,
                t_stat=round(float(t_stat), 4),
                significant=significant,
            )
            contributions.append(contrib)
            total_return += contrib.contribution_return
            total_alpha += contrib.contribution_alpha

        portfolio_return = self.pm.cumulative_return(
            self._compute_portfolio_return(asset_returns, portfolio_weights)
        )

        return {
            "fatores": contributions,
            "total_contribuicao_retorno": round(total_return, 4),
            "total_contribuicao_alpha": round(total_alpha, 4),
            "retorno_real_carteira": round(float(portfolio_return), 4),
            "fatores_significativos": sum(1 for c in contributions if c.significant),
            "melhor_fator": max(contributions, key=lambda c: c.contribution_alpha).factor,
            "pior_fator": min(contributions, key=lambda c: c.contribution_alpha).factor,
        }

    def _compute_factor_return(
        self, asset_returns: pd.DataFrame, tickers: List[str],
    ) -> pd.Series:
        """Compute equal-weighted return for a factor portfolio."""
        available = [t for t in tickers if t in asset_returns.columns]
        if not available:
            return pd.Series(dtype=float)
        subset = asset_returns[available].dropna(how="all")
        return subset.mean(axis=1)

    def _compute_portfolio_return(
        self, asset_returns: pd.DataFrame, weights: Dict[str, float],
    ) -> pd.Series:
        """Compute weighted portfolio return."""
        available = {t: w for t, w in weights.items() if t in asset_returns.columns}
        if not available:
            return pd.Series(dtype=float)
        total_w = sum(available.values())
        if total_w == 0:
            return pd.Series(dtype=float)
        weighted = sum(
            asset_returns[t] * (w / total_w) for t, w in available.items()
        )
        return weighted

    def _test_significance(
        self, factor_returns: pd.Series, benchmark_returns: pd.Series,
    ) -> Tuple[float, bool]:
        """Test if factor excess returns are statistically significant."""
        from scipy import stats
        aligned = pd.concat(
            {"f": factor_returns, "b": benchmark_returns},
            axis=1,
        ).dropna()
        if len(aligned) < 10:
            return 0.0, False

        rf_period = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess = aligned["f"] - aligned["b"] - rf_period
        t_stat, p_value = stats.ttest_1samp(excess, 0)
        return float(t_stat), bool(p_value < 0.05)

    def report(self, attribution_result: Dict[str, Any]) -> str:
        """Generate a formatted alpha attribution report."""
        lines = []
        lines.append("## 📊 Alpha Attribution Report")
        lines.append("")
        lines.append(f"**Portfolio Return:** {attribution_result.get('retorno_real_carteira', 0)*100:.2f}%")
        lines.append(f"**Total Attribution Alpha:** {attribution_result.get('total_contribuicao_alpha', 0)*100:.2f}%")
        lines.append("")
        lines.append("| Factor | Weight | Return Contrib | Alpha Contrib | t-stat | Significant |")
        lines.append("|--------|--------|----------------|---------------|--------|-------------|")

        for c in attribution_result.get("fatores", []):
            sig = "✅" if c.significant else "❌"
            lines.append(
                f"| {c.factor.capitalize()} | {c.weight*100:.1f}% | "
                f"{c.contribution_return*100:.2f}% | {c.contribution_alpha*100:.2f}% | "
                f"{c.t_stat:.2f} | {sig} |"
            )

        lines.append("")
        best = attribution_result.get("melhor_fator", "-")
        worst = attribution_result.get("pior_fator", "-")
        lines.append(f"**Melhor fator:** {best}")
        lines.append(f"**Pior fator:** {worst}")
        lines.append(f"**Fatores significativos:** {attribution_result.get('fatores_significativos', 0)}/{len(self.FACTORS)}")

        return "\n".join(lines)
