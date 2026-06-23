import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from qpe.portfolio_profiles import PortfolioProfile, get_profile, PROFILES
from qpe.conviction_score import ConvictionEngine
from qpe.market_score import QPEMarketScore
from qpe.alpha_engine import AlphaEngine
from qpe.portfolio_construction import MeanVarianceOptimizer, TwoStagePortfolioBuilder
from qpe.risk_models import RiskBudget
from qpe.covariance_models import auto_select_covariance
from qpe.performance_metrics import PerformanceMetrics
from qpe.robustness_index import RobustnessIndex
from qpe.stress_test import StressTest
from qpe.enhanced_stress import AdvancedStressTest


@dataclass
class RecommendedPortfolio:
    """Output of a portfolio recommendation."""
    profile: str
    description: str
    objective: str
    regime: str
    market_score: Dict[str, Any]
    positions: List[Dict[str, Any]]
    weights: Dict[str, float]
    metrics: Dict[str, Any]
    conviction_media: float
    irp_result: Dict[str, Any]
    stress_test: Dict[str, Any]
    advanced_stress: Dict[str, Any]
    score_medio: float
    explicacoes: List[Dict[str, Any]]


class RecommendationEngine:
    """
    Generate professional portfolio recommendations for different investment profiles.

    For each profile:
    1. Screen universe by profile-specific criteria
    2. Compute conviction scores
    3. Apply allocation method (risk parity, max sharpe, min variance)
    4. Compute portfolio metrics and risk assessment
    """

    def __init__(self) -> None:
        self.conviction = ConvictionEngine()
        self.alpha_engine = AlphaEngine()
        self.robustness = RobustnessIndex()
        self.pm = PerformanceMetrics(risk_free_rate=0.1325)
        self.current_regime: str = "unknown"

    def recommend(
        self,
        profile_name: str,
        scored_assets: List[Dict[str, Any]],
        regime: str = "unknown",
        returns: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RecommendedPortfolio:
        """
        Generate a portfolio recommendation for a given profile.

        Parameters
        ----------
        profile_name : str
            Profile key: core, alpha, dividendos, valor, crescimento, defensiva.
        scored_assets : list of dict
            Assets with scores and factor data.
        regime : str
            Current market regime.
        returns : pd.DataFrame, optional
            Historical returns for covariance estimation.
        benchmark_returns : pd.Series, optional
            Benchmark returns for metrics.

        Returns
        -------
        RecommendedPortfolio
            Complete recommendation with positions, weights, and metrics.
        """
        profile = get_profile(profile_name)
        self.current_regime = regime

        screened = self._screen_assets(scored_assets, profile)
        if not screened:
            screened = scored_assets[:profile.max_positions]

        for a in screened:
            a["irp_score"] = self._estimate_irp_score(a)

        enriched = self.conviction.compute_batch(
            screened, regime=regime, walk_forward_result=None,
        )

        weights = self._allocate(enriched, profile, returns)
        tickers = list(weights.keys())

        weights_pct = {t: w * 100 for t, w in weights.items()}
        sector_map = {a["ticker"]: a.get("setor", "Outros") for a in enriched}

        metrics = self._compute_metrics(enriched, weights, returns, benchmark_returns)

        irp_result = self.robustness.compute(
            num_assets=len(weights),
            quality_scores=[a.get("quality", 50) for a in enriched],
            dy_values=[a.get("dy") for a in enriched],
            debt_values=[a.get("divida_pl") for a in enriched],
            sector_weights=None,
        )

        st = StressTest()
        stress_result = st.run_all(weights_pct, sector_map)

        ast = AdvancedStressTest()
        adv_stress = ast.run_all(weights_pct, sector_map)

        conviction_media = float(np.mean([a.get("conviction_score", 50) for a in enriched])) if enriched else 0
        score_medio = float(np.mean([a.get("total_score", a.get("alpha_score", 50)) for a in enriched])) if enriched else 0

        positions = [
            {
                "ticker": a["ticker"],
                "peso": round(weights.get(a["ticker"], 0) * 100, 2),
                "score": a.get("total_score", a.get("alpha_score", 0)),
                "conviction": a.get("conviction_score", 0),
                "conviction_label": a.get("conviction_label", ""),
                "qualidade": a.get("quality", 0),
                "valuation": a.get("valuation", 0),
                "dividendos": a.get("dividends", 0),
                "crescimento": a.get("growth", 0),
                "seguranca": a.get("safety", 0),
            }
            for a in enriched
            if weights.get(a["ticker"], 0) > 0.001
        ]

        market_score = self._compute_market_score(scored_assets, regime)

        explicacoes = self._generate_explanations(enriched, weights, profile)

        return RecommendedPortfolio(
            profile=profile.name,
            description=profile.description,
            objective=profile.objective,
            regime=regime,
            market_score=market_score,
            positions=positions,
            weights=weights,
            metrics=metrics,
            conviction_media=round(conviction_media, 1),
            irp_result=irp_result,
            stress_test=stress_result,
            advanced_stress=adv_stress,
            score_medio=round(score_medio, 1),
            explicacoes=explicacoes,
        )

    def recommend_all(
        self,
        scored_assets: List[Dict[str, Any]],
        regime: str = "unknown",
        returns: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, RecommendedPortfolio]:
        """
        Generate recommendations for all profiles.

        Returns
        -------
        dict
            Profile name -> RecommendedPortfolio.
        """
        results = {}
        for name in PROFILES:
            results[name] = self.recommend(
                name, scored_assets, regime, returns, benchmark_returns,
            )
        return results

    def _screen_assets(
        self,
        assets: List[Dict[str, Any]],
        profile: PortfolioProfile,
    ) -> List[Dict[str, Any]]:
        """Apply profile-specific screening filters."""
        screened = []
        for a in assets:
            passes = True
            for field, threshold in profile.score_thresholds.items():
                val = a.get(field, 0)
                if val is None:
                    val = 0
                if val < threshold:
                    passes = False
                    break

            irp_score = self._estimate_irp_score(a)
            if profile.min_irp > 0 and irp_score < profile.min_irp:
                passes = False

            if passes:
                screened.append(a)

        sort_key = profile.sort_key
        screened.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
        return screened[:profile.max_positions]

    @staticmethod
    def _estimate_irp_score(asset: Dict[str, Any]) -> float:
        quality = asset.get("quality", 50) or 50
        dy_score = min(100, (asset.get("dy", 0) or 0) * 500)
        debt_score = 100 - min(100, (asset.get("divida_pl", 0) or 0) * 20)
        irp = 0.4 * quality + 0.3 * dy_score + 0.3 * debt_score
        return float(irp)

    def _allocate(
        self,
        enriched: List[Dict[str, Any]],
        profile: PortfolioProfile,
        returns: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """Allocate weights based on profile method."""
        tickers = [a["ticker"] for a in enriched]

        if profile.allocation_method == "risk_parity":
            return self._allocate_risk_parity(enriched, returns)
        elif profile.allocation_method == "max_sharpe":
            return self._allocate_max_sharpe(enriched, returns, profile)
        elif profile.allocation_method == "min_variance":
            return self._allocate_min_variance(enriched, returns, profile)
        else:
            return self._allocate_equal(enriched)

    def _allocate_equal(self, enriched: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(enriched)
        if n == 0:
            return {}
        w = 1.0 / n
        return {a["ticker"]: w for a in enriched}

    def _allocate_risk_parity(
        self, enriched: List[Dict[str, Any]], returns: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        tickers = [a["ticker"] for a in enriched]
        if returns is not None and len(tickers) >= 2:
            available = [t for t in tickers if t in returns.columns]
            if len(available) >= 2:
                ret_subset = returns[available].dropna()
                if len(ret_subset) >= 20:
                    cov_result = auto_select_covariance(ret_subset.values)
                    rb = RiskBudget()
                    w = rb.equal_risk_contribution(cov_result.covariance)
                    return {t: float(w[i]) for i, t in enumerate(available)}

        return self._allocate_equal(enriched)

    def _allocate_max_sharpe(
        self, enriched: List[Dict[str, Any]], returns: Optional[pd.DataFrame],
        profile: PortfolioProfile,
    ) -> Dict[str, float]:
        tickers = [a["ticker"] for a in enriched]
        if returns is not None and len(tickers) >= 2:
            available = [t for t in tickers if t in returns.columns]
            if len(available) >= 2:
                ret_subset = returns[available].dropna()
                if len(ret_subset) >= 20:
                    cov_result = auto_select_covariance(ret_subset.values)
                    alpha_scores = np.array([a.get("alpha_score", a.get("total_score", 50)) for a in enriched if a["ticker"] in available])
                    expected_ret = alpha_scores / alpha_scores.mean() * 0.15
                    mvo = MeanVarianceOptimizer(
                        max_weight=profile.max_asset_weight,
                        risk_free_rate=0.1325,
                    )
                    result = mvo.optimize(expected_ret, cov_result.covariance, available, method="max_sharpe")
                    return {t: w for t, w in result.weights.items() if w > 0.001}

        return self._allocate_equal(enriched)

    def _allocate_min_variance(
        self, enriched: List[Dict[str, Any]], returns: Optional[pd.DataFrame],
        profile: PortfolioProfile,
    ) -> Dict[str, float]:
        tickers = [a["ticker"] for a in enriched]
        if returns is not None and len(tickers) >= 2:
            available = [t for t in tickers if t in returns.columns]
            if len(available) >= 2:
                ret_subset = returns[available].dropna()
                if len(ret_subset) >= 20:
                    cov_result = auto_select_covariance(ret_subset.values)
                    mvo = MeanVarianceOptimizer(
                        max_weight=profile.max_asset_weight,
                        risk_free_rate=0.1325,
                    )
                    result = mvo.optimize(
                        np.ones(len(available)) * 0.10,
                        cov_result.covariance, available, method="min_variance",
                    )
                    return {t: w for t, w in result.weights.items() if w > 0.001}

        return self._allocate_equal(enriched)

    def _compute_metrics(
        self, enriched: List[Dict[str, Any]],
        weights: Dict[str, float],
        returns: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
    ) -> Dict[str, Any]:
        if returns is None or returns.empty:
            return {"retorno_anualizado": 0, "sharpe_ratio": 0}

        pf_ret = self._compute_portfolio_returns(returns, weights)
        if pf_ret.empty:
            return {"retorno_anualizado": 0, "sharpe_ratio": 0}

        return self.pm.all_metrics(pf_ret, benchmark_returns)

    @staticmethod
    def _compute_portfolio_returns(
        returns: pd.DataFrame, weights: Dict[str, float],
    ) -> pd.Series:
        available = {t: w for t, w in weights.items() if t in returns.columns}
        if not available:
            return pd.Series(dtype=float)
        total = sum(available.values())
        if total == 0:
            return pd.Series(dtype=float)
        weighted = sum(returns[t] * (w / total) for t, w in available.items())
        return weighted.dropna()

    def _compute_market_score(
        self, assets: List[Dict[str, Any]], regime: str,
    ) -> Dict[str, Any]:
        scores = [a.get("total_score", a.get("alpha_score", 50)) or 50 for a in assets]
        factor_scores = [
            {
                "quality": a.get("quality", 50),
                "valuation": a.get("valuation", 50),
                "dividends": a.get("dividends", 50),
                "growth": a.get("growth", 50),
                "safety": a.get("safety", 50),
            }
            for a in assets
        ]
        return QPEMarketScore.compute(scores, factor_scores, regime)

    def _generate_explanations(
        self,
        enriched: List[Dict[str, Any]],
        weights: Dict[str, float],
        profile: PortfolioProfile,
    ) -> List[Dict[str, Any]]:
        explained = []
        for a in enriched:
            t = a["ticker"]
            w = weights.get(t, 0)
            if w < 0.001:
                continue

            motivos = []
            pontos_fortes = []
            riscos = []

            qpe = a.get("total_score", a.get("alpha_score", 50))
            cv = a.get("conviction_score", 50)
            cv_label = a.get("conviction_label", "")

            motivos.append(f"Score QPE: {qpe:.1f}/100")
            motivos.append(f"Conviction: {cv:.1f} ({cv_label})")

            for factor in ["quality", "valuation", "dividends", "growth", "safety"]:
                val = a.get(factor, 0)
                threshold = profile.score_thresholds.get(factor, 0)
                if val >= threshold and val >= 70:
                    pontos_fortes.append(
                        f"{factor.capitalize()} score {val:.0f}/100"
                    )
                elif val < 40:
                    riscos.append(
                        f"{factor.capitalize()} baixo ({val:.0f}/100)"
                    )

            explained.append({
                "ticker": t,
                "peso": round(w * 100, 2),
                "score": qpe,
                "conviction": cv,
                "conviction_label": cv_label,
                "motivos": motivos,
                "pontos_fortes": pontos_fortes,
                "riscos": riscos,
            })

        return explained
