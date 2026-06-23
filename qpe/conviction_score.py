import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class ConvictionEngine:
    """
    Compute conviction scores for assets by combining multiple signals.

    Conviction = w1 * QPE_Score + w2 * IRP + w3 * Regime_Alignment
                + w4 * Estabilidade + w5 * WalkForward_Consistency

    Scale:
    0-20   Muito Baixa
    20-40  Baixa
    40-60  Média
    60-80  Alta
    80-100 Muito Alta
    """

    WEIGHTS = {
        "qpe_score": 0.30,
        "irp": 0.20,
        "regime_alignment": 0.15,
        "estabilidade": 0.20,
        "walk_forward": 0.15,
    }

    def __init__(self) -> None:
        self.historical_scores: Dict[str, List[float]] = {}

    def _normalize(self, value: float, min_v: float, max_v: float) -> float:
        if max_v <= min_v:
            return 50.0
        return max(0, min(100, (value - min_v) / (max_v - min_v) * 100))

    def _regime_alignment_score(
        self,
        factor_scores: Dict[str, float],
        regime: str,
    ) -> float:
        """
        Score how well an asset's factor profile aligns with the current regime.

        Bull: growth+quality heavy
        Bear: safety+quality heavy
        Crisis: safety heavy
        Recovery: growth heavy
        High rates: dividends heavy
        Low rates: growth+valuation heavy
        """
        from qpe.alpha_engine import AlphaEngine
        ae = AlphaEngine()
        weights = ae.get_factor_weights(regime)
        alignment = 0.0
        for factor, w in weights.items():
            score = factor_scores.get(factor, 50)
            alignment += w * score
        return alignment

    def _estabilidade_score(
        self,
        ticker: str,
        current_score: float,
    ) -> float:
        """
        Score based on historical score stability.
        Lower deviation = higher stability score.
        """
        hist = self.historical_scores.get(ticker, [current_score])
        if len(hist) < 2:
            return 50.0
        mean_s = np.mean(hist)
        std_s = np.std(hist) if len(hist) > 1 else 0
        if mean_s == 0:
            return 50.0
        cv = std_s / abs(mean_s)
        stability = max(0, 100 - cv * 200)
        return float(stability)

    def _walk_forward_score(
        self,
        walk_forward_result: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Score based on walk-forward consistency."""
        if walk_forward_result is None:
            return 50.0
        taxa = walk_forward_result.get("taxa_acerto", 50)
        retorno = walk_forward_result.get("retorno_medio_teste", 0)
        score = taxa * 0.6 + max(0, min(100, retorno * 2)) * 0.4
        return float(score)

    def compute(
        self,
        ticker: str,
        qpe_score: float,
        irp_score: float,
        factor_scores: Dict[str, float],
        regime: str = "unknown",
        walk_forward_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute conviction score for a single asset.

        Parameters
        ----------
        ticker : str
            Asset ticker.
        qpe_score : float
            QPE total score (0-100).
        irp_score : float
            IRP score (0-100), or 50 if unavailable.
        factor_scores : dict
            Factor scores: quality, valuation, dividends, growth, safety.
        regime : str
            Current market regime.
        walk_forward_result : dict, optional
            Walk-forward validation results.

        Returns
        -------
        dict
            Ticker, conviction_score, conviction_label, and components.
        """
        qpe_norm = self._normalize(qpe_score, 0, 100)
        irp_norm = self._normalize(irp_score, 0, 100)
        regime_alignment = self._regime_alignment_score(factor_scores, regime)
        estab = self._estabilidade_score(ticker, qpe_score)
        wf = self._walk_forward_score(walk_forward_result)

        conviction = (
            self.WEIGHTS["qpe_score"] * qpe_norm
            + self.WEIGHTS["irp"] * irp_norm
            + self.WEIGHTS["regime_alignment"] * regime_alignment
            + self.WEIGHTS["estabilidade"] * estab
            + self.WEIGHTS["walk_forward"] * wf
        )

        conviction = max(0, min(100, conviction))
        label = self.classify(conviction)

        return {
            "ticker": ticker,
            "conviction_score": round(conviction, 1),
            "conviction_label": label,
            "componentes": {
                "qpe_score": round(qpe_norm, 1),
                "irp_score": round(irp_norm, 1),
                "regime_alignment": round(regime_alignment, 1),
                "estabilidade": round(estab, 1),
                "walk_forward": round(wf, 1),
            },
        }

    def compute_batch(
        self,
        assets: List[Dict[str, Any]],
        regime: str = "unknown",
        walk_forward_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute conviction scores for a batch of assets.

        Parameters
        ----------
        assets : list of dict
            Each asset must have: ticker, total_score, irp_score,
            and factor scores (quality, valuation, etc.).
        regime : str
            Current market regime.
        walk_forward_result : dict, optional
            Walk-forward results.

        Returns
        -------
        list of dict
            Assets with conviction scores added.
        """
        results = []
        for a in assets:
            factor_scores = {
                "quality": a.get("quality", 50),
                "valuation": a.get("valuation", 50),
                "dividends": a.get("dividends", 50),
                "growth": a.get("growth", 50),
                "safety": a.get("safety", 50),
            }
            conviction = self.compute(
                ticker=a["ticker"],
                qpe_score=a.get("total_score", a.get("alpha_score", 50)),
                irp_score=a.get("irp_score", 50),
                factor_scores=factor_scores,
                regime=regime,
                walk_forward_result=walk_forward_result,
            )
            enriched = dict(a)
            enriched.update(conviction)
            results.append(enriched)

        return sorted(results, key=lambda x: x["conviction_score"], reverse=True)

    @staticmethod
    def classify(score: float) -> str:
        if score >= 80:
            return "Muito Alta"
        elif score >= 60:
            return "Alta"
        elif score >= 40:
            return "Media"
        elif score >= 20:
            return "Baixa"
        return "Muito Baixa"

    def update_historical_scores(
        self,
        ticker: str,
        score: float,
        max_history: int = 12,
    ) -> None:
        """Update historical score tracking for stability calculation."""
        if ticker not in self.historical_scores:
            self.historical_scores[ticker] = []
        self.historical_scores[ticker].append(score)
        if len(self.historical_scores[ticker]) > max_history:
            self.historical_scores[ticker] = self.historical_scores[ticker][-max_history:]
