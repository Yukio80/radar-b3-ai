import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class QPEMarketScore:
    """
    Measure overall market attractiveness.

    Formula:
    MarketScore = w1 * media_scores + w2 * valuation_dispersion
                 + w3 * quality_level + w4 * regime_score

    Scale:
    0-20   Muito Caro
    20-40  Caro
    40-60  Neutro
    60-80  Atrativo
    80-100 Muito Atrativo
    """

    WEIGHTS = {
        "media_scores": 0.30,
        "valuation_dispersion": 0.20,
        "quality_level": 0.25,
        "regime_score": 0.25,
    }

    REGIME_MAP = {
        "bull": 60,
        "bear": 30,
        "crisis": 20,
        "recovery": 70,
        "high_rates": 40,
        "low_rates": 65,
        "unknown": 50,
    }

    @staticmethod
    def compute(
        scores: List[float],
        factor_scores: List[Dict[str, float]],
        regime: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Compute the overall market score.

        Parameters
        ----------
        scores : list of float
            QPE total scores for the universe.
        factor_scores : list of dict
            Factor scores for each asset.
        regime : str
            Current market regime.

        Returns
        -------
        dict
            Market score, label, and components.
        """
        if not scores:
            return {"market_score": 50.0, "market_label": "Neutro"}

        # Media dos scores do universo
        media_scores = float(np.mean(scores))
        media_norm = min(100, max(0, media_scores))

        # Dispersão de valuation (quanto maior a dispersão, mais oportunidades)
        if factor_scores:
            valuation_scores = [f.get("valuation", 50) for f in factor_scores]
            quality_scores = [f.get("quality", 50) for f in factor_scores]
            std_val = float(np.std(valuation_scores)) if len(valuation_scores) > 1 else 0
            valuation_dispersion = min(100, std_val * 5)
            quality_level = float(np.mean(quality_scores))
        else:
            valuation_dispersion = 50.0
            quality_level = 50.0

        # Regime score
        regime_score = QPEMarketScore.REGIME_MAP.get(regime, 50)

        market_score = (
            QPEMarketScore.WEIGHTS["media_scores"] * media_norm
            + QPEMarketScore.WEIGHTS["valuation_dispersion"] * valuation_dispersion
            + QPEMarketScore.WEIGHTS["quality_level"] * quality_level
            + QPEMarketScore.WEIGHTS["regime_score"] * regime_score
        )

        market_score = max(0, min(100, market_score))
        label = QPEMarketScore.classify(market_score)

        return {
            "market_score": round(market_score, 1),
            "market_label": label,
            "total_ativos_analisados": len(scores),
            "componentes": {
                "media_scores_universo": round(media_norm, 1),
                "dispersao_valuation": round(valuation_dispersion, 1),
                "nivel_qualidade": round(quality_level, 1),
                "regime_score": round(regime_score, 1),
            },
        }

    @staticmethod
    def classify(score: float) -> str:
        if score >= 80:
            return "Muito Atrativo"
        elif score >= 60:
            return "Atrativo"
        elif score >= 40:
            return "Neutro"
        elif score >= 20:
            return "Caro"
        return "Muito Caro"
