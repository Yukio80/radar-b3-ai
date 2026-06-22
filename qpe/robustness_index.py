import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


class RobustnessIndex:
    """
    Calculate the Índice de Robustez Patrimonial (IRP).

    IRP = 0.25 * Diversificação + 0.25 * Qualidade Média
        + 0.25 * Estabilidade Dividendos + 0.25 * Baixa Alavancagem

    Returns a score from 0 to 100 with classification.
    """

    CLASSIFICACOES = [
        (90, "Excelente"),
        (75, "Muito Boa"),
        (60, "Boa"),
        (40, "Regular"),
        (0, "Fraca"),
    ]

    def __init__(self) -> None:
        self.scores: Dict[str, float] = {}

    def diversification_score(
        self,
        num_assets: int,
        sector_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Score based on number of assets and sector concentration.

        Parameters
        ----------
        num_assets : int
            Number of assets in the portfolio.
        sector_weights : dict, optional
            Sector -> percentage weight.

        Returns
        -------
        float
            Score from 0 to 100.
        """
        base = min(num_assets / 20.0 * 100, 100)

        if sector_weights:
            shares = np.array(list(sector_weights.values()))
            herfindahl = (shares / 100.0 ** 2).sum()
            hhi_score = max(0, 100 - herfindahl * 100)
            base = 0.5 * base + 0.5 * hhi_score

        return round(base, 1)

    def average_quality_score(
        self,
        quality_scores: List[float],
    ) -> float:
        """
        Average quality score of portfolio assets.

        Parameters
        ----------
        quality_scores : list of float
            Quality sub-scores for each asset.

        Returns
        -------
        float
            Score from 0 to 100.
        """
        if not quality_scores:
            return 50.0
        return round(float(np.mean(quality_scores)), 1)

    def dividend_stability_score(
        self,
        dy_values: List[Optional[float]],
        consistency_years: Optional[List[Optional[float]]] = None,
    ) -> float:
        """
        Score based on dividend stability.

        Parameters
        ----------
        dy_values : list of float or None
            Dividend Yield values.
        consistency_years : list of float or None, optional
            Years of consistent dividend payments.

        Returns
        -------
        float
            Score from 0 to 100.
        """
        clean_dy = [d for d in dy_values if d is not None and d > 0]
        if not clean_dy:
            return 30.0

        dy_pct = np.array(clean_dy) * 100
        mean_dy = dy_pct.mean()
        std_dy = dy_pct.std() if len(dy_pct) > 1 else 0

        stability = 50.0
        if std_dy > 0:
            cv = std_dy / mean_dy if mean_dy > 0 else 1
            stability = max(0, 100 - cv * 50)

        dy_level = min(mean_dy / 10.0 * 100, 100)

        consistency = 50.0
        if consistency_years:
            clean_y = [c for c in consistency_years if c is not None]
            if clean_y:
                consistency = min(float(np.mean(clean_y)) / 10.0 * 100, 100)

        return round(0.4 * stability + 0.3 * dy_level + 0.3 * consistency, 1)

    def low_leverage_score(
        self,
        debt_values: List[Optional[float]],
    ) -> float:
        """
        Score based on low leverage.

        Parameters
        ----------
        debt_values : list of float or None
            Dívida/PL or Dívida Líquida/EBITDA values.

        Returns
        -------
        float
            Score from 0 to 100.
        """
        clean = [d for d in debt_values if d is not None and d > 0]
        if not clean:
            return 70.0

        ratios = np.array(clean)
        avg_ratio = ratios.mean()
        score = max(0, 100 - avg_ratio * 20)
        return round(score, 1)

    def compute(
        self,
        num_assets: int,
        quality_scores: List[float],
        dy_values: List[Optional[float]],
        debt_values: List[Optional[float]],
        sector_weights: Optional[Dict[str, float]] = None,
        consistency_years: Optional[List[Optional[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate IRP and all sub-scores.

        Parameters
        ----------
        num_assets : int
            Total assets in portfolio.
        quality_scores : list of float
            Quality scores per asset.
        dy_values : list of float or None
            DY per asset.
        debt_values : list of float or None
            Dívida/PL per asset.
        sector_weights : dict, optional
            Sector allocation percentages.
        consistency_years : list of float or None, optional
            Dividend consistency years.

        Returns
        -------
        dict
            IRP with sub-scores and classification.
        """
        self.scores["diversificacao"] = self.diversification_score(
            num_assets, sector_weights
        )
        self.scores["qualidade_media"] = self.average_quality_score(quality_scores)
        self.scores["estabilidade_dividendos"] = self.dividend_stability_score(
            dy_values, consistency_years
        )
        self.scores["baixa_alavancagem"] = self.low_leverage_score(debt_values)

        irp = (
            0.25 * self.scores["diversificacao"]
            + 0.25 * self.scores["qualidade_media"]
            + 0.25 * self.scores["estabilidade_dividendos"]
            + 0.25 * self.scores["baixa_alavancagem"]
        )

        classification = self.classify(irp)

        return {
            "IRP": round(irp, 1),
            "classificacao": classification,
            "sub_scores": self.scores,
        }

    @staticmethod
    def classify(irp: float) -> str:
        """
        Classify an IRP score.

        Parameters
        ----------
        irp : float
            IRP value.

        Returns
        -------
        str
            Classification label.
        """
        for threshold, label in RobustnessIndex.CLASSIFICACOES:
            if irp >= threshold:
                return label
        return "Fraca"
