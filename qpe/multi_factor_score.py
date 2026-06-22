import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.preprocessing import MinMaxScaler


class MultiFactorScore:
    """
    Compute a multi-factor score for asset selection.

    Factors:
    - Quality  (25%): ROE, ROIC, Margem Líquida
    - Valuation (20%): P/L, P/VP, EV/EBITDA
    - Dividends (20%): Dividend Yield, Consistency
    - Growth   (20%): CAGR Revenue, CAGR Net Income
    - Safety   (15%): Dívida Líquida/EBITDA, Liquidez Corrente
    """

    WEIGHTS = {
        "quality": 0.25,
        "valuation": 0.20,
        "dividends": 0.20,
        "growth": 0.20,
        "safety": 0.15,
    }

    def __init__(self) -> None:
        self.scaler = MinMaxScaler(feature_range=(0, 100))

    def _score_quality(self, row: Dict[str, Any]) -> float:
        """Quality sub-score: ROE (40%), ROIC (30%), Margem Líquida (30%)."""
        roe = row.get("roe")
        roic = row.get("roic")
        margin = row.get("margem_liquida")

        parts = []
        if roe is not None and roe != 0:
            parts.append(40.0 * min(max((roe + 0.3) / 0.6 * 100, 0), 100) / 100.0)
        if roic is not None and roic != 0:
            parts.append(30.0 * min(max((roic + 0.2) / 0.5 * 100, 0), 100) / 100.0)
        if margin is not None and margin != 0:
            parts.append(30.0 * min(max((margin + 0.3) / 0.6 * 100, 0), 100) / 100.0)

        if not parts:
            return 50.0
        return round(sum(parts) / sum(self._weights_quality(len(parts))) * 100, 1)

    def _weights_quality(self, n: int) -> List[float]:
        base = [40.0, 30.0, 30.0]
        return base[:n]

    def _score_valuation(self, row: Dict[str, Any]) -> float:
        """Valuation sub-score: P/L (35%), P/VP (35%), EV/EBITDA (30%)."""
        pl = row.get("pl")
        pvp = row.get("pvp")
        ev_ebit = row.get("ev_ebit")

        parts = []
        if pl is not None and pl > 0:
            parts.append(35.0 * min(max((20 - pl) / 18 * 100, 0), 100) / 100.0)
        if pvp is not None and pvp > 0:
            parts.append(35.0 * min(max((3 - pvp) / 2.8 * 100, 0), 100) / 100.0)
        if ev_ebit is not None and ev_ebit > 0:
            parts.append(30.0 * min(max((15 - ev_ebit) / 14 * 100, 0), 100) / 100.0)

        if not parts:
            return 50.0
        base_w = [35.0, 35.0, 30.0]
        w = base_w[:len(parts)]
        return round(sum(parts) / sum(w) * 100, 1)

    def _score_dividends(self, row: Dict[str, Any]) -> float:
        """Dividend sub-score: DY (60%), Consistency (40%)."""
        dy = row.get("dy")
        consistency = row.get("dividend_consistency")

        parts = []
        if dy is not None and dy > 0:
            dy_norm = min(max((dy - 0.02) / 0.15 * 100, 0), 100)
            parts.append(60.0 * dy_norm / 100.0)
        if consistency is not None:
            parts.append(40.0 * min(consistency * 100, 100) / 100.0)

        if not parts:
            return 40.0
        base_w = [60.0, 40.0]
        w = base_w[:len(parts)]
        return round(sum(parts) / sum(w) * 100, 1)

    def _score_growth(self, row: Dict[str, Any]) -> float:
        """Growth sub-score: CAGR Revenue (50%), CAGR Net Income (50%)."""
        cagr_rev = row.get("cagr_revenue")
        cagr_ni = row.get("cagr_net_income")

        parts = []
        if cagr_rev is not None:
            rev_norm = min(max((cagr_rev + 0.3) / 0.6 * 100, 0), 100)
            parts.append(50.0 * rev_norm / 100.0)
        if cagr_ni is not None:
            ni_norm = min(max((cagr_ni + 0.3) / 0.6 * 100, 0), 100)
            parts.append(50.0 * ni_norm / 100.0)

        if not parts:
            return 50.0
        base_w = [50.0, 50.0]
        w = base_w[:len(parts)]
        return round(sum(parts) / sum(w) * 100, 1)

    def _score_safety(self, row: Dict[str, Any]) -> float:
        """Safety sub-score: Dívida/EBITDA (50%), Liquidez Corrente (50%)."""
        debt = row.get("divida_pl")
        liquidity = row.get("liquidez_corrente")

        parts = []
        if debt is not None and debt > 0:
            debt_norm = min(max((3 - debt) / 3 * 100, 0), 100)
            parts.append(50.0 * debt_norm / 100.0)
        if liquidity is not None and liquidity > 0:
            liq_norm = min(max((liquidity - 0.5) / 3.5 * 100, 0), 100)
            parts.append(50.0 * liq_norm / 100.0)

        if not parts:
            return 50.0
        base_w = [50.0, 50.0]
        w = base_w[:len(parts)]
        return round(sum(parts) / sum(w) * 100, 1)

    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all factor scores and the final composite score.

        Parameters
        ----------
        row : dict
            Asset fundamentals with keys: roe, roic, margem_liquida, pl, pvp,
            ev_ebit, dy, dividend_consistency, cagr_revenue, cagr_net_income,
            divida_pl, liquidez_corrente.

        Returns
        -------
        dict
            Keys: quality, valuation, dividends, growth, safety, total_score.
        """
        quality = self._score_quality(row)
        valuation = self._score_valuation(row)
        dividends = self._score_dividends(row)
        growth = self._score_growth(row)
        safety = self._score_safety(row)

        total = (
            quality * self.WEIGHTS["quality"]
            + valuation * self.WEIGHTS["valuation"]
            + dividends * self.WEIGHTS["dividends"]
            + growth * self.WEIGHTS["growth"]
            + safety * self.WEIGHTS["safety"]
        )

        return {
            "quality": quality,
            "valuation": valuation,
            "dividends": dividends,
            "growth": growth,
            "safety": safety,
            "total_score": round(total, 1),
        }

    def batch_compute(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute scores for a DataFrame of assets.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Must contain columns matching the row dict keys.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional factor columns and total_score.
        """
        results = []
        for _, row in dataframe.iterrows():
            scores = self.compute(row.to_dict())
            results.append(scores)

        scores_df = pd.DataFrame(results)
        out = dataframe.reset_index(drop=True).copy()
        out = pd.concat([out, scores_df], axis=1)
        return out

    def apply_percentile_ranking(
        self,
        scores: pd.Series,
    ) -> pd.Series:
        """
        Convert raw scores to percentile ranks for better distribution.

        Results in a distribution like:
        90-100: Elite
        80-89:  Excelente
        70-79:  Boa
        60-69:  Média
        <60:    Fraca

        Parameters
        ----------
        scores : pd.Series
            Raw total scores.

        Returns
        -------
        pd.Series
            Percentile-ranked scores (0-100).
        """
        if scores.empty:
            return scores
        ranks = scores.rank(pct=True) * 100
        return ranks.round(1)

    @staticmethod
    def classify(score: float) -> str:
        """
        Classify a score into a category.

        Parameters
        ----------
        score : float
            Score value (0-100).

        Returns
        -------
        str
            Category label.
        """
        if score >= 90:
            return "Elite"
        elif score >= 80:
            return "Excelente"
        elif score >= 70:
            return "Boa"
        elif score >= 60:
            return "Média"
        else:
            return "Fraca"
