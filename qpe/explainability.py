import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class Explainability:
    """
    Generate human-readable explanations for each factor score.

    For each sub-score, compares the asset value to a reference
    (industry average or global median) and produces a reason.
    """

    REFERENCE = {
        "roe": 0.12,
        "roic": 0.10,
        "margem_liquida": 0.08,
        "pl": 15.0,
        "pvp": 1.5,
        "ev_ebit": 10.0,
        "dy": 0.04,
        "cagr_revenue": 0.05,
        "cagr_net_income": 0.05,
        "divida_pl": 1.0,
        "liquidez_corrente": 1.5,
    }

    def __init__(self) -> None:
        self.reasons: List[str] = []

    def _compare(
        self,
        value: Optional[float],
        ref_key: str,
        label: str,
        higher_is_better: bool = True,
        invert: bool = False,
    ) -> Optional[str]:
        if value is None:
            return None
        ref = self.REFERENCE.get(ref_key)
        if ref is None:
            return None

        if invert:
            value_cmp = -value
            ref_cmp = -ref
        else:
            value_cmp = value
            ref_cmp = ref

        diff_pct = ((value_cmp - ref_cmp) / abs(ref_cmp)) * 100 if ref_cmp != 0 else 0

        if higher_is_better:
            if diff_pct > 20:
                return f"{label} acima da média ({value:.2f} vs ref {ref:.2f})"
            elif diff_pct < -20:
                return f"{label} abaixo da média ({value:.2f} vs ref {ref:.2f})"
            else:
                return f"{label} na média ({value:.2f})"
        else:
            if diff_pct < -20:
                return f"{label} abaixo da média ({value:.2f} vs ref {ref:.2f})"
            elif diff_pct > 20:
                return f"{label} acima da média ({value:.2f} vs ref {ref:.2f})"
            else:
                return f"{label} na média ({value:.2f})"

    def explain_asset(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for a single asset.

        Parameters
        ----------
        row : dict
            Asset data with fundamental metrics and factor scores.
            Expected keys: roe, roic, margem_liquida, pl, pvp, ev_ebit, dy,
            cagr_revenue, cagr_net_income, divida_pl, liquidez_corrente,
            quality, valuation, dividends, growth, safety, total_score.

        Returns
        -------
        dict
            Ticker, score, and list of scored reasons.
        """
        ticker = row.get("ticker", "Unknown")
        reasons: List[str] = []
        positive: List[str] = []
        negative: List[str] = []

        # Quality
        roe = row.get("roe")
        if roe is not None:
            r = self._compare(roe, "roe", "ROE")
            if r:
                reasons.append(r)
                if roe > self.REFERENCE["roe"]:
                    positive.append(r)
                else:
                    negative.append(r)

        margin = row.get("margem_liquida")
        if margin is not None:
            r = self._compare(margin, "margem_liquida", "Margem Líquida")
            if r:
                reasons.append(r)
                if margin > self.REFERENCE["margem_liquida"]:
                    positive.append(r)
                else:
                    negative.append(r)

        # Valuation
        pl = row.get("pl")
        if pl is not None and pl > 0:
            r = self._compare(pl, "pl", "P/L", higher_is_better=False)
            if r:
                reasons.append(r)
                if pl < self.REFERENCE["pl"]:
                    positive.append(r)
                else:
                    negative.append(r)

        pvp = row.get("pvp")
        if pvp is not None and pvp > 0:
            r = self._compare(pvp, "pvp", "P/VP", higher_is_better=False)
            if r:
                reasons.append(r)
                if pvp < self.REFERENCE["pvp"]:
                    positive.append(r)
                else:
                    negative.append(r)

        # Dividends
        dy = row.get("dy")
        if dy is not None:
            r = self._compare(dy, "dy", "Dividend Yield")
            if r:
                reasons.append(r)
                if dy > self.REFERENCE["dy"]:
                    positive.append(r)
                else:
                    negative.append(r)

        # Growth
        cagr_ni = row.get("cagr_net_income")
        if cagr_ni is not None:
            r = self._compare(cagr_ni, "cagr_net_income", "Crescimento Lucro")
            if r:
                reasons.append(r)
                if cagr_ni > self.REFERENCE["cagr_net_income"]:
                    positive.append(r)
                else:
                    negative.append(r)

        # Safety
        debt = row.get("divida_pl")
        if debt is not None:
            r = self._compare(debt, "divida_pl", "Endividamento", higher_is_better=False)
            if r:
                reasons.append(r)
                if debt < self.REFERENCE["divida_pl"]:
                    positive.append(r)
                else:
                    negative.append(r)

        liquidity = row.get("liquidez_corrente")
        if liquidity is not None:
            r = self._compare(liquidity, "liquidez_corrente", "Liquidez Corrente")
            if r:
                reasons.append(r)
                if liquidity > self.REFERENCE["liquidez_corrente"]:
                    positive.append(r)
                else:
                    negative.append(r)

        score = row.get("total_score", row.get("score_total", 50))
        motivos = [r for r in reasons if r]

        return {
            "ticker": ticker,
            "score": score,
            "motivos": motivos,
            "pontos_fortes": positive[:5],
            "pontos_fracos": negative[:5],
        }

    def batch_explain(
        self,
        assets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a list of assets.

        Parameters
        ----------
        assets : list of dict
            Each dict must contain fundamental metrics and scores.

        Returns
        -------
        list of dict
            Explanations sorted by score descending.
        """
        results = [self.explain_asset(a) for a in assets]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
