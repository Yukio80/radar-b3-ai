import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class GrowthFactor:
    """
    Calculate growth metrics (CAGR) for revenue, profit, and dividends
    using historical financial data from yfinance.

    Parameters
    ----------
    years : int, default=5
        Number of years for CAGR calculation.
    """

    def __init__(self, years: int = 5) -> None:
        self.years = years

    def calc_cagr(self, values: List[float]) -> Optional[float]:
        """
        Calculate Compound Annual Growth Rate.

        CAGR = (final / initial) ** (1 / years) - 1

        Parameters
        ----------
        values : list of float
            Ordered time series (oldest first, newest last).

        Returns
        -------
        float or None
            CAGR as decimal, or None if insufficient data.
        """
        clean = [v for v in values if v is not None and v > 0]
        if len(clean) < 2:
            return None
        n = min(len(clean) - 1, self.years)
        if n < 1:
            return None
        initial = clean[0]
        final = clean[-1]
        if initial <= 0 or final <= 0:
            return None
        return (final / initial) ** (1.0 / n) - 1.0

    def extract_financials(self, ticker: str) -> Dict[str, List[float]]:
        """
        Extract annual financial data from yfinance.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol (without .SA suffix).

        Returns
        -------
        dict
            Keys: 'revenue', 'net_income', 'dividends'.
            Values: ordered lists (oldest first).
        """
        import yfinance as yf  # type: ignore

        t = yf.Ticker(ticker + ".SA")
        result: Dict[str, List[float]] = {
            "revenue": [],
            "net_income": [],
            "dividends": [],
        }

        try:
            fin = t.financials
        except Exception:
            fin = pd.DataFrame()

        if not fin.empty and "Total Revenue" in fin.index:
            rev = fin.loc["Total Revenue"].dropna().values
            result["revenue"] = [float(v) for v in rev]

        if not fin.empty and "Net Income" in fin.index:
            ni = fin.loc["Net Income"].dropna().values
            result["net_income"] = [float(v) for v in ni]

        try:
            div = t.dividends
        except Exception:
            div = pd.Series(dtype=float)

        if not div.empty:
            annual_div = div.groupby(div.index.year).sum()
            result["dividends"] = annual_div.sort_index().values.tolist()

        return result

    def compute(
        self,
        ticker: str,
        financials: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Compute CAGR for revenue, net income, and dividends.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        financials : dict, optional
            Pre-extracted financial data. If None, fetches from yfinance.

        Returns
        -------
        dict
            Keys: cagr_revenue, cagr_net_income, cagr_dividends.
        """
        if financials is None:
            financials = self.extract_financials(ticker)

        cagr_rev = self.calc_cagr(financials.get("revenue", []))
        cagr_ni = self.calc_cagr(financials.get("net_income", []))
        cagr_div = self.calc_cagr(financials.get("dividends", []))

        return {
            "cagr_revenue": round(cagr_rev, 4) if cagr_rev is not None else None,
            "cagr_net_income": round(cagr_ni, 4) if cagr_ni is not None else None,
            "cagr_dividends": round(cagr_div, 4) if cagr_div is not None else None,
        }

    def normalize_cagr(
        self,
        values: List[Optional[float]],
    ) -> List[float]:
        """
        Normalize CAGR values to a 0-100 scale.

        Uses percentile rank with clipping at -0.5 and +0.5 CAGR.

        Parameters
        ----------
        values : list of float or None
            CAGR values.

        Returns
        -------
        list of float
            Scores from 0 to 100.
        """
        arr = np.array([v if v is not None else 0.0 for v in values], dtype=float)
        clipped = np.clip(arr, -0.5, 0.5)
        normalised = (clipped + 0.5) / 1.0 * 100.0
        return [round(float(v), 1) for v in normalised]
