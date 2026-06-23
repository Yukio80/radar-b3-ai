import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """
    Download and manage benchmark indices for Brazilian market.

    Available benchmarks:
    - IBOV: Ibovespa (^BVSP)
    - IDIV: Dividendos Index (^IDIV)
    - CDI:  (approximated via Selic-related ETF or constant rate)
    """

    BENCHMARK_TICKERS = {
        "IBOV": "^BVSP",
        "IDIV": "^IDIV",
    }

    def __init__(
        self,
        cdi_rate: float = 0.1325,
        start_date: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        cdi_rate : float, default=0.1325
            Annual CDI rate to use as fallback (13.25%).
        start_date : str, optional
            Start date for benchmark data (YYYY-MM-DD).
        """
        self.cdi_rate = cdi_rate
        self.start_date = start_date or (datetime.now() - timedelta(days=1825)).strftime(
            "%Y-%m-%d"
        )
        self._data: Dict[str, pd.Series] = {}

    def download(self, name: str) -> pd.Series:
        """
        Download daily returns for a single benchmark.

        Parameters
        ----------
        name : str
            Benchmark name ('IBOV', 'IDIV', 'CDI').

        Returns
        -------
        pd.Series
            Daily returns indexed by date.

        Raises
        ------
        ValueError
            If benchmark name is unknown.
        """
        name = name.upper()

        if name == "CDI":
            return self._get_cdi_series()

        ticker = self.BENCHMARK_TICKERS.get(name)
        if not ticker:
            raise ValueError(f"Unknown benchmark: {name}. Use: {list(self.BENCHMARK_TICKERS.keys()) + ['CDI']}")

        logger.info("Downloading %s (%s) from %s", name, ticker, self.start_date)
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                logger.warning("No data for %s, returning empty series", name)
                return pd.Series(dtype=float)

            close = data["Close"]
            if isinstance(close.columns, pd.MultiIndex):
                close = close.xs(ticker, level=1, axis=1)

            close = close.squeeze()
            returns = close.pct_change().dropna()
            returns.name = name
            self._data[name] = returns
            return returns
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            return pd.Series(dtype=float)

    def download_all(self) -> Dict[str, pd.Series]:
        """
        Download all benchmarks.

        Returns
        -------
        dict
            Name -> daily returns Series.
        """
        for name in list(self.BENCHMARK_TICKERS.keys()) + ["CDI"]:
            self.download(name)
        return self._data

    def _get_cdi_series(self) -> pd.Series:
        """
        Generate synthetic CDI daily returns.

        CDI is approximated by the annual CDI rate converted to daily
        business-day returns, correlated with the Selic rate.

        Returns
        -------
        pd.Series
            Daily CDI returns.
        """
        if "CDI" in self._data:
            return self._data["CDI"]

        daily_cdi = (1 + self.cdi_rate) ** (1 / 252) - 1
        end = datetime.now()
        start = datetime.strptime(self.start_date, "%Y-%m-%d") if isinstance(self.start_date, str) else datetime.now() - timedelta(days=1825)
        dates = pd.bdate_range(start=start, end=end)
        cdi_series = pd.Series(daily_cdi, index=dates, name="CDI")
        self._data["CDI"] = cdi_series
        return cdi_series

    def get_benchmark_returns(
        self,
        benchmark_name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """
        Get returns for a benchmark filtered by date range.

        Parameters
        ----------
        benchmark_name : str
            Name of the benchmark.
        start : str, optional
            Start date filter (YYYY-MM-DD).
        end : str, optional
            End date filter (YYYY-MM-DD).

        Returns
        -------
        pd.Series
            Filtered returns.
        """
        name = benchmark_name.upper()
        if name not in self._data:
            self.download(name)

        series = self._data.get(name, pd.Series(dtype=float))
        if series.empty:
            return series

        if start:
            series = series[series.index >= start]
        if end:
            series = series[series.index <= end]
        return series

    def cumulative_comparison(
        self,
        portfolio_returns: pd.Series,
        benchmark_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a cumulative return comparison DataFrame.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Portfolio daily returns.
        benchmark_names : list of str, optional
            Benchmarks to compare against. Defaults to all.

        Returns
        -------
        pd.DataFrame
            Cumulative returns for portfolio and benchmarks.
        """
        if benchmark_names is None:
            benchmark_names = list(self.BENCHMARK_TICKERS.keys()) + ["CDI"]

        series_dict: Dict[str, pd.Series] = {"Portfolio": portfolio_returns}

        for bm in benchmark_names:
            bm_returns = self.get_benchmark_returns(bm)
            if not bm_returns.empty:
                series_dict[bm] = bm_returns

        combined = pd.DataFrame(series_dict)
        combined = combined.dropna(how="all")
        cumulative = (1 + combined).cumprod()

        return cumulative
