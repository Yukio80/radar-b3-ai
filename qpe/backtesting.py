import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import yfinance as yf

from qpe.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Historical backtesting engine with configurable rebalance frequency.

    Simulates portfolio performance by rebalancing at specified intervals
    using a scoring function, applying weight constraints, and tracking
    forward returns. Designed to avoid look-ahead bias.
    """

    FREQ_MAP = {
        "mensal": 21,
        "trimestral": 63,
        "semestral": 126,
        "anual": 252,
    }

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        initial_capital: float = 100000.0,
        rebalance_frequency: str = "trimestral",
        top_n: int = 10,
        peso_min: float = 0.02,
        peso_max: float = 0.10,
        transaction_cost: float = 0.001,
    ) -> None:
        """
        Parameters
        ----------
        tickers : list of str
            Universe of tickers to consider.
        start_date : str
            Backtest start date (YYYY-MM-DD).
        end_date : str, optional
            Backtest end date (YYYY-MM-DD). Defaults to today.
        initial_capital : float, default=100000
            Starting capital in R$.
        rebalance_frequency : str, default="trimestral"
            Rebalance frequency: 'mensal', 'trimestral', 'semestral', 'anual'.
        top_n : int, default=10
            Number of top-scoring assets to select.
        peso_min : float, default=0.02
            Minimum weight per asset (2%).
        peso_max : float, default=0.10
            Maximum weight per asset (10%).
        transaction_cost : float, default=0.001
            Transaction cost per trade (0.1%).
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.top_n = top_n
        self.peso_min = peso_min
        self.peso_max = peso_max
        self.transaction_cost = transaction_cost

        self.optimizer = PortfolioOptimizer(peso_min=peso_min, peso_max=peso_max)
        self._prices: Optional[pd.DataFrame] = None
        self._dividends: Optional[pd.DataFrame] = None
        self._rebalance_dates: List[str] = []

    def download_data(self) -> None:
        """
        Download historical prices and dividends for all tickers.
        """
        logger.info(
            "Downloading data for %d tickers from %s to %s",
            len(self.tickers),
            self.start_date,
            self.end_date,
        )

        padded_start = (
            datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=30)
        ).strftime("%Y-%m-%d")

        try:
            data = yf.download(
                self.tickers,
                start=padded_start,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                logger.error("No price data downloaded")
                self._prices = pd.DataFrame()
                return

            if "Close" in data.columns:
                close = data["Close"]
                if isinstance(close.columns, pd.MultiIndex):
                    close = close.xs("Close", level=0, axis=1)
            else:
                close = data

            if isinstance(close, pd.Series):
                close = close.to_frame()

            self._prices = close

            div_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                actions=True,
            )
            if "Dividends" in div_data.columns:
                div = div_data["Dividends"]
                if isinstance(div.columns, pd.MultiIndex):
                    div = div.xs("Dividends", level=0, axis=1)
                self._dividends = div
            else:
                self._dividends = pd.DataFrame(0.0, index=self._prices.index, columns=self._prices.columns)

        except Exception as e:
            logger.error("Data download failed: %s", e)
            self._prices = pd.DataFrame()

    def _get_rebalance_dates(self) -> List[str]:
        """
        Generate rebalance dates based on frequency.

        Returns
        -------
        list of str
            Rebalance dates in 'YYYY-MM-DD' format.
        """
        if self._prices is None or self._prices.empty:
            return []

        all_dates = sorted(self._prices.index)
        if not all_dates:
            return []

        freq_days = self.FREQ_MAP.get(self.rebalance_frequency, 63)

        dates: List[str] = [str(all_dates[0].date())]
        last = all_dates[0]

        for d in all_dates[1:]:
            if (d - last).days >= freq_days:
                dates.append(str(d.date()))
                last = d

        return dates

    def _get_prices_at(
        self, date: str, window: int = 5
    ) -> pd.Series:
        """
        Get the most recent available prices up to a given date.

        Parameters
        ----------
        date : str
            Target date.
        window : int, default=5
            Max days to look back.

        Returns
        -------
        pd.Series
            Prices for each ticker (last available).
        """
        if self._prices is None or self._prices.empty:
            return pd.Series(dtype=float)

        date_dt = pd.Timestamp(date)
        mask = self._prices.index <= date_dt
        if not mask.any():
            return pd.Series(index=self._prices.columns, dtype=float)

        available = self._prices.loc[mask]
        if available.empty:
            return pd.Series(index=self._prices.columns, dtype=float)

        return available.iloc[-1]

    def _compute_score_proxy(
        self,
        date: str,
        scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute scores at a given date.

        Uses provided scores if available (snapshot mode), otherwise
        computes trailing dividend yield as a simple proxy.

        Parameters
        ----------
        date : str
            Rebalance date.
        scores : dict, optional
            Pre-computed scores (ticker -> total_score).

        Returns
        -------
        dict
            Ticker -> score.
        """
        if scores is not None:
            result: Dict[str, float] = {}
            for t in self.tickers:
                lookup = t.replace(".SA", "")
                val = scores.get(t) or scores.get(lookup)
                result[t] = val if val is not None else 50.0
            return result

        if self._prices is None or self._prices.empty:
            return {t: 50.0 for t in self.tickers}

        price_series = self._get_prices_at(date)
        result: Dict[str, float] = {}

        for ticker in self.tickers:
            px = price_series.get(ticker)
            if px is None or px == 0:
                result[ticker] = 50.0
                continue

            trailing_return = 0.0
            date_dt = pd.Timestamp(date)
            lookback = date_dt - timedelta(days=252)
            prices_before = self._prices[ticker] if ticker in self._prices.columns else pd.Series(dtype=float)
            prices_before = prices_before[prices_before.index <= date_dt]
            prices_before = prices_before[prices_before.index >= lookback]

            if len(prices_before) > 20:
                trailing_return = (prices_before.iloc[-1] / prices_before.iloc[0]) - 1

            score = 50.0 + max(-30, min(30, trailing_return * 100))
            result[ticker] = round(max(1, min(100, score)), 1)

        return result

    def run(
        self,
        scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the backtest simulation using a returns-based approach.

        For each rebalance date:
        1. Compute or retrieve scores
        2. Select top N assets
        3. Optimize weights
        4. Calculate daily portfolio returns until next rebalance

        Parameters
        ----------
        scores : dict, optional
            Pre-computed scores (ticker -> total_score). If None,
            uses trailing momentum as a proxy.

        Returns
        -------
        dict
            Full backtest results with equity curve, returns, and trades.
        """
        if self._prices is None:
            self.download_data()

        if self._prices is None or self._prices.empty:
            return {
                "error": "No price data available",
                "equity_curve": pd.Series(dtype=float),
                "returns": pd.Series(dtype=float),
                "trades": [],
                "rebalance_dates": [],
            }

        prices = self._prices.copy()
        returns_df = prices.pct_change().dropna(how="all")
        if returns_df.empty:
            return {
                "error": "No returns generated from price data",
                "equity_curve": pd.Series(dtype=float),
                "returns": pd.Series(dtype=float),
                "trades": [],
                "rebalance_dates": [],
            }

        self._rebalance_dates = self._get_rebalance_dates()
        if not self._rebalance_dates:
            return {
                "error": "No rebalance dates generated",
                "equity_curve": pd.Series(dtype=float),
                "returns": pd.Series(dtype=float),
                "trades": [],
                "rebalance_dates": [],
            }

        daily_returns_list: List[float] = []
        date_index: List[pd.Timestamp] = []
        trades: List[Dict[str, Any]] = []
        capital = self.initial_capital

        current_weights: Dict[str, float] = {}
        first = True

        for i, rebal_date_str in enumerate(self._rebalance_dates):
            rebal_dt = pd.Timestamp(rebal_date_str)

            score_map = self._compute_score_proxy(rebal_date_str, scores=scores)

            sorted_tickers = sorted(
                score_map.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:self.top_n]

            selected_tickers = [t for t, s in sorted_tickers if s > 0]
            if not selected_tickers:
                continue

            selected_scores = [s for t, s in sorted_tickers if s > 0]

            opt_result = self.optimizer.optimize(selected_scores, selected_tickers)
            new_weights: Dict[str, float] = {}
            for _, row in opt_result.iterrows():
                new_weights[row["ticker"]] = row["weight_pct"] / 100.0

            if first:
                current_weights = new_weights
                first = False

            next_rebal_str = (
                self._rebalance_dates[i + 1]
                if i + 1 < len(self._rebalance_dates)
                else self.end_date
            )

            period_mask = (returns_df.index > rebal_dt) & (
                returns_df.index <= pd.Timestamp(next_rebal_str)
            )
            period_dates = returns_df.index[period_mask]

            if not period_dates.empty:
                period_returns = returns_df.loc[period_dates]

                for date in period_dates:
                    day_ret = period_returns.loc[date]
                    weighted_ret = 0.0
                    for t, w in current_weights.items():
                        asset_ret = day_ret.get(t) if hasattr(day_ret, "get") else None
                        if asset_ret is not None and not pd.isna(asset_ret):
                            weighted_ret += w * asset_ret
                    daily_returns_list.append(weighted_ret)
                    date_index.append(date)
                    capital *= (1 + weighted_ret)

            trades.append({
                "data": rebal_date_str,
                "tickers": selected_tickers,
                "weights": {t: round(w * 100, 2) for t, w in new_weights.items()},
                "capital": round(capital, 2) if not daily_returns_list else round(
                    self.initial_capital * (1 + pd.Series(daily_returns_list)).prod(), 2
                ),
            })

            current_weights = new_weights

        if daily_returns_list:
            daily_returns = pd.Series(daily_returns_list, index=date_index)
            equity = self.initial_capital * (1 + daily_returns).cumprod()
        else:
            daily_returns = pd.Series(dtype=float)
            equity = pd.Series(dtype=float)

        return {
            "equity_curve": equity,
            "returns": daily_returns,
            "capital_final": round(float(equity.iloc[-1]), 2) if not equity.empty else self.initial_capital,
            "capital_inicial": self.initial_capital,
            "retorno_total": round((float(equity.iloc[-1]) / self.initial_capital - 1) * 100, 2) if not equity.empty else 0,
            "trades": trades,
            "rebalance_dates": self._rebalance_dates,
            "frequencia": self.rebalance_frequency,
            "qtd_rebalances": len(trades),
        }

    def run_multiple_frequencies(
        self,
        scores: Optional[Dict[str, float]] = None,
        frequencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest at multiple frequencies for comparison.

        Parameters
        ----------
        scores : dict, optional
            Pre-computed scores.
        frequencies : list of str, optional
            Frequencies to test. Defaults to all.

        Returns
        -------
        dict
            Frequency -> backtest results.
        """
        if frequencies is None:
            frequencies = list(self.FREQ_MAP.keys())

        results: Dict[str, Any] = {}
        for freq in frequencies:
            engine = BacktestEngine(
                tickers=self.tickers,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                rebalance_frequency=freq,
                top_n=self.top_n,
                peso_min=self.peso_min,
                peso_max=self.peso_max,
                transaction_cost=self.transaction_cost,
            )
            engine._prices = self._prices
            results[freq] = engine.run(scores=scores)

        return results
