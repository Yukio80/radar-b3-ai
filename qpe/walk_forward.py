import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qpe.backtesting import BacktestEngine

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for out-of-sample robustness testing.

    Splits historical data into rolling training/test windows:
      Train -> Test
            Train -> Test
                  Train -> Test

    Aggregates results across all test windows to assess
    out-of-sample performance.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        train_years: int = 2,
        test_months: int = 6,
        rebalance_frequency: str = "trimestral",
        top_n: int = 10,
        peso_min: float = 0.02,
        peso_max: float = 0.10,
    ) -> None:
        """
        Parameters
        ----------
        tickers : list of str
            Universe of tickers.
        start_date : str
            Start date (YYYY-MM-DD).
        end_date : str, optional
            End date (YYYY-MM-DD).
        train_years : int, default=2
            Length of each training window in years.
        test_months : int, default=6
            Length of each test window in months.
        rebalance_frequency : str, default="trimestral"
            Rebalance frequency within each period.
        top_n : int, default=10
            Number of top assets to select.
        peso_min : float, default=0.02
            Minimum weight.
        peso_max : float, default=0.10
            Maximum weight.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.train_years = train_years
        self.test_months = test_months
        self.rebalance_frequency = rebalance_frequency
        self.top_n = top_n
        self.peso_min = peso_min
        self.peso_max = peso_max

    def _generate_windows(self) -> List[Tuple[str, str, str, str]]:
        """
        Generate contiguous train/test windows.

        Returns
        -------
        list of tuple
            Each tuple: (train_start, train_end, test_start, test_end).
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        train_days = self.train_years * 365
        test_days = self.test_months * 30

        windows: List[Tuple[str, str, str, str]] = []
        current = start_dt

        while True:
            train_end = current + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)

            if test_end > end_dt:
                test_end = end_dt

            if test_start >= end_dt:
                break

            if train_end <= end_dt and test_start < test_end:
                windows.append((
                    current.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                    test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                ))

            current = test_start
            if test_end >= end_dt:
                break

        return windows

    def validate(self) -> Dict[str, Any]:
        """
        Run walk-forward validation across all windows.

        Returns
        -------
        dict
            Aggregated results with per-window breakdown.
        """
        windows = self._generate_windows()
        logger.info("Generated %d walk-forward windows", len(windows))

        if not windows:
            return {
                "error": "No windows generated. Check date range.",
                "windows": [],
                "resultados_consolidados": {},
            }

        window_results: List[Dict[str, Any]] = []

        all_test_returns: List[pd.Series] = []
        all_train_returns: List[pd.Series] = []

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            logger.info(
                "Window %d/%d: Train %s->%s | Test %s->%s",
                i + 1, len(windows), tr_s, tr_e, te_s, te_e,
            )

            train_engine = BacktestEngine(
                tickers=self.tickers,
                start_date=tr_s,
                end_date=tr_e,
                rebalance_frequency=self.rebalance_frequency,
                top_n=self.top_n,
                peso_min=self.peso_min,
                peso_max=self.peso_max,
            )
            train_engine.download_data()
            train_result = train_engine.run()

            train_returns = train_result.get("returns", pd.Series(dtype=float))

            train_scores: Dict[str, float] = {}
            if train_engine._prices is not None and not train_engine._prices.empty:
                last_prices = train_engine._prices.iloc[-1]
                for t in self.tickers:
                    px = last_prices.get(t) if hasattr(last_prices, 'get') else None
                    if px is not None and px > 0:
                        momentum = 0.0
                        window_prices = train_engine._prices[t]
                        if len(window_prices) > 60:
                            momentum = (window_prices.iloc[-1] / window_prices.iloc[-60]) - 1
                        train_scores[t] = round(50.0 + max(-30, min(30, momentum * 100)), 1)
                    else:
                        train_scores[t] = 50.0

            test_engine = BacktestEngine(
                tickers=self.tickers,
                start_date=te_s,
                end_date=te_e,
                rebalance_frequency=self.rebalance_frequency,
                top_n=self.top_n,
                peso_min=self.peso_min,
                peso_max=self.peso_max,
            )
            test_engine._prices = train_engine._prices
            test_result = test_engine.run(scores=train_scores)

            test_returns = test_result.get("returns", pd.Series(dtype=float))

            all_train_returns.append(train_returns)
            all_test_returns.append(test_returns)

            window_results.append({
                "janela": i + 1,
                "treino": {"inicio": tr_s, "fim": tr_e},
                "teste": {"inicio": te_s, "fim": te_e},
                "retorno_treino": train_result.get("retorno_total", 0),
                "retorno_teste": test_result.get("retorno_total", 0),
                "capital_final_teste": test_result.get("capital_final", 0),
                "num_trades": len(test_result.get("trades", [])),
            })

        combined_test = pd.concat(all_test_returns) if all_test_returns else pd.Series(dtype=float)
        combined_train = pd.concat(all_train_returns) if all_train_returns else pd.Series(dtype=float)

        test_retornos = [w["retorno_teste"] for w in window_results]
        train_retornos = [w["retorno_treino"] for w in window_results]

        return {
            "janelas": window_results,
            "total_janelas": len(window_results),
            "resultados_consolidados": {
                "retorno_medio_treino": round(float(np.mean(train_retornos)), 2) if train_retornos else 0,
                "retorno_medio_teste": round(float(np.mean(test_retornos)), 2) if test_retornos else 0,
                "mediana_retorno_teste": round(float(np.median(test_retornos)), 2) if test_retornos else 0,
                "std_retorno_teste": round(float(np.std(test_retornos)), 2) if len(test_retornos) > 1 else 0,
                "min_retorno_teste": round(float(min(test_retornos)), 2) if test_retornos else 0,
                "max_retorno_teste": round(float(max(test_retornos)), 2) if test_retornos else 0,
                "janelas_positivas": sum(1 for r in test_retornos if r > 0),
                "janelas_negativas": sum(1 for r in test_retornos if r < 0),
                "taxa_acerto": round(sum(1 for r in test_retornos if r > 0) / len(test_retornos) * 100, 1) if test_retornos else 0,
            },
            "returns_teste": combined_test,
            "returns_treino": combined_train,
        }
