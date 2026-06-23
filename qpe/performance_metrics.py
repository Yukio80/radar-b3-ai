import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class PerformanceMetrics:
    """
    Calculate portfolio performance and risk metrics.

    Provides a comprehensive suite of metrics for evaluating strategy
    performance relative to benchmarks and risk-free rate.

    Metrics:
    - Cumulative / annualized return
    - Volatility (annualized)
    - Sharpe, Sortino, Calmar ratios
    - Maximum drawdown
    - Alpha, Beta (CAPM)
    - Tracking error
    - Information ratio
    """

    def __init__(self, risk_free_rate: float = 0.1325) -> None:
        """
        Parameters
        ----------
        risk_free_rate : float, default=0.1325
            Annual risk-free rate (Selic ~13.25% ao ano).
        """
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _to_series(returns: pd.Series) -> pd.Series:
        return returns.dropna().astype(float)

    def cumulative_return(self, returns: pd.Series) -> float:
        """
        Calculate cumulative total return.

        Parameters
        ----------
        returns : pd.Series
            Period returns (daily, monthly, etc.).

        Returns
        -------
        float
            Cumulative return as a decimal.
        """
        r = self._to_series(returns)
        if r.empty:
            return 0.0
        return float((1 + r).prod() - 1)

    def annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.

        Parameters
        ----------
        returns : pd.Series
            Period returns.
        periods_per_year : int, default=252
            Number of periods per year (252 for daily, 12 for monthly).

        Returns
        -------
        float
            Annualized return as a decimal.
        """
        r = self._to_series(returns)
        if r.empty:
            return 0.0
        n = len(r)
        if n == 0:
            return 0.0
        total_ret = (1 + r).prod()
        years = n / periods_per_year
        if years <= 0:
            return 0.0
        return float(total_ret ** (1.0 / years) - 1)

    def annualized_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).

        Parameters
        ----------
        returns : pd.Series
            Period returns.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        float
            Annualized volatility as a decimal.
        """
        r = self._to_series(returns)
        if len(r) < 2:
            return 0.0
        return float(r.std() * np.sqrt(periods_per_year))

    def sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (Return_p - Rf) / Vol_p

        Parameters
        ----------
        returns : pd.Series
            Period returns.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        float
            Sharpe ratio.
        """
        r = self._to_series(returns)
        if len(r) < 2:
            return 0.0
        ann_ret = self.annualized_return(r, periods_per_year)
        ann_vol = self.annualized_volatility(r, periods_per_year)
        if ann_vol == 0:
            return 0.0
        return float((ann_ret - self.risk_free_rate) / ann_vol)

    def sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio using downside deviation.

        Sortino = (Return_p - Rf) / DownsideDev_p

        Parameters
        ----------
        returns : pd.Series
            Period returns.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        float
            Sortino ratio.
        """
        r = self._to_series(returns)
        if len(r) < 2:
            return 0.0
        ann_ret = self.annualized_return(r, periods_per_year)
        downside = r[r < 0].std()
        if len(r[r < 0]) == 0 or pd.isna(downside) or downside == 0:
            return 0.0 if ann_ret <= self.risk_free_rate else 10.0
        downside_ann = downside * np.sqrt(periods_per_year)
        if downside_ann == 0:
            return 0.0
        return float((ann_ret - self.risk_free_rate) / downside_ann)

    def max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from peak to trough.

        Parameters
        ----------
        returns : pd.Series
            Period returns.

        Returns
        -------
        float
            Max drawdown as a decimal (positive value).
        """
        r = self._to_series(returns)
        if r.empty:
            return 0.0
        cumulative = (1 + r).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(abs(drawdown.min()))

    def calmar_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Parameters
        ----------
        returns : pd.Series
            Period returns.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        float
            Calmar ratio.
        """
        r = self._to_series(returns)
        if r.empty:
            return 0.0
        ann_ret = self.annualized_return(r, periods_per_year)
        mdd = self.max_drawdown(r)
        if mdd == 0:
            return 0.0
        return float(ann_ret / mdd)

    def alpha_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """
        Calculate Alpha and Beta relative to a benchmark.

        Uses OLS regression: R_p - Rf = alpha + beta * (R_m - Rf)

        Parameters
        ----------
        returns : pd.Series
            Portfolio period returns.
        benchmark_returns : pd.Series
            Benchmark period returns.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        dict
            Alpha (annualized), Beta, R-squared.
        """
        r_p = self._to_series(returns)
        r_m = self._to_series(benchmark_returns)

        aligned = pd.concat({"portfolio": r_p, "benchmark": r_m}, axis=1).dropna()
        if len(aligned) < 5:
            return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0}

        p_vals = aligned["portfolio"].values
        m_vals = aligned["benchmark"].values

        rf_period = (1 + self.risk_free_rate) ** (1.0 / periods_per_year) - 1
        excess_p = p_vals - rf_period
        excess_m = m_vals - rf_period

        cov = np.cov(excess_m, excess_p)
        if cov[0, 0] == 0:
            beta = 1.0
        else:
            beta = cov[0, 1] / cov[0, 0]

        alpha_period = np.mean(excess_p) - beta * np.mean(excess_m)
        alpha_ann = (1 + alpha_period) ** periods_per_year - 1

        correlation = np.corrcoef(excess_m, excess_p)[0, 1]
        r_squared = correlation ** 2

        return {
            "alpha": round(alpha_ann, 4),
            "beta": round(beta, 4),
            "r_squared": round(r_squared, 4),
        }

    def tracking_error(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate tracking error (active risk).

        TE = std(R_p - R_b)

        Parameters
        ----------
        returns : pd.Series
            Portfolio period returns.
        benchmark_returns : pd.Series
            Benchmark period returns.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        float
            Annualized tracking error.
        """
        r_p = self._to_series(returns)
        r_m = self._to_series(benchmark_returns)

        aligned = pd.concat({"portfolio": r_p, "benchmark": r_m}, axis=1).dropna()
        if len(aligned) < 2:
            return 0.0

        diff = aligned["portfolio"] - aligned["benchmark"]
        return float(diff.std() * np.sqrt(periods_per_year))

    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Information Ratio.

        IR = (Return_p - Return_b) / TrackingError

        Parameters
        ----------
        returns : pd.Series
            Portfolio period returns.
        benchmark_returns : pd.Series
            Benchmark period returns.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        float
            Information ratio.
        """
        r_p = self._to_series(returns)
        r_m = self._to_series(benchmark_returns)

        aligned = pd.concat({"portfolio": r_p, "benchmark": r_m}, axis=1).dropna()
        if len(aligned) < 5:
            return 0.0

        ann_p = self.annualized_return(aligned["portfolio"], periods_per_year)
        ann_m = self.annualized_return(aligned["benchmark"], periods_per_year)
        te = self.tracking_error(aligned["portfolio"], aligned["benchmark"], periods_per_year)

        if te == 0:
            return 0.0
        return float((ann_p - ann_m) / te)

    def all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """
        Compute all available performance metrics in one call.

        Parameters
        ----------
        returns : pd.Series
            Portfolio period returns.
        benchmark_returns : pd.Series, optional
            Benchmark period returns for alpha/beta/IR/TE.
        periods_per_year : int, default=252
            Number of periods per year.

        Returns
        -------
        dict
            All computed metrics.
        """
        r = self._to_series(returns)
        metrics: Dict[str, Any] = {
            "retorno_acumulado": round(self.cumulative_return(r), 4),
            "retorno_anualizado": round(self.annualized_return(r, periods_per_year), 4),
            "volatilidade_anualizada": round(self.annualized_volatility(r, periods_per_year), 4),
            "sharpe_ratio": round(self.sharpe_ratio(r, periods_per_year), 4),
            "sortino_ratio": round(self.sortino_ratio(r, periods_per_year), 4),
            "calmar_ratio": round(self.calmar_ratio(r, periods_per_year), 4),
            "max_drawdown": round(self.max_drawdown(r), 4),
        }

        if benchmark_returns is not None:
            bm = self._to_series(benchmark_returns)
            ab = self.alpha_beta(r, bm, periods_per_year)
            metrics["alpha"] = ab["alpha"]
            metrics["beta"] = ab["beta"]
            metrics["r_squared"] = ab["r_squared"]
            metrics["tracking_error"] = round(
                self.tracking_error(r, bm, periods_per_year), 4
            )
            metrics["information_ratio"] = round(
                self.information_ratio(r, bm, periods_per_year), 4
            )

        return metrics
