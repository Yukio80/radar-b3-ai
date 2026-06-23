import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for portfolio risk analysis.

    Generates thousands of possible return paths based on historical
    asset parameters and calculates risk metrics:
    - VaR (Value at Risk) at 95% and 99%
    - CVaR (Conditional VaR)
    - Probability of loss
    - Probability of outperforming benchmarks
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        horizon_days: int = 252,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Parameters
        ----------
        num_simulations : int, default=10000
            Number of Monte Carlo paths to simulate.
        horizon_days : int, default=252
            Simulation horizon in business days (1 year).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.num_simulations = num_simulations
        self.horizon_days = horizon_days
        self.seed = seed

    def simulate_gbm(
        self,
        annual_return: float,
        annual_volatility: float,
        initial_value: float = 1.0,
    ) -> np.ndarray:
        """
        Simulate paths using Geometric Brownian Motion.

        Parameters
        ----------
        annual_return : float
            Annualized expected return (decimal).
        annual_volatility : float
            Annualized volatility (decimal).
        initial_value : float, default=1.0
            Starting value.

        Returns
        -------
        np.ndarray
            Shape (num_simulations, horizon_days + 1) with final values.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        dt = 1.0 / 252
        mu = annual_return
        sigma = annual_volatility

        z = np.random.normal(0, 1, (self.num_simulations, self.horizon_days))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        log_returns = drift + diffusion
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        paths = initial_value * np.exp(cumulative_log_returns)

        paths = np.column_stack([np.full(self.num_simulations, initial_value), paths])
        return paths

    def simulate_from_returns(
        self,
        historical_returns: pd.Series,
        initial_value: float = 1.0,
    ) -> np.ndarray:
        """
        Simulate paths by resampling historical returns (bootstrap).

        Parameters
        ----------
        historical_returns : pd.Series
            Historical daily returns for resampling.
        initial_value : float, default=1.0
            Starting value.

        Returns
        -------
        np.ndarray
            Shape (num_simulations, horizon_days + 1).
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        clean_returns = historical_returns.dropna().values
        if len(clean_returns) < 10:
            clean_returns = np.random.normal(0.0005, 0.02, 1000)

        n = len(clean_returns)
        paths = np.ones((self.num_simulations, self.horizon_days + 1))

        for i in range(self.num_simulations):
            indices = np.random.randint(0, n, size=self.horizon_days)
            sampled = clean_returns[indices]
            cum_ret = np.cumprod(1 + sampled)
            paths[i, 1:] = initial_value * cum_ret

        return paths

    def compute_var(self, paths: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.

        Parameters
        ----------
        paths : np.ndarray
            Simulated paths.
        confidence : float, default=0.95
            Confidence level (95% or 99%).

        Returns
        -------
        float
            VaR as a decimal (e.g., -0.15 means 15% loss at confidence).
        """
        final_values = paths[:, -1]
        if len(final_values) == 0:
            return 0.0
        initial = paths[0, 0] if paths.shape[1] > 0 else 1.0
        returns = (final_values / initial) - 1
        var = np.percentile(returns, (1 - confidence) * 100)
        return float(round(var, 4))

    def compute_cvar(self, paths: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        Parameters
        ----------
        paths : np.ndarray
            Simulated paths.
        confidence : float, default=0.95
            Confidence level.

        Returns
        -------
        float
            CVaR as a decimal.
        """
        final_values = paths[:, -1]
        if len(final_values) == 0:
            return 0.0
        initial = paths[0, 0] if paths.shape[1] > 0 else 1.0
        returns = (final_values / initial) - 1
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        return float(round(cvar, 4)) if not np.isnan(cvar) else 0.0

    def probability_of_loss(self, paths: np.ndarray) -> float:
        """
        Probability of ending below starting value.

        Parameters
        ----------
        paths : np.ndarray
            Simulated paths.

        Returns
        -------
        float
            Probability of loss (0-1).
        """
        final_values = paths[:, -1]
        if len(final_values) == 0:
            return 0.0
        initial = paths[0, 0] if paths.shape[1] > 0 else 1.0
        losses = (final_values < initial).sum()
        return float(round(losses / len(final_values), 4))

    def probability_above_benchmark(
        self,
        paths: np.ndarray,
        benchmark_return: float,
    ) -> float:
        """
        Probability of exceeding a benchmark return.

        Parameters
        ----------
        paths : np.ndarray
            Simulated paths.
        benchmark_return : float
            Benchmark return over the horizon (decimal).

        Returns
        -------
        float
            Probability of outperformance (0-1).
        """
        final_values = paths[:, -1]
        if len(final_values) == 0:
            return 0.0
        initial = paths[0, 0] if paths.shape[1] > 0 else 1.0
        returns = (final_values / initial) - 1
        outperforms = (returns > benchmark_return).sum()
        return float(round(outperforms / len(final_values), 4))

    def full_analysis(
        self,
        annual_return: float,
        annual_volatility: float,
        cdi_return: float = 0.1325,
        ibov_return: float = 0.10,
        initial_value: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run full Monte Carlo analysis with all risk metrics.

        Parameters
        ----------
        annual_return : float
            Expected annual return (decimal).
        annual_volatility : float
            Expected annual volatility (decimal).
        cdi_return : float, default=0.1325
            CDI annual return for comparison.
        ibov_return : float, default=0.10
            IBOV annual return for comparison.
        initial_value : float, default=1.0
            Starting portfolio value.

        Returns
        -------
        dict
            Complete Monte Carlo analysis results.
        """
        paths = self.simulate_gbm(
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            initial_value=initial_value,
        )

        final_values = paths[:, -1]
        returns = (final_values / initial_value) - 1

        hist, bin_edges = np.histogram(returns, bins=50)

        return {
            "num_simulacoes": self.num_simulations,
            "horizonte_dias": self.horizon_days,
            "retorno_esperado": annual_return,
            "volatilidade_esperada": annual_volatility,
            "var_95": self.compute_var(paths, 0.95),
            "var_99": self.compute_var(paths, 0.99),
            "cvar_95": self.compute_cvar(paths, 0.95),
            "cvar_99": self.compute_cvar(paths, 0.99),
            "probabilidade_perda": self.probability_of_loss(paths),
            "probabilidade_superar_cdi": self.probability_above_benchmark(
                paths, cdi_return
            ),
            "probabilidade_superar_ibov": self.probability_above_benchmark(
                paths, ibov_return
            ),
            "retorno_medio_simulado": float(round(np.mean(returns), 4)),
            "retorno_mediano_simulado": float(round(np.median(returns), 4)),
            "std_retornos_simulados": float(round(np.std(returns), 4)),
            "histograma": {
                "valores": hist.tolist(),
                "bordas": bin_edges.tolist(),
            },
            "melhor_cenario": float(round(np.max(returns), 4)),
            "pior_cenario": float(round(np.min(returns), 4)),
        }

    def full_analysis_bootstrap(
        self,
        historical_returns: pd.Series,
        cdi_return: float = 0.1325,
        ibov_return: float = 0.10,
        initial_value: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run full Monte Carlo analysis using bootstrap resampling.

        Parameters
        ----------
        historical_returns : pd.Series
            Historical daily returns for resampling.
        cdi_return : float, default=0.1325
            CDI annual return for comparison.
        ibov_return : float, default=0.10
            IBOV annual return for comparison.
        initial_value : float, default=1.0
            Starting portfolio value.

        Returns
        -------
        dict
            Complete Monte Carlo analysis results.
        """
        paths = self.simulate_from_returns(
            historical_returns=historical_returns,
            initial_value=initial_value,
        )

        final_values = paths[:, -1]
        returns = (final_values / initial_value) - 1

        hist, bin_edges = np.histogram(returns, bins=50)

        return {
            "num_simulacoes": self.num_simulations,
            "horizonte_dias": self.horizon_days,
            "metodo": "bootstrap",
            "var_95": self.compute_var(paths, 0.95),
            "var_99": self.compute_var(paths, 0.99),
            "cvar_95": self.compute_cvar(paths, 0.95),
            "cvar_99": self.compute_cvar(paths, 0.99),
            "probabilidade_perda": self.probability_of_loss(paths),
            "probabilidade_superar_cdi": self.probability_above_benchmark(
                paths, cdi_return
            ),
            "probabilidade_superar_ibov": self.probability_above_benchmark(
                paths, ibov_return
            ),
            "retorno_medio_simulado": float(round(np.mean(returns), 4)),
            "retorno_mediano_simulado": float(round(np.median(returns), 4)),
            "std_retornos_simulados": float(round(np.std(returns), 4)),
            "histograma": {
                "valores": hist.tolist(),
                "bordas": bin_edges.tolist(),
            },
            "melhor_cenario": float(round(np.max(returns), 4)),
            "pior_cenario": float(round(np.min(returns), 4)),
        }
