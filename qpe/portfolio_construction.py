import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

from qpe.covariance_models import auto_select_covariance, LedoitWolfCovariance, OASCovariance
from qpe.risk_models import TrackingErrorConstraint, ConcentrationConstraint, RiskBudget
from scipy.optimize import minimize


@dataclass
class PortfolioResult:
    """Result from portfolio optimization."""
    weights: Dict[str, float]
    tickers: List[str]
    method: str
    expected_return: float
    expected_vol: float
    sharpe: float
    turnover: Optional[float] = None


class MeanVarianceOptimizer:
    """
    Mean-Variance portfolio optimization with multiple objectives.

    Methods:
    - 'min_variance': Minimize portfolio variance
    - 'max_sharpe': Maximize Sharpe ratio
    - 'risk_parity': Equal risk contribution
    - 'classic': Classic Markowitz (max return for given risk)
    """

    METHODS = ["min_variance", "max_sharpe", "risk_parity", "classic"]

    def __init__(
        self,
        risk_free_rate: float = 0.1325,
        max_weight: float = 0.08,
        min_weight: float = 0.0,
        benchmark_weights: Optional[Dict[str, float]] = None,
        target_te: Optional[float] = None,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.benchmark_weights = benchmark_weights
        self.target_te = target_te

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        tickers: List[str],
        method: str = "max_sharpe",
        sector_map: Optional[Dict[str, str]] = None,
    ) -> PortfolioResult:
        """
        Run portfolio optimization.

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns vector (n_assets,).
        covariance : np.ndarray
            Covariance matrix (n_assets, n_assets).
        tickers : list of str
            Asset tickers.
        method : str
            Optimization method.
        sector_map : dict, optional
            Ticker -> sector for concentration constraints.

        Returns
        -------
        PortfolioResult
            Optimized weights and metrics.
        """
        n = len(tickers)
        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        if method == "min_variance":
            result = self._minimize_variance(covariance, bounds, constraints)
        elif method == "max_sharpe":
            result = self._maximize_sharpe(expected_returns, covariance, bounds, constraints)
        elif method == "risk_parity":
            rb = RiskBudget()
            weights = rb.equal_risk_contribution(covariance)
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / weights.sum()
            return self._build_result(weights, expected_returns, covariance, tickers, method)
        elif method == "classic":
            result = self._classic_markowitz(expected_returns, covariance, bounds, constraints)
        else:
            raise ValueError(f"Unknown method: {method}")

        if not result.success:
            weights = np.ones(n) / n
        else:
            weights = result.x

        return self._build_result(weights, expected_returns, covariance, tickers, method)

    def _minimize_variance(
        self, cov: np.ndarray, bounds: List[Tuple[float, float]],
        constraints: List[Dict[str, Any]],
    ) -> Any:
        n = cov.shape[0]
        w0 = np.ones(n) / n

        def objective(w):
            return w @ cov @ w

        return minimize(objective, w0, method="SLSQP", bounds=bounds,
                        constraints=constraints, options={"maxiter": 500, "ftol": 1e-12})

    def _maximize_sharpe(
        self, mu: np.ndarray, cov: np.ndarray, bounds: List[Tuple[float, float]],
        constraints: List[Dict[str, Any]],
    ) -> Any:
        n = cov.shape[0]
        w0 = np.ones(n) / n

        def objective(w):
            port_var = w @ cov @ w
            if port_var <= 0:
                return 1e10
            port_ret = w @ mu
            sharpe = (port_ret - self.risk_free_rate) / np.sqrt(port_var)
            return -sharpe

        return minimize(objective, w0, method="SLSQP", bounds=bounds,
                        constraints=constraints, options={"maxiter": 500, "ftol": 1e-12})

    def _classic_markowitz(
        self, mu: np.ndarray, cov: np.ndarray, bounds: List[Tuple[float, float]],
        constraints: List[Dict[str, Any]],
    ) -> Any:
        n = cov.shape[0]
        w0 = np.ones(n) / n

        target_return = mu.max() * 0.5
        constraints = constraints + [
            {"type": "eq", "fun": lambda w: w @ mu - target_return}
        ]

        def objective(w):
            return w @ cov @ w

        return minimize(objective, w0, method="SLSQP", bounds=bounds,
                        constraints=constraints, options={"maxiter": 500, "ftol": 1e-12})

    def _build_result(
        self, weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
        tickers: List[str], method: str,
    ) -> PortfolioResult:
        port_ret = weights @ mu
        port_var = weights @ cov @ weights
        port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        w_dict = {t: round(float(w), 4) for t, w in zip(tickers, weights)}

        return PortfolioResult(
            weights=w_dict,
            tickers=tickers,
            method=method,
            expected_return=float(port_ret),
            expected_vol=float(port_vol),
            sharpe=float(sharpe),
        )

    def efficient_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        tickers: List[str],
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier for visualization.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns.
        cov : np.ndarray
            Covariance matrix.
        tickers : list of str
            Asset tickers.
        n_points : int, default=50
            Number of points on the frontier.

        Returns
        -------
        pd.DataFrame
            Risk, return, and weights for each frontier point.
        """
        bounds = [(self.min_weight, self.max_weight)] * len(tickers)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        min_var = self._minimize_variance(cov, bounds, constraints)
        max_ret_w = np.zeros(len(tickers))
        max_ret_w[np.argmax(mu)] = 1.0

        min_ret = min_var.x @ mu if min_var.success else mu.min()
        max_ret = mu.max()

        targets = np.linspace(min_ret, max_ret, n_points)
        results = []

        for target in targets:
            cons = constraints + [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
            w0 = np.ones(len(tickers)) / len(tickers)
            res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                           bounds=bounds, constraints=cons, options={"maxiter": 500})
            if res.success:
                vol = np.sqrt(res.x @ cov @ res.x)
                results.append({
                    "retorno": float(res.x @ mu),
                    "risco": float(vol),
                    "sharpe": float((res.x @ mu - self.risk_free_rate) / vol) if vol > 0 else 0,
                })

        return pd.DataFrame(results)


class TwoStagePortfolioBuilder:
    """
    Two-stage portfolio construction.

    Stage 1: Select top N assets by QPE score.
    Stage 2: Apply risk-based optimization with constraints.

    Tests top_k values: [20, 30, 40, 50]
    """

    def __init__(
        self,
        optim_method: str = "max_sharpe",
        covariance_method: str = "auto",
        max_asset_weight: float = 0.08,
        max_sector_weight: float = 0.20,
        min_positions: int = 15,
        max_positions: int = 30,
        target_te: float = 0.08,
        risk_free_rate: float = 0.1325,
    ) -> None:
        self.optim_method = optim_method
        self.covariance_method = covariance_method
        self.max_asset_weight = max_asset_weight
        self.max_sector_weight = max_sector_weight
        self.min_positions = min_positions
        self.max_positions = max_positions
        self.target_te = target_te
        self.risk_free_rate = risk_free_rate

    def build(
        self,
        tickers: List[str],
        scores: Dict[str, float],
        returns: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        top_k: int = 30,
        regime_adjusted_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Build portfolio using two-stage process.

        Parameters
        ----------
        tickers : list of str
            Full universe.
        scores : dict
            Ticker -> QPE score.
        returns : pd.DataFrame
            Historical returns (columns = tickers).
        sector_map : dict, optional
            Ticker -> sector.
        top_k : int, default=30
            Number of top assets to select in stage 1.
        regime_adjusted_scores : dict, optional
            Regime-adjusted scores from AlphaEngine.

        Returns
        -------
        dict
            Full results with stage 1, stage 2, and metrics.
        """
        use_scores = regime_adjusted_scores if regime_adjusted_scores is not None else scores

        sorted_tickers = sorted(
            use_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_n = min(top_k, len(sorted_tickers))
        selected = [t for t, s in sorted_tickers[:top_n] if s > 0]
        selected_scores = [s for t, s in sorted_tickers[:top_n] if s > 0]

        available_returns = returns[[c for c in returns.columns if c in selected]]
        if available_returns.shape[1] < 2:
            return {"error": "Insufficient return data", "selected": selected}

        return_array = available_returns.dropna().values
        if return_array.shape[0] < 10:
            return {"error": "Insufficient history", "selected": selected}

        cov_result = auto_select_covariance(return_array)
        covariance = cov_result.covariance

        expected_returns = np.array(selected_scores, dtype=float)
        expected_returns = expected_returns / expected_returns.mean() * 0.15

        mvo = MeanVarianceOptimizer(
            risk_free_rate=self.risk_free_rate,
            max_weight=self.max_asset_weight,
            min_weight=0.0,
        )

        result = mvo.optimize(
            expected_returns=expected_returns,
            covariance=covariance,
            tickers=selected,
            method=self.optim_method,
            sector_map=sector_map,
        )

        if self.target_te and self.target_te > 0:
            te_control = TrackingErrorConstraint(max_te=self.target_te)
            bm_weights = np.ones(len(selected)) / len(selected)
            w_array = np.array([result.weights[t] for t in selected])
            w_adj = te_control.project(w_array, bm_weights, covariance)
            result = self._rebuild_result(result, w_adj, selected, expected_returns, covariance)

        if sector_map:
            conc = ConcentrationConstraint(
                max_asset_weight=self.max_asset_weight,
                max_sector_weight=self.max_sector_weight,
            )
            idx_sector = {i: sector_map.get(t, "Outros") for i, t in enumerate(selected)}
            w_array = np.array([result.weights[t] for t in selected])
            w_adj = conc.apply_sector_cap(w_array, idx_sector)
            result = self._rebuild_result(result, w_adj, selected, expected_returns, covariance)

        w_array = np.array([result.weights.get(t, 0) for t in selected])
        active = (w_array > 0.001).sum()
        active = max(self.min_positions, min(active, self.max_positions))

        return {
            "stage1": {
                "universo_total": len(tickers),
                "selecionados": len(selected),
                "top_k": top_k,
                "tickers": selected,
            },
            "stage2": {
                "metodo": result.method,
                "metodo_covariancia": cov_result.method,
                "shrinkage": cov_result.shrinkage,
                "pesos": result.weights,
                "retorno_esperado": round(result.expected_return * 100, 2),
                "vol_esperada": round(result.expected_vol * 100, 2),
                "sharpe_esperado": round(result.sharpe, 4),
                "ativos_ativos": int(active),
            },
            "covariancia": covariance.tolist(),
        }

    @staticmethod
    def _rebuild_result(
        result: PortfolioResult, w_adj: np.ndarray, tickers: List[str],
        mu: np.ndarray, cov: np.ndarray,
    ) -> PortfolioResult:
        w_dict = {t: round(float(w), 4) for t, w in zip(tickers, w_adj)}
        port_ret = w_adj @ mu
        port_vol = np.sqrt(w_adj @ cov @ w_adj) if (w_adj @ cov @ w_adj) > 0 else 0.0
        rf = 0.1325
        sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
        return PortfolioResult(
            weights=w_dict, tickers=tickers, method=result.method,
            expected_return=float(port_ret), expected_vol=float(port_vol),
            sharpe=float(sharpe),
        )
