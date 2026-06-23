import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization model.

    Combines market equilibrium returns (prior) with investor views
    (from QPE scores) to produce a blended expected return vector.
    Then optimizes using mean-variance.

    Reference: Black & Litterman (1992),
    "Global Portfolio Optimization", Financial Analysts Journal
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.1325,
        max_weight: float = 0.08,
    ) -> None:
        """
        Parameters
        ----------
        risk_aversion : float, default=2.5
            Market risk aversion coefficient (lambda).
        tau : float, default=0.05
            Uncertainty of the prior (scaling factor for Σ).
        risk_free_rate : float, default=0.1325
            Risk-free rate (CDI).
        max_weight : float, default=0.08
            Maximum weight per asset.
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight

    def implied_equilibrium_returns(
        self,
        covariance: np.ndarray,
        market_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute implied equilibrium returns (Π) from CAPM.

        Π = λ * Σ * w_mkt

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix (n x n).
        market_weights : np.ndarray
            Market capitalization weights.

        Returns
        -------
        np.ndarray
            Implied excess returns.
        """
        return self.risk_aversion * covariance @ market_weights

    def build_views(
        self,
        scores: Dict[str, float],
        tickers: List[str],
        score_to_return_scale: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build Black-Litterman views from QPE scores.

        Views are constructed as relative outperformance statements:
        - Assets are ranked by score
        - Top quartile will outperform bottom quartile
        - Each view = score-weighted relative return

        Parameters
        ----------
        scores : dict
            Asset -> QPE score.
        tickers : list of str
            Ordered list of tickers corresponding to the covariance matrix.
        score_to_return_scale : float, default=0.05
            How many % return difference a score difference implies.

        Returns
        -------
        tuple of (P, Q, omega)
            P : pick matrix (n_views x n_assets)
            Q : view returns vector
            omega : uncertainty diagonal matrix
        """
        n = len(tickers)
        score_values = np.array([scores.get(t, 50) for t in tickers])
        ranks = np.argsort(score_values)
        rank_scores = score_values[ranks]

        quartile = n // 4
        if quartile < 1:
            quartile = 1

        views_P: List[np.ndarray] = []
        views_Q: List[float] = []

        for q in range(4):
            start = q * quartile
            end = min(start + quartile, n)
            if start >= end:
                continue

            top = ranks[end - quartile // 2:end] if end - quartile // 2 > start else ranks[start:end]
            bot = ranks[start:start + quartile // 2] if start + quartile // 2 < end else ranks[start:end]

            if len(top) == 0 or len(bot) == 0:
                continue

            p = np.zeros(n)
            for idx in top:
                p[idx] = score_values[idx] / score_values[top].sum()
            for idx in bot:
                p[idx] = -score_values[idx] / score_values[bot].sum()

            views_P.append(p)
            views_Q.append(float(abs(score_values[top].mean() - score_values[bot].mean()) * score_to_return_scale))

        if not views_P:
            mean_score = score_values.mean()
            direction = np.where(score_values > mean_score, 1, -1)
            p = direction / len(tickers)
            views_P = [p]
            views_Q = [0.03]

        P = np.array(views_P)
        Q = np.array(views_Q)
        omega = np.diag([0.01] * len(views_Q))

        return P, Q, omega

    def posterior_returns(
        self,
        prior_returns: np.ndarray,
        covariance: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Black-Litterman posterior expected returns.

        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]

        Parameters
        ----------
        prior_returns : np.ndarray
            Prior expected returns (Π).
        covariance : np.ndarray
            Covariance matrix (Σ).
        P : np.ndarray
            Pick matrix.
        Q : np.ndarray
            View return vector.
        omega : np.ndarray
            View uncertainty matrix.

        Returns
        -------
        np.ndarray
            Posterior expected returns.
        """
        n = len(prior_returns)
        tau_sigma_inv = np.linalg.inv(self.tau * covariance)
        p_omega_p = P.T @ np.linalg.inv(omega) @ P

        posterior_cov_inv = tau_sigma_inv + p_omega_p
        posterior_cov = np.linalg.pinv(posterior_cov_inv)

        posterior_mean = posterior_cov @ (tau_sigma_inv @ prior_returns + P.T @ np.linalg.inv(omega) @ Q)

        return posterior_mean

    def optimize(
        self,
        covariance: np.ndarray,
        tickers: List[str],
        scores: Dict[str, float],
        market_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Full Black-Litterman optimization pipeline.

        1. Compute implied equilibrium returns
        2. Build views from QPE scores
        3. Compute posterior returns
        4. Mean-variance optimize

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix.
        tickers : list of str
            Asset tickers.
        scores : dict
            QPE scores.
        market_weights : np.ndarray, optional
            Market cap weights. Defaults to equal weight.

        Returns
        -------
        dict
            BL results with posterior returns, weights, and metrics.
        """
        n = len(tickers)
        if market_weights is None:
            market_weights = np.ones(n) / n

        prior = self.implied_equilibrium_returns(covariance, market_weights)

        P, Q, omega = self.build_views(scores, tickers)
        posterior = self.posterior_returns(prior, covariance, P, Q, omega)

        from scipy.optimize import minimize
        bounds = [(0, self.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        w0 = np.ones(n) / n

        def objective(w):
            port_ret = w @ posterior
            port_var = w @ covariance @ w
            if port_var <= 0:
                return 1e10
            sharpe = (port_ret - self.risk_free_rate) / np.sqrt(port_var)
            return -sharpe

        result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 500})

        if not result.success:
            weights = np.ones(n) / n
        else:
            weights = result.x

        port_ret = weights @ posterior
        port_vol = np.sqrt(weights @ covariance @ weights)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        w_dict = {t: round(float(w), 4) for t, w in zip(tickers, weights)}

        return {
            "method": "black_litterman",
            "pesos": w_dict,
            "retorno_esperado": round(float(port_ret), 4),
            "vol_esperada": round(float(port_vol), 4),
            "sharpe_esperado": round(float(sharpe), 4),
            "prior_returns": prior.tolist(),
            "posterior_returns": posterior.tolist(),
            "n_views": len(Q),
            "view_returns": Q.tolist(),
        }
