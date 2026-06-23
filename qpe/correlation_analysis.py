import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyze correlation structure, diversification, and factor redundancy.

    Provides:
    - Pearson / Spearman correlation matrices
    - Covariance matrix
    - VIF (Variance Inflation Factor) for multicollinearity
    - PCA for dimensionality / redundancy detection
    - Effective diversification metrics
    - Sector and factor concentration analysis
    """

    def __init__(
        self,
        returns: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Parameters
        ----------
        returns : pd.DataFrame, optional
            Asset returns (columns = tickers, index = dates).
        """
        self.returns = returns
        self._correlation: Optional[pd.DataFrame] = None
        self._covariance: Optional[pd.DataFrame] = None

    def set_returns(self, returns: pd.DataFrame) -> None:
        """Set or update the returns DataFrame."""
        self.returns = returns
        self._correlation = None
        self._covariance = None

    def correlation_matrix(
        self,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Calculate the correlation matrix.

        Parameters
        ----------
        method : str, default="pearson"
            Correlation method: 'pearson' or 'spearman'.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        if self.returns is None or self.returns.empty:
            return pd.DataFrame()

        self._correlation = self.returns.corr(method=method)
        return self._correlation

    def covariance_matrix(self) -> pd.DataFrame:
        """
        Calculate the covariance matrix.

        Returns
        -------
        pd.DataFrame
            Covariance matrix.
        """
        if self.returns is None or self.returns.empty:
            return pd.DataFrame()

        self._covariance = self.returns.cov()
        return self._covariance

    def effective_diversification(self) -> float:
        """
        Calculate effective number of bets (diversification).

        Uses 1 / sum(w^2) where weights are derived from inverse
        variance or from the eigendecomposition of the correlation matrix.

        Returns
        -------
        float
            Effective diversification count (>= 1).
        """
        if self._correlation is None:
            self.correlation_matrix()

        if self._correlation is None or self._correlation.empty:
            return 1.0

        eigvals = np.linalg.eigvalsh(self._correlation.values)
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total == 0:
            return 1.0
        weights = eigvals / total
        effective = 1.0 / (weights ** 2).sum()
        return float(round(effective, 2))

    def average_correlation(self) -> float:
        """
        Calculate the average pairwise correlation.

        Returns
        -------
        float
            Average correlation coefficient.
        """
        if self._correlation is None:
            self.correlation_matrix()

        if self._correlation is None or self._correlation.empty:
            return 0.0

        n = len(self._correlation)
        if n < 2:
            return 0.0

        upper = self._correlation.values[np.triu_indices(n, k=1)]
        if len(upper) == 0:
            return 0.0
        return float(round(float(np.mean(upper)), 4))

    def vif(self) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for each asset.

        VIF = 1 / (1 - R^2) where R^2 comes from regressing one
        asset against all others. High VIF (>5) indicates redundancy.

        Returns
        -------
        pd.DataFrame
            Asset -> VIF value.
        """
        if self.returns is None or self.returns.empty:
            return pd.DataFrame()

        from sklearn.linear_model import LinearRegression

        data = self.returns.dropna()
        if data.shape[1] < 2:
            return pd.DataFrame({"VIF": [1.0]}, index=data.columns)

        results: Dict[str, float] = {}
        for col in data.columns:
            y = data[col].values
            X = data.drop(columns=[col]).values
            if X.shape[1] == 0 or X.shape[0] < 2:
                results[col] = 1.0
                continue

            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else 10.0
            results[col] = round(vif, 2)

        return pd.DataFrame.from_dict(results, orient="index", columns=["VIF"]).sort_values(
            "VIF", ascending=False
        )

    def pca(
        self,
        n_components: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform PCA on the returns to detect redundancy.

        Parameters
        ----------
        n_components : int, optional
            Number of components. Defaults to min(features, 10).

        Returns
        -------
        dict
            PCA results with explained variance, loadings, etc.
        """
        if self.returns is None or self.returns.empty:
            return {}

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        data = self.returns.dropna()
        if data.shape[1] < 2:
            return {"error": "Need at least 2 assets for PCA"}

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        n = min(data.shape[1], n_components or min(data.shape[1], 10))
        pca = PCA(n_components=n)
        components = pca.fit_transform(scaled)

        explained_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(explained_var)

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n)],
            index=data.columns,
        )

        n_redundant = 0
        threshold = 0.95
        for i, cv in enumerate(cum_var):
            if cv >= threshold:
                n_redundant = data.shape[1] - (i + 1)
                break

        return {
            "n_componentes": n,
            "n_ativos": data.shape[1],
            "variancia_explicada": explained_var.tolist(),
            "variancia_acumulada": cum_var.tolist(),
            "componentes_para_95": int(np.argmax(cum_var >= threshold) + 1) if any(cum_var >= threshold) else n,
            "ativos_redundantes": n_redundant,
            "loadings": loadings,
        }

    def sector_concentration(
        self,
        weights: Dict[str, float],
        sector_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Analyze sector concentration of a portfolio.

        Parameters
        ----------
        weights : dict
            Ticker -> weight percentage.
        sector_map : dict
            Ticker -> sector name.

        Returns
        -------
        dict
            Sector allocation, HHI, and concentration assessment.
        """
        sector_weights: Dict[str, float] = {}
        for ticker, w in weights.items():
            sector = sector_map.get(ticker, "Outros")
            sector_weights[sector] = sector_weights.get(sector, 0) + w

        shares = np.array(list(sector_weights.values()))
        total = shares.sum()
        if total > 0:
            proportions = shares / total
        else:
            proportions = shares

        hhi = (proportions ** 2).sum()

        if hhi > 0.3:
            concentration = "Alta"
        elif hhi > 0.15:
            concentration = "Moderada"
        else:
            concentration = "Baixa"

        return {
            "alocacao_setorial": sector_weights,
            "hhi": round(float(hhi), 4),
            "concentracao": concentration,
            "num_setores": len(sector_weights),
        }

    def factor_correlation(
        self,
        factor_scores: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyze correlation between QPE factors.

        Parameters
        ----------
        factor_scores : pd.DataFrame
            Columns should include: quality, valuation, dividends,
            growth, safety, total_score.

        Returns
        -------
        dict
            Factor correlation matrix, VIF, and redundancy notes.
        """
        factors = ["quality", "valuation", "dividends", "growth", "safety"]
        available = [f for f in factors if f in factor_scores.columns]

        if len(available) < 2:
            return {
                "error": "Need at least 2 factors for correlation analysis",
                "correlacao": pd.DataFrame(),
            }

        corr = factor_scores[available].corr()

        vif_results: Dict[str, float] = {}
        for col in available:
            y = factor_scores[col].values
            others = [f for f in available if f != col]
            if len(others) == 0:
                vif_results[col] = 1.0
                continue
            X = factor_scores[others].values
            if X.shape[0] < 3:
                vif_results[col] = 1.0
                continue
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            vif_results[col] = round(1.0 / (1.0 - r2) if r2 < 1.0 else 10.0, 2)

        high_vif = {k: v for k, v in vif_results.items() if v > 5}

        return {
            "correlacao": corr,
            "vif": vif_results,
            "alta_colinearidade": high_vif,
            "recomendacao": (
                "Redundância detectada entre fatores. Considere "
                "combinar ou remover fatores com VIF > 5."
                if high_vif
                else "Fatores com baixa colinearidade."
            ),
        }
