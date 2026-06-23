import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detect and classify market regimes based on returns, volatility,
    and macroeconomic proxies.

    Regimes:
    - Bull Market: strong positive returns, low volatility
    - Bear Market: sustained negative returns, high volatility
    - Alta de Juros: high-rate environment (proxied by benchmark behavior)
    - Baixa de Juros: low-rate environment
    - Crise: extreme negative returns, very high volatility
    - Recuperação: recovery from crisis, positive returns, falling vol
    """

    REGIME_LABELS = {
        "bull": "Bull Market",
        "bear": "Bear Market",
        "high_rates": "Alta de Juros",
        "low_rates": "Baixa de Juros",
        "crisis": "Crise",
        "recovery": "Recuperação",
        "unknown": "Indefinido",
    }

    def __init__(
        self,
        lookback_days: int = 252,
        vol_window: int = 63,
    ) -> None:
        """
        Parameters
        ----------
        lookback_days : int, default=252
            Window for return calculation (1 year).
        vol_window : int, default=63
            Window for volatility calculation (3 months).
        """
        self.lookback_days = lookback_days
        self.vol_window = vol_window
        self._current_regime: str = "unknown"

    def detect(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        cdi_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect the current market regime.

        Parameters
        ----------
        returns : pd.Series
            Market returns (e.g., IBOV daily returns).
        benchmark_returns : pd.Series, optional
            Additional benchmark for cross-validation.
        cdi_rate : float, optional
            Current CDI rate for rate environment detection.

        Returns
        -------
        dict
            Detected regime with supporting metrics.
        """
        r = returns.dropna()
        if len(r) < self.vol_window:
            return {
                "regime": "unknown",
                "classificacao": "Indefinido",
                "metricas": {},
                "confianca": 0.0,
            }

        lookback = r.iloc[-self.lookback_days:] if len(r) >= self.lookback_days else r
        recent = r.iloc[-self.vol_window:] if len(r) >= self.vol_window else r

        ann_return = (1 + lookback).prod() ** (252 / len(lookback)) - 1
        ann_vol = recent.std() * np.sqrt(252)

        recent_return = (1 + recent).prod() - 1
        recent_vol = recent.std() * np.sqrt(252)

        max_dd = self._compute_max_drawdown(lookback)
        sharpe = (ann_return - 0.1325) / ann_vol if ann_vol > 0 else 0

        regime, confidence = self._classify(
            ann_return=ann_return,
            ann_vol=ann_vol,
            recent_return=recent_return,
            recent_vol=recent_vol,
            max_dd=max_dd,
            sharpe=sharpe,
            cdi_rate=cdi_rate,
        )

        self._current_regime = regime

        return {
            "regime": regime,
            "classificacao": self.REGIME_LABELS.get(regime, "Indefinido"),
            "confianca": round(confidence, 2),
            "metricas": {
                "retorno_anualizado": round(float(ann_return), 4),
                "volatilidade_anualizada": round(float(ann_vol), 4),
                "retorno_recente_3m": round(float(recent_return), 4),
                "volatilidade_recente_3m": round(float(recent_vol), 4),
                "max_drawdown": round(float(max_dd), 4),
                "sharpe_12m": round(float(sharpe), 4),
            },
        }

    def detect_rolling(
        self,
        returns: pd.Series,
        window_size: int = 252,
        step: int = 21,
    ) -> pd.DataFrame:
        """
        Detect regimes over rolling windows for historical analysis.

        Parameters
        ----------
        returns : pd.Series
            Daily returns.
        window_size : int, default=252
            Rolling window size.
        step : int, default=21
            Step between windows (monthly).

        Returns
        -------
        pd.DataFrame
            Date -> regime and metrics.
        """
        r = returns.dropna()
        if len(r) < window_size:
            return pd.DataFrame()

        dates = r.index[window_size - 1 :: step]
        results: List[Dict[str, Any]] = []

        for d in dates:
            pos = r.index.get_loc(d)
            window = r.iloc[pos - window_size + 1 : pos + 1]
            if len(window) < window_size:
                continue

            result = self.detect(window)
            result["data"] = d
            results.append(result)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("data")
        return df

    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(abs(drawdown.min()))

    def _classify(
        self,
        ann_return: float,
        ann_vol: float,
        recent_return: float,
        recent_vol: float,
        max_dd: float,
        sharpe: float,
        cdi_rate: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Classify the regime based on market metrics.

        Returns
        -------
        tuple of (regime_key, confidence)
        """
        if ann_return < -0.20 and max_dd > 0.30:
            return "crisis", 0.85
        if max_dd > 0.25 and recent_return > 0.05 and recent_vol < ann_vol:
            return "recovery", 0.70
        if ann_return > 0.15 and ann_vol < 0.25 and sharpe > 0.5:
            return "bull", 0.80
        if ann_return < -0.05 and ann_vol > 0.30:
            return "bear", 0.75
        if cdi_rate is not None and cdi_rate > 0.14:
            return "high_rates", 0.60
        if cdi_rate is not None and cdi_rate < 0.08:
            return "low_rates", 0.55
        if ann_return > 0 and ann_vol < 0.20:
            return "bull", 0.50
        if ann_return < -0.05:
            return "bear", 0.50
        if ann_vol > 0.30:
            return "bear", 0.40

        return "unknown", 0.30

    @staticmethod
    def adjust_weights_for_regime(
        regime: str,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Return factor weight adjustments for a given regime.

        Parameters
        ----------
        regime : str
            Detected regime key.
        current_weights : dict, optional
            Current factor weights to adjust.

        Returns
        -------
        dict
            Adjusted factor weights.
        """
        base = {
            "quality": 0.25,
            "valuation": 0.20,
            "dividends": 0.20,
            "growth": 0.20,
            "safety": 0.15,
        }

        adjustments: Dict[str, Dict[str, float]] = {
            "bull": {"growth": 0.05, "dividends": -0.05},
            "bear": {"safety": 0.05, "quality": 0.03, "growth": -0.05, "valuation": -0.03},
            "high_rates": {"dividends": 0.05, "quality": 0.02, "growth": -0.05, "valuation": -0.02},
            "low_rates": {"growth": 0.05, "valuation": 0.03, "dividends": -0.05, "safety": -0.03},
            "crisis": {"safety": 0.10, "quality": 0.05, "growth": -0.10, "valuation": -0.05},
            "recovery": {"growth": 0.05, "valuation": 0.05, "safety": -0.05, "dividends": -0.05},
        }

        adj = adjustments.get(regime, {})

        if current_weights:
            weights = dict(current_weights)
        else:
            weights = dict(base)

        for factor, delta in adj.items():
            if factor in weights:
                weights[factor] = max(0.05, min(0.50, weights[factor] + delta))

        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] = round(weights[k] / total, 4)

        return weights

    @staticmethod
    def regime_description(regime: str) -> str:
        """Get a description of a regime."""
        descriptions = {
            "bull": "Mercado em alta com tendencia positiva e volatilidade controlada. Aumentar exposicao a crescimento.",
            "bear": "Mercado em queda com volatilidade elevada. Aumentar seguranca e qualidade, reduzir crescimento.",
            "high_rates": "Juros altos favorecem ativos de dividendos e qualidade. Reduzir exposicao a crescimento.",
            "low_rates": "Juros baixos favorecem growth e valuation. Reduzir exposicao a dividendos.",
            "crisis": "Crise severa. Maxima protecao: seguranca e qualidade. Evitar crescimento e valuation.",
            "recovery": "Recuperacao pos-crise. Aumentar exposicao a growth e valuation. Reduzir seguranca.",
            "unknown": "Regime indefinido. Manter pesos padrao.",
        }
        return descriptions.get(regime, "Regime nao reconhecido.")
