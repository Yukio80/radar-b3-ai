import numpy as np
import pandas as pd
from qpe.performance_metrics import PerformanceMetrics


def test_cumulative_return():
    pm = PerformanceMetrics()
    r = pd.Series([0.01, 0.02, -0.01, 0.03])
    result = pm.cumulative_return(r)
    assert abs(result - 0.0503) < 0.01


def test_annualized_return():
    pm = PerformanceMetrics()
    r = pd.Series([0.001] * 252)
    result = pm.annualized_return(r, 252)
    assert abs(result - 0.286) < 0.02


def test_annualized_volatility():
    pm = PerformanceMetrics()
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.001, 0.02, 252))
    vol = pm.annualized_volatility(r, 252)
    assert 0.15 < vol < 0.50


def test_sharpe_ratio():
    pm = PerformanceMetrics(risk_free_rate=0.1325)
    r = pd.Series([0.001] * 252)
    sharpe = pm.sharpe_ratio(r, 252)
    assert sharpe > 0


def test_sortino_ratio():
    pm = PerformanceMetrics(risk_free_rate=0.1325)
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.001, 0.02, 252))
    sortino = pm.sortino_ratio(r, 252)
    assert isinstance(sortino, float)


def test_max_drawdown():
    pm = PerformanceMetrics()
    r = pd.Series([0.01, -0.05, 0.02, -0.03, 0.01])
    mdd = pm.max_drawdown(r)
    assert 0 < mdd < 0.10


def test_max_drawdown_known():
    r = pd.Series([-0.10, 0.05, -0.20, 0.10])
    pm = PerformanceMetrics()
    mdd = pm.max_drawdown(r)
    assert mdd > 0.10
    assert mdd < 0.30


def test_calmar_ratio():
    pm = PerformanceMetrics()
    r = pd.Series([0.001] * 252)
    calmar = pm.calmar_ratio(r, 252)
    assert isinstance(calmar, float)


def test_alpha_beta():
    pm = PerformanceMetrics(risk_free_rate=0.10)
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252))
    ab = pm.alpha_beta(portfolio, benchmark, 252)
    assert "alpha" in ab
    assert "beta" in ab
    assert "r_squared" in ab


def test_alpha_beta_perfect():
    pm = PerformanceMetrics(risk_free_rate=0.10)
    np.random.seed(42)
    common = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = common * 1.2 + 0.0001
    ab = pm.alpha_beta(portfolio, common, 252)
    assert abs(ab["beta"] - 1.2) < 0.1


def test_tracking_error():
    pm = PerformanceMetrics()
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    te = pm.tracking_error(portfolio, benchmark, 252)
    assert te >= 0


def test_information_ratio():
    pm = PerformanceMetrics()
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.002, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    ir = pm.information_ratio(portfolio, benchmark, 252)
    assert isinstance(ir, float)


def test_all_metrics():
    pm = PerformanceMetrics()
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252))
    metrics = pm.all_metrics(portfolio, benchmark, 252)
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "alpha" in metrics
    assert "beta" in metrics
    assert "information_ratio" in metrics


def test_empty_returns():
    pm = PerformanceMetrics()
    r = pd.Series([], dtype=float)
    metrics = pm.all_metrics(r)
    assert metrics["retorno_acumulado"] == 0
    assert metrics["sharpe_ratio"] == 0


def test_single_return():
    pm = PerformanceMetrics()
    r = pd.Series([0.01])
    assert pm.sharpe_ratio(r) == 0


def test_cumulative_return_negative():
    pm = PerformanceMetrics()
    r = pd.Series([-0.01, -0.02, -0.01])
    result = pm.cumulative_return(r)
    assert result < 0


def test_sharpe_negative_rf():
    pm = PerformanceMetrics(risk_free_rate=0.20)
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.0001, 0.01, 252))
    sharpe = pm.sharpe_ratio(r, 252)
    assert sharpe < 1.0


def test_sortino_no_downside():
    pm = PerformanceMetrics()
    r = pd.Series([0.01] * 100)
    sortino = pm.sortino_ratio(r, 252)
    assert isinstance(sortino, float)


def test_beta_identity():
    pm = PerformanceMetrics(risk_free_rate=0)
    np.random.seed(42)
    x = pd.Series(np.random.normal(0, 0.01, 100))
    ab = pm.alpha_beta(x, x, 252)
    assert abs(ab["beta"] - 1.0) < 0.01
    assert abs(ab["alpha"]) < 0.01


def test_tracking_zero():
    pm = PerformanceMetrics()
    x = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02] * 50)
    te = pm.tracking_error(x, x, 252)
    assert abs(te) < 1e-10
