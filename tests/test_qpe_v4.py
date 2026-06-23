import numpy as np
import pandas as pd
from qpe.covariance_models import LedoitWolfCovariance, OASCovariance, auto_select_covariance
from qpe.risk_models import TrackingErrorConstraint, ConcentrationConstraint, RiskBudget
from qpe.portfolio_construction import MeanVarianceOptimizer, TwoStagePortfolioBuilder, PortfolioResult
from qpe.alpha_engine import AlphaEngine
from qpe.attribution import AlphaAttributionEngine, FactorContribution
from qpe.black_litterman import BlackLittermanOptimizer
from qpe.enhanced_stress import AdvancedStressTest


def test_ledoit_wolf_shape():
    np.random.seed(42)
    returns = np.random.randn(100, 5)
    lw = LedoitWolfCovariance()
    result = lw.fit(returns)
    assert result.covariance.shape == (5, 5)
    assert 0 <= result.shrinkage <= 1
    assert result.method == "ledoit_wolf"


def test_oas_shape():
    np.random.seed(42)
    returns = np.random.randn(100, 5)
    oas = OASCovariance()
    result = oas.fit(returns)
    assert result.covariance.shape == (5, 5)
    assert 0 <= result.shrinkage <= 1
    assert result.method == "oas"


def test_auto_select():
    np.random.seed(42)
    returns = np.random.randn(100, 5)
    result = auto_select_covariance(returns)
    assert result.covariance.shape == (5, 5)


def test_tracking_error_project():
    np.random.seed(42)
    te = TrackingErrorConstraint(max_te=0.08)
    cov = np.random.randn(10, 10)
    cov = cov.T @ cov
    w = np.random.rand(10)
    w = w / w.sum()
    bm = np.ones(10) / 10
    result = te.project(w, bm, cov)
    assert abs(result.sum() - 1.0) < 0.01
    diff = result - bm
    actual_te = np.sqrt(diff @ cov @ diff) * np.sqrt(252)
    assert actual_te <= 0.09


def test_concentration_asset_cap():
    conc = ConcentrationConstraint(max_asset_weight=0.08)
    w = np.array([0.5, 0.3, 0.2])
    result = conc.apply_asset_cap(w)
    assert result[0] == 0.08


def test_concentration_sector():
    conc = ConcentrationConstraint(max_sector_weight=0.20)
    w = np.array([0.4, 0.3, 0.3])
    sec_map = {0: "A", 1: "B", 2: "B"}
    result = conc.apply_sector_cap(w, sec_map)
    assert abs(result.sum() - 1.0) < 0.01
    assert all(result >= 0)


def test_risk_parity():
    np.random.seed(42)
    cov = np.random.randn(5, 5)
    cov = cov.T @ cov
    rb = RiskBudget()
    w = rb.equal_risk_contribution(cov)
    assert abs(w.sum() - 1.0) < 0.01
    assert all(w > 0)


def test_mean_variance_min_variance():
    np.random.seed(42)
    mu = np.array([0.12, 0.15, 0.10, 0.13, 0.11])
    cov = np.random.randn(5, 5)
    cov = cov.T @ cov * 0.01
    mvo = MeanVarianceOptimizer(max_weight=0.40)
    result = mvo.optimize(mu, cov, ["A", "B", "C", "D", "E"], method="min_variance")
    assert abs(sum(result.weights.values()) - 1.0) < 0.01
    assert result.method == "min_variance"


def test_mean_variance_max_sharpe():
    np.random.seed(42)
    mu = np.array([0.15, 0.12, 0.10, 0.14, 0.11])
    cov = np.random.randn(5, 5)
    cov = cov.T @ cov * 0.01
    mvo = MeanVarianceOptimizer(max_weight=0.40, risk_free_rate=0.10)
    result = mvo.optimize(mu, cov, ["A", "B", "C", "D", "E"], method="max_sharpe")
    assert abs(sum(result.weights.values()) - 1.0) < 0.01


def test_efficient_frontier():
    np.random.seed(42)
    mu = np.array([0.12, 0.15, 0.10])
    cov = np.random.randn(3, 3)
    cov = cov.T @ cov * 0.01
    mvo = MeanVarianceOptimizer(max_weight=0.50)
    ef = mvo.efficient_frontier(mu, cov, ["A", "B", "C"], n_points=10)
    assert len(ef) > 0
    assert "risco" in ef.columns


def test_alpha_engine_base():
    ae = AlphaEngine()
    scores = {"quality": 80, "valuation": 60, "dividends": 70, "growth": 50, "safety": 90}
    alpha = ae.compute_alpha(scores)
    assert 0 <= alpha <= 100


def test_alpha_engine_regime_adjust():
    ae = AlphaEngine()
    weights = ae.get_factor_weights("bull")
    assert weights["growth"] > 0.20
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_alpha_engine_batch():
    ae = AlphaEngine()
    assets = [
        {"ticker": "PETR4", "quality": 90, "valuation": 50, "dividends": 50,
         "growth": 80, "safety": 40},
        {"ticker": "VALE3", "quality": 80, "valuation": 60, "dividends": 60,
         "growth": 50, "safety": 70},
    ]
    results = ae.compute_alpha_batch(assets, regime="bull")
    assert len(results) == 2
    assert "alpha_score" in results[0]


def test_attribution():
    np.random.seed(42)
    ae = AlphaAttributionEngine(risk_free_rate=0.10)
    factor_weights = {"quality": 0.25, "valuation": 0.20, "dividends": 0.20,
                      "growth": 0.20, "safety": 0.15}
    n = 20
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    scores = pd.DataFrame(np.random.rand(n, 5), columns=list(factor_weights.keys()),
                          index=[f"ASSET{i}" for i in range(n)])
    asset_ret = pd.DataFrame(np.random.randn(500, n) * 0.02, index=dates,
                             columns=[f"ASSET{i}" for i in range(n)])
    bm_ret = pd.Series(np.random.randn(500) * 0.015, index=dates, name="IBOV")
    pw = {f"ASSET{i}": 1.0/n for i in range(n)}
    result = ae.attribute(factor_weights, scores, asset_ret, bm_ret, pw)
    assert "fatores" in result
    assert "melhor_fator" in result
    assert len(result["fatores"]) == 5


def test_attribution_report():
    ae = AlphaAttributionEngine()
    result = {
        "fatores": [
            FactorContribution("quality", 0.25, 0.02, 0.10, 0.01, 2.1, True),
            FactorContribution("valuation", 0.20, 0.01, 0.05, 0.005, 1.5, False),
        ],
        "total_contribuicao_retorno": 0.03,
        "total_contribuicao_alpha": 0.015,
        "retorno_real_carteira": 0.12,
        "fatores_significativos": 1,
        "melhor_fator": "quality",
        "pior_fator": "valuation",
    }
    report = ae.report(result)
    assert "Alpha Attribution" in report
    assert "quality" in report


def test_black_litterman_implied():
    np.random.seed(42)
    cov = np.random.randn(5, 5)
    cov = cov.T @ cov * 0.01
    bl = BlackLittermanOptimizer()
    w_mkt = np.ones(5) / 5
    implied = bl.implied_equilibrium_returns(cov, w_mkt)
    assert implied.shape == (5,)


def test_black_litterman_views():
    bl = BlackLittermanOptimizer()
    scores = {"A": 90, "B": 80, "C": 70, "D": 60, "E": 50}
    tickers = ["A", "B", "C", "D", "E"]
    P, Q, omega = bl.build_views(scores, tickers)
    assert P.shape[0] == len(Q)
    assert omega.shape == (len(Q), len(Q))


def test_black_litterman_posterior():
    np.random.seed(42)
    bl = BlackLittermanOptimizer()
    prior = np.array([0.12, 0.15, 0.10, 0.13, 0.11])
    cov = np.random.randn(5, 5)
    cov = cov.T @ cov * 0.01
    scores = {"A": 90, "B": 80, "C": 70, "D": 60, "E": 50}
    tickers = ["A", "B", "C", "D", "E"]
    P, Q, omega = bl.build_views(scores, tickers)
    posterior = bl.posterior_returns(prior, cov, P, Q, omega)
    assert posterior.shape == (5,)


def test_black_litterman_optimize():
    np.random.seed(42)
    bl = BlackLittermanOptimizer(max_weight=0.40)
    cov = np.random.randn(5, 5)
    cov = cov.T @ cov * 0.01
    scores = {"A": 90, "B": 80, "C": 70, "D": 60, "E": 50}
    tickers = ["A", "B", "C", "D", "E"]
    result = bl.optimize(cov, tickers, scores)
    assert "pesos" in result
    assert abs(sum(result["pesos"].values()) - 1.0) < 0.01


def test_advanced_stress():
    stress = AdvancedStressTest()
    weights = {"PETR4": 30, "VALE3": 20, "ITUB4": 20, "BBDC4": 15, "ELET3": 15}
    sectors = {"PETR4": "Petróleo", "VALE3": "Mineração",
               "ITUB4": "Financeiro", "BBDC4": "Financeiro", "ELET3": "Utilities"}
    result = stress.run_scenario("Crash Commodities", weights, sectors)
    assert "perda_estimada" in result
    assert result["perda_estimada"] < 0


def test_advanced_stress_all():
    stress = AdvancedStressTest()
    weights = {"PETR4": 30, "VALE3": 20, "ITUB4": 20, "BBDC4": 15, "ELET3": 15}
    sectors = {"PETR4": "Petróleo", "VALE3": "Mineração",
               "ITUB4": "Financeiro", "BBDC4": "Financeiro", "ELET3": "Utilities"}
    results = stress.run_all(weights, sectors)
    assert "pior_cenario" in results
    assert results["classificacao_risco"] in ["Risco Extremo", "Risco Alto", "Risco Moderado", "Risco Baixo"]


def test_two_stage_builder():
    np.random.seed(42)
    builder = TwoStagePortfolioBuilder(optim_method="min_variance", max_asset_weight=0.40)
    tickers = [f"ASSET{i}" for i in range(20)]
    scores = {t: np.random.uniform(30, 90) for t in tickers}
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    returns = pd.DataFrame(np.random.randn(100, 20) * 0.02, index=dates, columns=tickers)
    result = builder.build(tickers, scores, returns, top_k=10)
    assert "stage1" in result
    assert "stage2" in result


def test_ledoit_wolf_small_sample():
    returns = np.random.randn(5, 10)
    lw = LedoitWolfCovariance()
    try:
        result = lw.fit(returns)
        assert result.covariance.shape == (10, 10)
    except ValueError:
        pass


def test_tracking_error_no_iteration():
    te = TrackingErrorConstraint(max_te=1.0)
    cov = np.eye(5)
    w = np.ones(5) / 5
    result = te.project(w, w, cov)
    assert np.allclose(result, w)


def test_risk_parity_identity():
    rb = RiskBudget()
    cov = np.eye(3)
    w = rb.equal_risk_contribution(cov)
    assert np.allclose(w, 1/3, atol=0.01)


def test_alpha_engine_all_regimes():
    ae = AlphaEngine()
    for regime in ["bull", "bear", "crisis", "recovery", "high_rates", "low_rates", "unknown"]:
        w = ae.get_factor_weights(regime)
        assert abs(sum(w.values()) - 1.0) < 0.01


def test_efficient_frontier_single():
    mu = np.array([0.10])
    cov = np.array([[0.01]])
    mvo = MeanVarianceOptimizer(max_weight=1.0, min_weight=0.0)
    ef = mvo.efficient_frontier(mu, cov, ["A"], n_points=5)
    assert len(ef) <= 6
