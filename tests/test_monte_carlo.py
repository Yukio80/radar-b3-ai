import pandas as pd
import numpy as np
from qpe.monte_carlo import MonteCarloEngine


def test_simulate_gbm_shape():
    mc = MonteCarloEngine(num_simulations=1000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0.12, annual_volatility=0.20)
    assert paths.shape == (1000, 253)


def test_simulate_gbm_final_value():
    mc = MonteCarloEngine(num_simulations=1000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0.15, annual_volatility=0)
    final = paths[:, -1]
    expected = np.exp(0.15)
    assert np.allclose(final, expected, atol=0.001)


def test_var_95():
    mc = MonteCarloEngine(num_simulations=10000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0, annual_volatility=0.20)
    var95 = mc.compute_var(paths, 0.95)
    assert var95 < 0


def test_var_99():
    mc = MonteCarloEngine(num_simulations=10000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0, annual_volatility=0.20)
    var99 = mc.compute_var(paths, 0.99)
    assert var99 < mc.compute_var(paths, 0.95)


def test_cvar():
    mc = MonteCarloEngine(num_simulations=10000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0, annual_volatility=0.20)
    cvar95 = mc.compute_cvar(paths, 0.95)
    var95 = mc.compute_var(paths, 0.95)
    assert cvar95 <= var95


def test_probability_of_loss():
    mc = MonteCarloEngine(num_simulations=10000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=-0.50, annual_volatility=0.10)
    prob = mc.probability_of_loss(paths)
    assert prob > 0.5


def test_probability_above_benchmark():
    mc = MonteCarloEngine(num_simulations=10000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0.50, annual_volatility=0.10)
    prob = mc.probability_above_benchmark(paths, 0.30)
    assert prob > 0.5


def test_full_analysis():
    mc = MonteCarloEngine(num_simulations=1000, horizon_days=252, seed=42)
    result = mc.full_analysis(
        annual_return=0.12,
        annual_volatility=0.20,
        cdi_return=0.1325,
        ibov_return=0.10,
    )
    assert "var_95" in result
    assert "var_99" in result
    assert "cvar_95" in result
    assert "probabilidade_perda" in result
    assert "probabilidade_superar_cdi" in result
    assert "probabilidade_superar_ibov" in result
    assert "histograma" in result


def test_full_analysis_bootstrap():
    mc = MonteCarloEngine(num_simulations=1000, horizon_days=252, seed=42)
    np.random.seed(42)
    hist_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
    result = mc.full_analysis_bootstrap(
        historical_returns=hist_returns,
        cdi_return=0.1325,
        ibov_return=0.10,
    )
    assert "var_95" in result
    assert result["metodo"] == "bootstrap"


def test_empty_paths_var():
    mc = MonteCarloEngine()
    paths = np.array([]).reshape(0, 10)
    assert mc.compute_var(paths) == 0


def test_constant_return():
    mc = MonteCarloEngine(num_simulations=100, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0.10, annual_volatility=0)
    assert mc.probability_of_loss(paths) == 0


def test_high_volatility_var():
    mc = MonteCarloEngine(num_simulations=5000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0, annual_volatility=0.50)
    var95 = mc.compute_var(paths, 0.95)
    assert var95 < -0.3


def test_probability_superar_cdi_alta():
    mc = MonteCarloEngine(num_simulations=1000, horizon_days=252, seed=42)
    paths = mc.simulate_gbm(annual_return=0.30, annual_volatility=0.05)
    prob = mc.probability_above_benchmark(paths, 0.1325)
    assert prob > 0.90
