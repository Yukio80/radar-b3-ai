import numpy as np
import pandas as pd
from qpe.correlation_analysis import CorrelationAnalyzer


def test_correlation_matrix():
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(100, 5),
        columns=["A", "B", "C", "D", "E"],
    )
    ca = CorrelationAnalyzer(df)
    corr = ca.correlation_matrix()
    assert corr.shape == (5, 5)
    assert abs(corr.iloc[0, 0] - 1.0) < 1e-10


def test_covariance_matrix():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    ca = CorrelationAnalyzer(df)
    cov = ca.covariance_matrix()
    assert cov.shape == (3, 3)


def test_average_correlation():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])
    ca = CorrelationAnalyzer(df)
    ca.correlation_matrix()
    avg = ca.average_correlation()
    assert -1 <= avg <= 1


def test_effective_diversification():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 5), columns=list("ABCDE"))
    ca = CorrelationAnalyzer(df)
    ca.correlation_matrix()
    eff = ca.effective_diversification()
    assert eff >= 1


def test_effective_diversification_single():
    df = pd.DataFrame(np.random.randn(100, 1), columns=["A"])
    ca = CorrelationAnalyzer(df)
    ca.correlation_matrix()
    eff = ca.effective_diversification()
    assert abs(eff - 1.0) < 0.01


def test_vif():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))
    ca = CorrelationAnalyzer(df)
    vif_result = ca.vif()
    assert "VIF" in vif_result.columns
    assert len(vif_result) == 4


def test_pca():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 6), columns=list("ABCDEF"))
    ca = CorrelationAnalyzer(df)
    result = ca.pca(n_components=3)
    assert "variancia_explicada" in result
    assert len(result["variancia_explicada"]) == 3
    assert "loadings" in result


def test_sector_concentration():
    weights = {"PETR4": 30, "VALE3": 30, "ITUB4": 20, "BBDC4": 20}
    sector_map = {
        "PETR4": "Petróleo", "VALE3": "Mineração",
        "ITUB4": "Financeiro", "BBDC4": "Financeiro",
    }
    ca = CorrelationAnalyzer()
    result = ca.sector_concentration(weights, sector_map)
    assert result["num_setores"] == 3
    assert result["concentracao"] in ["Alta", "Moderada", "Baixa"]


def test_factor_correlation():
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(50, 5),
        columns=["quality", "valuation", "dividends", "growth", "safety"],
    )
    ca = CorrelationAnalyzer()
    result = ca.factor_correlation(df)
    assert "correlacao" in result
    assert "vif" in result


def test_set_returns():
    ca = CorrelationAnalyzer()
    df = pd.DataFrame(np.random.randn(50, 3), columns=list("ABC"))
    ca.set_returns(df)
    corr = ca.correlation_matrix()
    assert corr.shape == (3, 3)


def test_empty_returns():
    ca = CorrelationAnalyzer(pd.DataFrame())
    assert ca.correlation_matrix().empty
    assert ca.covariance_matrix().empty
    assert ca.average_correlation() == 0
    assert ca.vif().empty


def test_pca_single_asset():
    df = pd.DataFrame(np.random.randn(100, 1), columns=["A"])
    ca = CorrelationAnalyzer(df)
    result = ca.pca()
    assert "error" in result


def test_factor_correlation_single():
    df = pd.DataFrame({"quality": np.random.randn(50)})
    ca = CorrelationAnalyzer()
    result = ca.factor_correlation(df)
    assert "error" in result


def test_high_correlation_vif():
    np.random.seed(42)
    base = np.random.randn(100)
    df = pd.DataFrame({
        "A": base,
        "B": base * 0.99 + np.random.randn(100) * 0.01,
        "C": np.random.randn(100),
    })
    ca = CorrelationAnalyzer(df)
    vif = ca.vif()
    assert vif.loc["A", "VIF"] > 1
