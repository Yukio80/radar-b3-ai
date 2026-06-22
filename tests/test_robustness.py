from qpe.robustness_index import RobustnessIndex


def test_diversification():
    ri = RobustnessIndex()
    score = ri.diversification_score(20)
    assert score == 100


def test_diversification_few():
    ri = RobustnessIndex()
    score = ri.diversification_score(5)
    assert score < 100


def test_quality():
    ri = RobustnessIndex()
    score = ri.average_quality_score([80, 90, 70])
    assert score == 80


def test_dividend_stability():
    ri = RobustnessIndex()
    score = ri.dividend_stability_score([0.05, 0.06, 0.055])
    assert 0 < score <= 100


def test_low_leverage():
    ri = RobustnessIndex()
    score = ri.low_leverage_score([0.3, 0.5, 0.8])
    assert score > 50


def test_compute():
    ri = RobustnessIndex()
    result = ri.compute(
        num_assets=15,
        quality_scores=[80, 70, 90, 85],
        dy_values=[0.05, 0.06, 0.04],
        debt_values=[0.5, 0.8, 0.3],
    )
    assert "IRP" in result
    assert "classificacao" in result
    assert 0 < result["IRP"] <= 100


def test_classify():
    assert RobustnessIndex.classify(95) == "Excelente"
    assert RobustnessIndex.classify(70) == "Boa"
    assert RobustnessIndex.classify(30) == "Fraca"
