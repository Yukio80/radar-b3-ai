from qpe.multi_factor_score import MultiFactorScore


def test_score_quality():
    mfs = MultiFactorScore()
    row = {"roe": 0.25, "roic": 0.18, "margem_liquida": 0.15}
    assert mfs._score_quality(row) > 50


def test_score_valuation():
    mfs = MultiFactorScore()
    row = {"pl": 8, "pvp": 0.9, "ev_ebit": 5}
    score = mfs._score_valuation(row)
    assert 0 < score <= 100


def test_score_dividends():
    mfs = MultiFactorScore()
    row = {"dy": 0.06, "dividend_consistency": 0.8}
    score = mfs._score_dividends(row)
    assert 0 < score <= 100


def test_score_safety():
    mfs = MultiFactorScore()
    row = {"divida_pl": 0.5, "liquidez_corrente": 2.0}
    score = mfs._score_safety(row)
    assert 0 < score <= 100


def test_compute_full():
    mfs = MultiFactorScore()
    row = {
        "roe": 0.20, "roic": 0.15, "margem_liquida": 0.12,
        "pl": 10, "pvp": 1.2, "ev_ebit": 7,
        "dy": 0.05, "dividend_consistency": 0.7,
        "cagr_revenue": 0.08, "cagr_net_income": 0.10,
        "divida_pl": 0.8, "liquidez_corrente": 1.8,
    }
    result = mfs.compute(row)
    assert "total_score" in result
    assert 0 < result["total_score"] <= 100
    assert result["quality"] > 0
    assert result["valuation"] > 0
    assert result["growth"] > 0


def test_classify():
    assert MultiFactorScore.classify(95) == "Elite"
    assert MultiFactorScore.classify(85) == "Excelente"
    assert MultiFactorScore.classify(75) == "Boa"
    assert MultiFactorScore.classify(65) == "Média"
    assert MultiFactorScore.classify(55) == "Fraca"


def test_percentile():
    import pandas as pd
    mfs = MultiFactorScore()
    s = pd.Series([50, 60, 70, 80, 90])
    ranked = mfs.apply_percentile_ranking(s)
    assert ranked.min() >= 0
    assert ranked.max() <= 100
    assert ranked.is_monotonic_increasing
