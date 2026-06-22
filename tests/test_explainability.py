from qpe.explainability import Explainability


def test_explain_asset():
    exp = Explainability()
    result = exp.explain_asset({
        "ticker": "TEST4",
        "roe": 0.25,
        "roic": 0.18,
        "margem_liquida": 0.15,
        "pl": 8.0,
        "pvp": 1.0,
        "ev_ebit": 5.0,
        "dy": 0.06,
        "cagr_revenue": 0.10,
        "cagr_net_income": 0.12,
        "divida_pl": 0.5,
        "liquidez_corrente": 2.0,
        "total_score": 85.0,
    })
    assert result["ticker"] == "TEST4"
    assert result["score"] == 85.0
    assert len(result["motivos"]) > 0
    assert len(result["pontos_fortes"]) > 0


def test_explain_weak_asset():
    exp = Explainability()
    result = exp.explain_asset({
        "ticker": "WEAK4",
        "roe": 0.02,
        "roic": 0.01,
        "margem_liquida": 0.01,
        "pl": 50.0,
        "pvp": 5.0,
        "ev_ebit": 30.0,
        "dy": 0.005,
        "cagr_revenue": -0.05,
        "cagr_net_income": -0.10,
        "divida_pl": 5.0,
        "liquidez_corrente": 0.5,
        "total_score": 30.0,
    })
    assert len(result.get("pontos_fracos", [])) > 0


def test_batch_explain():
    exp = Explainability()
    assets = [
        {"ticker": "A", "roe": 0.20, "dy": 0.05, "pl": 10, "pvp": 1.2,
         "ev_ebit": 7, "margem_liquida": 0.10, "roic": 0.15,
         "cagr_revenue": 0.08, "cagr_net_income": 0.10,
         "divida_pl": 0.8, "liquidez_corrente": 1.8, "total_score": 80},
        {"ticker": "B", "roe": 0.05, "dy": 0.01, "pl": 30, "pvp": 3.0,
         "ev_ebit": 20, "margem_liquida": 0.03, "roic": 0.04,
         "cagr_revenue": -0.02, "cagr_net_income": -0.05,
         "divida_pl": 3.0, "liquidez_corrente": 0.8, "total_score": 40},
    ]
    results = exp.batch_explain(assets)
    assert len(results) == 2
    assert results[0]["ticker"] == "A"
    assert results[1]["ticker"] == "B"
