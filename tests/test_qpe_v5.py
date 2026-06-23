import numpy as np
import pandas as pd
import pytest
from qpe.portfolio_profiles import PortfolioProfile, get_profile, list_profiles, PROFILES
from qpe.conviction_score import ConvictionEngine
from qpe.market_score import QPEMarketScore
from qpe.recommendation_engine import RecommendationEngine, RecommendedPortfolio
from qpe.recommendation_reports import _carteira_report, generate_market_report, generate_validation_report, save_report


def test_all_profiles_defined():
    assert len(PROFILES) == 6
    for name in ["core", "alpha", "dividendos", "valor", "crescimento", "defensiva"]:
        assert name in PROFILES


def test_get_profile():
    p = get_profile("core")
    assert p.name == "CORE"
    assert p.allocation_method == "risk_parity"


def test_get_profile_case_insensitive():
    p = get_profile("ALPHA")
    assert p.name == "ALPHA"


def test_get_profile_invalid():
    import pytest
    with pytest.raises(ValueError):
        get_profile("nonexistent")


def test_list_profiles():
    names = list_profiles()
    assert len(names) == 6
    assert "core" in names


def test_profile_dataclass():
    p = PortfolioProfile(
        name="TESTE",
        description="Test profile",
        objective="Test objective",
        allocation_method="equal_weight",
        score_thresholds={"total_score": 50},
        sort_key="total_score",
        max_positions=10,
        max_asset_weight=0.20,
        max_sector_weight=0.30,
    )
    assert p.name == "TESTE"
    assert p.max_positions == 10
    assert p.target_te is None


def test_profile_with_optional_fields():
    p = PortfolioProfile(
        name="TESTE2",
        description="Test with optional fields",
        objective="Test",
        allocation_method="max_sharpe",
        score_thresholds={"total_score": 60},
        sort_key="alpha_score",
        max_positions=15,
        max_asset_weight=0.08,
        max_sector_weight=0.20,
        target_te=0.08,
        min_irp=65,
        required_profiles=["core"],
    )
    assert p.target_te == 0.08
    assert p.min_irp == 65
    assert p.required_profiles == ["core"]


def test_alpha_profile_constraints():
    p = get_profile("alpha")
    assert p.max_asset_weight == 0.08
    assert p.max_sector_weight == 0.20
    assert p.target_te == 0.08


def test_defensiva_profile():
    p = get_profile("defensiva")
    assert p.allocation_method == "min_variance"
    assert p.max_positions == 20


def test_dividendos_profile():
    p = get_profile("dividendos")
    assert "dividends" in p.score_thresholds
    assert p.score_thresholds["dividends"] >= 70


def test_conviction_engine_init():
    ce = ConvictionEngine()
    assert ce.WEIGHTS["qpe_score"] == 0.30
    assert ce.WEIGHTS["walk_forward"] == 0.15


def test_conviction_normalize():
    ce = ConvictionEngine()
    assert ce._normalize(80, 0, 100) == 80.0
    assert ce._normalize(0, 0, 100) == 0.0


def test_conviction_normalize_edge():
    ce = ConvictionEngine()
    assert ce._normalize(50, 50, 50) == 50.0


def test_conviction_regime_alignment():
    ce = ConvictionEngine()
    factor_scores = {"quality": 90, "valuation": 50, "dividends": 50, "growth": 50, "safety": 50}
    score = ce._regime_alignment_score(factor_scores, "bull")
    assert 0 <= score <= 100


def test_conviction_regime_alignment_crisis():
    ce = ConvictionEngine()
    factor_scores = {"quality": 50, "valuation": 50, "dividends": 50, "growth": 50, "safety": 90}
    score = ce._regime_alignment_score(factor_scores, "crisis")
    assert score > 50


def test_conviction_estabilidade_no_history():
    ce = ConvictionEngine()
    score = ce._estabilidade_score("PETR4", 75)
    assert score == 50.0


def test_conviction_estabilidade_stable():
    ce = ConvictionEngine()
    ce.update_historical_scores("PETR4", 80)
    ce.update_historical_scores("PETR4", 82)
    ce.update_historical_scores("PETR4", 81)
    score = ce._estabilidade_score("PETR4", 80)
    assert score > 50


def test_conviction_estabilidade_volatile():
    ce = ConvictionEngine()
    ce.update_historical_scores("PETR4", 90)
    ce.update_historical_scores("PETR4", 20)
    ce.update_historical_scores("PETR4", 80)
    score = ce._estabilidade_score("PETR4", 50)
    assert 0 <= score <= 100


def test_conviction_walk_forward_none():
    ce = ConvictionEngine()
    score = ce._walk_forward_score(None)
    assert score == 50.0


def test_conviction_walk_forward_good():
    ce = ConvictionEngine()
    result = {"taxa_acerto": 80, "retorno_medio_teste": 0.15}
    score = ce._walk_forward_score(result)
    expected = 80 * 0.6 + 0.30 * 0.4
    assert score == pytest.approx(expected, abs=0.01)


def test_conviction_walk_forward_poor():
    ce = ConvictionEngine()
    result = {"taxa_acerto": 30, "retorno_medio_teste": -0.10}
    score = ce._walk_forward_score(result)
    assert score < 50


def test_conviction_compute():
    ce = ConvictionEngine()
    result = ce.compute(
        ticker="PETR4",
        qpe_score=80,
        irp_score=70,
        factor_scores={"quality": 90, "valuation": 50, "dividends": 60, "growth": 70, "safety": 80},
        regime="bull",
    )
    assert result["ticker"] == "PETR4"
    assert 0 <= result["conviction_score"] <= 100
    assert "componentes" in result


def test_conviction_compute_max():
    ce = ConvictionEngine()
    result = ce.compute(
        ticker="TEST",
        qpe_score=100,
        irp_score=100,
        factor_scores={"quality": 100, "valuation": 100, "dividends": 100, "growth": 100, "safety": 100},
        regime="bull",
    )
    assert result["conviction_score"] == pytest.approx(82.5, abs=0.1)
    assert result["conviction_label"] == "Muito Alta"


def test_conviction_compute_zero():
    ce = ConvictionEngine()
    result = ce.compute(
        ticker="TEST",
        qpe_score=0,
        irp_score=0,
        factor_scores={"quality": 0, "valuation": 0, "dividends": 0, "growth": 0, "safety": 0},
        regime="crisis",
    )
    assert result["conviction_score"] > 0
    assert "Baixa" in result["conviction_label"] or "Muito Baixa" in result["conviction_label"]


def test_conviction_batch():
    ce = ConvictionEngine()
    assets = [
        {"ticker": "PETR4", "total_score": 80, "irp_score": 70,
         "quality": 90, "valuation": 50, "dividends": 60, "growth": 70, "safety": 80},
        {"ticker": "VALE3", "total_score": 60, "irp_score": 50,
         "quality": 70, "valuation": 60, "dividends": 40, "growth": 50, "safety": 60},
    ]
    results = ce.compute_batch(assets, regime="bull")
    assert len(results) == 2
    assert results[0]["conviction_score"] >= results[1]["conviction_score"]


def test_conviction_classify():
    assert ConvictionEngine.classify(85) == "Muito Alta"
    assert ConvictionEngine.classify(70) == "Alta"
    assert ConvictionEngine.classify(50) == "Media"
    assert ConvictionEngine.classify(30) == "Baixa"
    assert ConvictionEngine.classify(10) == "Muito Baixa"


def test_conviction_update_history():
    ce = ConvictionEngine()
    ce.update_historical_scores("PETR4", 80)
    ce.update_historical_scores("PETR4", 85)
    assert len(ce.historical_scores["PETR4"]) == 2


def test_market_score_empty():
    result = QPEMarketScore.compute([], [], "unknown")
    assert result["market_score"] == 50.0
    assert result["market_label"] == "Neutro"


def test_market_score_compute():
    scores = [80, 70, 60, 50, 40]
    factor_scores = [
        {"quality": 80, "valuation": 70, "dividends": 60, "growth": 50, "safety": 90},
        {"quality": 70, "valuation": 60, "dividends": 50, "growth": 60, "safety": 80},
        {"quality": 60, "valuation": 50, "dividends": 70, "growth": 40, "safety": 70},
    ]
    result = QPEMarketScore.compute(scores, factor_scores, "bull")
    assert 0 <= result["market_score"] <= 100
    assert "componentes" in result
    assert "total_ativos_analisados" in result


def test_market_score_regime_impact():
    scores = [60, 60, 60]
    factor_scores = [{"quality": 60, "valuation": 60, "dividends": 60, "growth": 60, "safety": 60}]
    bull_result = QPEMarketScore.compute(scores, factor_scores, "bull")
    bear_result = QPEMarketScore.compute(scores, factor_scores, "bear")
    assert bull_result["market_score"] > bear_result["market_score"]


def test_market_score_crisis():
    scores = [50, 50]
    factor_scores = [{"quality": 50, "valuation": 50, "dividends": 50, "growth": 50, "safety": 50}]
    result = QPEMarketScore.compute(scores, factor_scores, "crisis")
    assert result["market_score"] < 50


def test_market_score_recovery():
    scores = [60, 60]
    factor_scores = [{"quality": 60, "valuation": 60, "dividends": 60, "growth": 60, "safety": 60}]
    result = QPEMarketScore.compute(scores, factor_scores, "recovery")
    assert result["market_score"] > 50


def test_market_score_classify():
    assert QPEMarketScore.classify(85) == "Muito Atrativo"
    assert QPEMarketScore.classify(70) == "Atrativo"
    assert QPEMarketScore.classify(50) == "Neutro"
    assert QPEMarketScore.classify(30) == "Caro"
    assert QPEMarketScore.classify(15) == "Muito Caro"


def test_market_score_dispersion():
    scores = [50, 50, 50]
    factor_scores = [
        {"quality": 50, "valuation": 10, "dividends": 50, "growth": 50, "safety": 50},
        {"quality": 50, "valuation": 90, "dividends": 50, "growth": 50, "safety": 50},
        {"quality": 50, "valuation": 50, "dividends": 50, "growth": 50, "safety": 50},
    ]
    result = QPEMarketScore.compute(scores, factor_scores, "unknown")
    assert result["componentes"]["dispersao_valuation"] > 0


def test_recommendation_engine_init():
    engine = RecommendationEngine()
    assert engine.current_regime == "unknown"


def test_recommendation_engine_screen():
    engine = RecommendationEngine()
    assets = [
        {"ticker": "PETR4", "total_score": 90, "quality": 80, "valuation": 70,
         "dividends": 60, "growth": 50, "safety": 90, "setor": "Petroleo",
         "dy": 0.06, "divida_pl": 0.5},
        {"ticker": "VALE3", "total_score": 40, "quality": 30, "valuation": 30,
         "dividends": 50, "growth": 40, "safety": 30, "setor": "Mineracao",
         "dy": 0.02, "divida_pl": 1.5},
    ]
    from qpe.portfolio_profiles import get_profile
    profile = get_profile("core")
    screened = engine._screen_assets(assets, profile)
    assert len(screened) == 1
    assert screened[0]["ticker"] == "PETR4"


def test_recommendation_engine_estimate_irp():
    engine = RecommendationEngine()
    asset = {"quality": 80, "dy": 0.06, "divida_pl": 0.5}
    irp = engine._estimate_irp_score(asset)
    assert 0 <= irp <= 100


def test_recommendation_engine_estimate_irp_low():
    engine = RecommendationEngine()
    asset = {"quality": 20, "dy": 0.0, "divida_pl": 5.0}
    irp = engine._estimate_irp_score(asset)
    assert irp < 50


def test_recommendation_engine_estimate_irp_high():
    engine = RecommendationEngine()
    asset = {"quality": 95, "dy": 0.15, "divida_pl": 0.1}
    irp = engine._estimate_irp_score(asset)
    assert irp > 80


def test_recommendation_engine_allocate_equal():
    engine = RecommendationEngine()
    enriched = [{"ticker": "A", "total_score": 80}, {"ticker": "B", "total_score": 60},
                {"ticker": "C", "total_score": 40}]
    weights = engine._allocate_equal(enriched)
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_recommendation_engine_allocate_equal_empty():
    engine = RecommendationEngine()
    weights = engine._allocate_equal([])
    assert weights == {}


def test_recommendation_compute_market_score():
    engine = RecommendationEngine()
    assets = [
        {"ticker": "A", "total_score": 80, "quality": 80, "valuation": 70,
         "dividends": 60, "growth": 50, "safety": 90},
        {"ticker": "B", "total_score": 60, "quality": 60, "valuation": 50,
         "dividends": 50, "growth": 60, "safety": 70},
    ]
    result = engine._compute_market_score(assets, "bull")
    assert "market_score" in result


def test_recommendation_compute_portfolio_returns():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    returns = pd.DataFrame(np.random.randn(100, 2) * 0.02, index=dates, columns=["A", "B"])
    weights = {"A": 0.6, "B": 0.4}
    pf_ret = RecommendationEngine._compute_portfolio_returns(returns, weights)
    assert len(pf_ret) > 0
    assert not pf_ret.isnull().all()


def test_recommendation_compute_portfolio_returns_missing():
    returns = pd.DataFrame({"A": [0.01, 0.02]})
    weights = {"B": 1.0}
    pf_ret = RecommendationEngine._compute_portfolio_returns(returns, weights)
    assert pf_ret.empty


def test_recommendation_generate_explanations():
    engine = RecommendationEngine()
    from qpe.portfolio_profiles import get_profile
    profile = get_profile("core")
    enriched = [
        {"ticker": "PETR4", "total_score": 85, "alpha_score": 80, "conviction_score": 90,
         "conviction_label": "Muito Alta", "quality": 90, "valuation": 70,
         "dividends": 60, "growth": 50, "safety": 85, "setor": "Petroleo"},
    ]
    weights = {"PETR4": 0.08}
    expl = engine._generate_explanations(enriched, weights, profile)
    assert len(expl) == 1
    assert expl[0]["ticker"] == "PETR4"
    assert len(expl[0]["motivos"]) > 0


def test_recommendation_generate_explanations_low_weight():
    engine = RecommendationEngine()
    from qpe.portfolio_profiles import get_profile
    profile = get_profile("core")
    enriched = [
        {"ticker": "PETR4", "total_score": 85, "alpha_score": 80, "conviction_score": 90,
         "conviction_label": "Muito Alta", "quality": 90, "valuation": 70,
         "dividends": 60, "growth": 50, "safety": 85, "setor": "Petroleo"},
    ]
    weights = {"PETR4": 0.0005}
    expl = engine._generate_explanations(enriched, weights, profile)
    assert len(expl) == 0


def test_recommendation_engine_recommend_minimal():
    engine = RecommendationEngine()
    assets = [
        {"ticker": "PETR4", "total_score": 80, "alpha_score": 75,
         "quality": 80, "valuation": 60, "dividends": 70, "growth": 50, "safety": 85,
         "setor": "Petroleo", "dy": 0.06, "divida_pl": 0.5},
        {"ticker": "VALE3", "total_score": 70, "alpha_score": 65,
         "quality": 70, "valuation": 65, "dividends": 50, "growth": 55, "safety": 75,
         "setor": "Mineracao", "dy": 0.04, "divida_pl": 0.8},
    ]
    rec = engine.recommend("core", assets, regime="bull")
    assert isinstance(rec, RecommendedPortfolio)
    assert rec.profile == "CORE"
    assert len(rec.positions) > 0


def test_recommendation_engine_recommend_all():
    engine = RecommendationEngine()
    assets = [
        {"ticker": "PETR4", "total_score": 80, "alpha_score": 75,
         "quality": 80, "valuation": 60, "dividends": 70, "growth": 50, "safety": 85,
         "setor": "Petroleo", "dy": 0.06, "divida_pl": 0.5},
        {"ticker": "VALE3", "total_score": 70, "alpha_score": 65,
         "quality": 70, "valuation": 65, "dividends": 50, "growth": 55, "safety": 75,
         "setor": "Mineracao", "dy": 0.04, "divida_pl": 0.8},
        {"ticker": "ITUB4", "total_score": 75, "alpha_score": 70,
         "quality": 75, "valuation": 55, "dividends": 65, "growth": 45, "safety": 80,
         "setor": "Financeiro", "dy": 0.05, "divida_pl": 0.6},
    ]
    results = engine.recommend_all(assets, regime="bull")
    assert len(results) == 6
    for name in ["core", "alpha", "dividendos", "valor", "crescimento", "defensiva"]:
        assert name in results
        assert isinstance(results[name], RecommendedPortfolio)


def test_recommendation_engine_screen_empty():
    engine = RecommendationEngine()
    from qpe.portfolio_profiles import get_profile
    profile = get_profile("alpha")
    assets = [{"ticker": "A", "total_score": 20, "quality": 20, "dy": 0,
               "divida_pl": 10, "setor": "Outros"}]
    screened = engine._screen_assets(assets, profile)
    assert len(screened) <= 1


def test_recommendation_compute_metrics_empty():
    engine = RecommendationEngine()
    metrics = engine._compute_metrics([], {}, None, None)
    assert metrics.get("retorno_anualizado", 0) == 0


def test_recommendation_compute_metrics():
    np.random.seed(42)
    engine = RecommendationEngine()
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    returns = pd.DataFrame(np.random.randn(100, 2) * 0.02, index=dates, columns=["A", "B"])
    bm = pd.Series(np.random.randn(100) * 0.015, index=dates, name="IBOV")
    enriched = [{"ticker": "A", "total_score": 50}, {"ticker": "B", "total_score": 50}]
    weights = {"A": 0.5, "B": 0.5}
    metrics = engine._compute_metrics(enriched, weights, returns, bm)
    assert "sharpe_ratio" in metrics or "retorno_anualizado" in metrics


def test_carteira_report():
    from qpe.portfolio_profiles import PortfolioProfile
    rec = RecommendedPortfolio(
        profile="CORE",
        description="Test portfolio",
        objective="Long term",
        regime="bull",
        market_score={"market_score": 65, "market_label": "Atrativo"},
        positions=[{"ticker": "PETR4", "peso": 8.0, "score": 85, "conviction": 90,
                     "conviction_label": "Muito Alta", "qualidade": 90, "valuation": 70,
                     "dividendos": 60, "crescimento": 50, "seguranca": 85}],
        weights={"PETR4": 0.08},
        metrics={"retorno_anualizado": 0.15, "volatilidade_anualizada": 0.20,
                 "sharpe_ratio": 0.75, "sortino_ratio": 1.2, "max_drawdown": -0.15,
                 "alpha": 0.05, "beta": 1.1, "r_squared": 0.8},
        conviction_media=85.0,
        irp_result={"IRP": 75, "classificacao": "Bom", "sub_scores": {"diversificacao": 80}},
        stress_test={"cenarios": {"Crash": {"perda_estimada": -15, "recuperacao_estimada_dias": 120}}},
        advanced_stress={"cenarios": {"Crash": {"perda_estimada": -25, "recuperacao_estimada_dias": 180}}},
        score_medio=70.0,
        explicacoes=[{"ticker": "PETR4", "peso": 8.0, "score": 85, "conviction": 90,
                       "conviction_label": "Muito Alta",
                       "motivos": ["Score QPE: 85.0/100", "Conviction: 90.0 (Muito Alta)"],
                       "pontos_fortes": ["Quality score 90/100"],
                       "riscos": ["Growth baixo (50/100)"]}],
    )
    report = _carteira_report(rec, "core", "bull")
    assert "CORE" in report
    assert "PETR4" in report
    assert "Score QPE" in report


def test_carteira_report_empty_stress():
    rec = RecommendedPortfolio(
        profile="ALPHA", description="Alpha", objective="Beat benchmark",
        regime="bull", market_score={"market_score": 50, "market_label": "Neutro"},
        positions=[], weights={},
        metrics={"retorno_anualizado": 0, "sharpe_ratio": 0},
        conviction_media=50, irp_result={},
        stress_test={}, advanced_stress={},
        score_medio=50, explicacoes=[],
    )
    report = _carteira_report(rec, "alpha", "bull")
    assert "ALPHA" in report


def test_generate_market_report():
    market_score = {
        "market_score": 60.5,
        "market_label": "Atrativo",
        "total_ativos_analisados": 50,
        "componentes": {"media_scores_universo": 55, "dispersao_valuation": 30,
                        "nivel_qualidade": 70, "regime_score": 60},
    }
    consolidated = {
        "core": {"profile": "CORE", "score_medio": 70, "conviction_media": 80,
                 "sharpe": 0.8, "irp": 75, "num_ativos": 15},
        "alpha": {"profile": "ALPHA", "score_medio": 65, "conviction_media": 75,
                  "sharpe": 0.9, "irp": 60, "num_ativos": 12},
    }
    report = generate_market_report(market_score, "bull", "Mercado em alta", consolidated)
    assert "Relatorio de Mercado" in report or "Market" in report
    assert "CRESCIMENTO" in report


def test_generate_market_report_bear():
    market_score = {"market_score": 30, "market_label": "Caro",
                    "total_ativos_analisados": 50, "componentes": {}}
    report = generate_market_report(market_score, "bear", "Mercado em baixa", {})
    assert "DEFENSIVA" in report


def test_generate_validation_report():
    recommendations = {
        "core": {"profile": "CORE", "alpha": 0.05, "sharpe": 0.8,
                 "drawdown": -0.15, "conviction_media": 80, "score_medio": 70,
                 "irp": 75, "regime": "bull"},
        "alpha": {"profile": "ALPHA", "alpha": 0.08, "sharpe": 0.9,
                  "drawdown": -0.20, "conviction_media": 75, "score_medio": 65,
                  "irp": 60, "regime": "bull"},
        "crescimento": {"profile": "CRESCIMENTO", "alpha": 0.06, "sharpe": 0.7,
                        "drawdown": -0.25, "conviction_media": 70, "score_medio": 60,
                        "irp": 55, "regime": "bull"},
    }
    report = generate_validation_report(recommendations)
    assert "Relatorio de Validacao" in report or "Validacao" in report
    assert "ALPHA" in report
    assert "Qual possui maior alpha esperado" in report or "alpha esperado" in report


def test_generate_validation_report_empty():
    report = generate_validation_report({})
    assert report is not None


def test_save_report(tmp_path):
    content = "# Test Report"
    path = save_report(content, "test_report.md", output_dir=str(tmp_path))
    assert path is not None
    assert path.endswith("test_report.md")


def test_save_report_default_dir():
    import os
    content = "# Test"
    path = save_report(content, "tmp_test_report.md")
    assert os.path.exists(path)
    os.remove(path)


def test_conviction_engine_history_max():
    ce = ConvictionEngine()
    for i in range(20):
        ce.update_historical_scores("TEST", 50 + i, max_history=10)
    assert len(ce.historical_scores["TEST"]) == 10


def test_recommendation_engine_regime_change():
    engine = RecommendationEngine()
    assets = [
        {"ticker": "A", "total_score": 80, "alpha_score": 75,
         "quality": 80, "valuation": 60, "dividends": 70, "growth": 50, "safety": 85,
         "setor": "Outros", "dy": 0.06, "divida_pl": 0.5},
        {"ticker": "B", "total_score": 70, "alpha_score": 65,
         "quality": 70, "valuation": 65, "dividends": 50, "growth": 55, "safety": 75,
         "setor": "Outros", "dy": 0.04, "divida_pl": 0.8},
    ]
    bull_rec = engine.recommend("core", assets, regime="bull")
    bear_rec = engine.recommend("core", assets, regime="bear")
    assert isinstance(bull_rec, RecommendedPortfolio)
    assert isinstance(bear_rec, RecommendedPortfolio)
