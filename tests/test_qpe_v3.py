from qpe.performance_metrics import PerformanceMetrics
from qpe.monte_carlo import MonteCarloEngine
from qpe.correlation_analysis import CorrelationAnalyzer
from qpe.regime_detector import RegimeDetector
from qpe.backtesting import BacktestEngine
from qpe.walk_forward import WalkForwardValidator
from qpe.benchmark import BenchmarkEngine
from qpe.reports import BacktestReport, ValidationReport, PerformanceReport, save_report

import numpy as np
import pandas as pd
import tempfile
import os


def test_all_modules_importable():
    assert PerformanceMetrics is not None
    assert MonteCarloEngine is not None
    assert CorrelationAnalyzer is not None
    assert RegimeDetector is not None
    assert BacktestEngine is not None
    assert WalkForwardValidator is not None
    assert BenchmarkEngine is not None
    assert BacktestReport is not None
    assert ValidationReport is not None
    assert PerformanceReport is not None


def test_benchmark_engine_creation():
    be = BenchmarkEngine(cdi_rate=0.1325)
    assert be.cdi_rate == 0.1325


def test_benchmark_cdi_series():
    be = BenchmarkEngine(cdi_rate=0.10, start_date="2023-01-01")
    cdi = be._get_cdi_series()
    assert isinstance(cdi, pd.Series)
    assert len(cdi) > 0
    assert abs(cdi.iloc[0] - ((1 + 0.10) ** (1/252) - 1)) < 1e-10


def test_benchmark_download_unknown():
    be = BenchmarkEngine()
    try:
        be.download("UNKNOWN")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_walk_forward_windows():
    from datetime import datetime, timedelta
    start = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    wf = WalkForwardValidator(
        tickers=["PETR4.SA", "VALE3.SA"],
        start_date=start,
        end_date=end,
        train_years=1,
        test_months=3,
    )
    windows = wf._generate_windows()
    assert len(windows) > 0
    for tr_s, tr_e, te_s, te_e in windows:
        assert tr_s < tr_e
        assert te_s < te_e


def test_backtest_report():
    report = BacktestReport()
    results = {
        "capital_inicial": 100000,
        "capital_final": 110000,
        "retorno_total": 10.0,
        "frequencia": "trimestral",
        "qtd_rebalances": 4,
        "trades": [
            {"data": "2023-01-01", "tickers": ["PETR4", "VALE3"], "capital": 100000},
        ],
    }
    content = report.generate(results)
    assert "QPE v3" in content
    assert "10.00%" in content
    assert "trimestral" in content


def test_validation_report():
    report = ValidationReport()
    wf_results = {
        "total_janelas": 3,
        "resultados_consolidados": {
            "retorno_medio_treino": 12.5,
            "retorno_medio_teste": 8.3,
            "mediana_retorno_teste": 7.1,
            "std_retorno_teste": 3.2,
            "min_retorno_teste": 2.1,
            "max_retorno_teste": 15.0,
            "janelas_positivas": 2,
            "taxa_acerto": 66.7,
        },
        "janelas": [
            {"janela": 1, "treino": {"inicio": "2020", "fim": "2021"},
             "teste": {"inicio": "2021", "fim": "2022"},
             "retorno_treino": 15.0, "retorno_teste": 10.0},
        ],
    }
    mc_results = {
        "num_simulacoes": 10000,
        "horizonte_dias": 252,
        "retorno_esperado": 0.12,
        "volatilidade_esperada": 0.20,
        "var_95": -0.15,
        "var_99": -0.25,
        "cvar_95": -0.22,
        "probabilidade_perda": 0.25,
        "probabilidade_superar_cdi": 0.65,
        "probabilidade_superar_ibov": 0.55,
        "retorno_medio_simulado": 0.10,
        "melhor_cenario": 0.80,
        "pior_cenario": -0.40,
    }
    content = report.generate(wf_results, mc_results)
    assert "Walk-Forward" in content
    assert "Monte Carlo" in content
    assert "66.7%" in content


def test_performance_report():
    report = PerformanceReport()
    metrics = {
        "retorno_acumulado": 0.15,
        "retorno_anualizado": 0.10,
        "volatilidade_anualizada": 0.18,
        "sharpe_ratio": 0.55,
        "sortino_ratio": 0.70,
        "calmar_ratio": 0.40,
        "max_drawdown": 0.12,
        "alpha": 0.03,
        "beta": 0.90,
        "tracking_error": 0.08,
        "information_ratio": 0.35,
    }
    content = report.generate(portfolio_metrics=metrics)
    assert "Performance" in content
    assert "0.55" in content


def test_save_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_report("# Test", "test_report.md", output_dir=tmpdir)
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read().strip() == "# Test"
