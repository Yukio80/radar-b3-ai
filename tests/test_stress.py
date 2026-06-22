from qpe.stress_test import StressTest


def test_run_scenario():
    st = StressTest()
    result = st.run_scenario(
        "Crise 2008",
        {"A": 50, "B": 50},
        {"A": "Ações", "B": "FIIs"},
    )
    assert "perda_estimada" in result
    assert result["perda_estimada"] < 0
    assert len(result["impacto_por_ativo"]) == 2


def test_run_all():
    st = StressTest()
    results = st.run_all(
        {"A": 50, "B": 50},
        {"A": "Ações", "B": "FIIs"},
    )
    assert "cenarios" in results
    assert "pior_cenario" in results
    assert len(results["cenarios"]) == 3


def test_recovery():
    st = StressTest()
    scenario = st.run_scenario(
        "Pandemia",
        {"A": 100},
        {"A": "Ações"},
    )
    recovery = st.recovery_analysis(scenario)
    assert "dias_para_recuperar" in recovery
    assert recovery["dias_para_recuperar"] > 0
