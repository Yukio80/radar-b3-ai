import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class StressTest:
    """
    Simulate portfolio performance under adverse market scenarios.

    Scenarios:
    - Crise 2008: -40% market drop
    - Pandemia: -30% market drop
    - Alta de Juros: +5% interest rate impact on bond-like assets
    """

    SCENARIOS = {
        "Crise 2008": {
            "descricao": "Queda generalizada de -40% nos ativos de risco",
            "impacto_acoes": -0.40,
            "impacto_bdrs": -0.40,
            "impacto_etfs": -0.35,
            "impacto_fiis": -0.30,
            "recuperacao_dias": 720,
        },
        "Pandemia": {
            "descricao": "Queda de -30% com recuperação em V",
            "impacto_acoes": -0.30,
            "impacto_bdrs": -0.30,
            "impacto_etfs": -0.25,
            "impacto_fiis": -0.20,
            "recuperacao_dias": 360,
        },
        "Alta de Juros": {
            "descricao": "Elevação da SELIC em +5pp, impacto em FIIs e dividendo",
            "impacto_acoes": -0.15,
            "impacto_bdrs": -0.10,
            "impacto_etfs": -0.10,
            "impacto_fiis": -0.25,
            "recuperacao_dias": 180,
        },
    }

    CATEGORY_MAP = {
        "Ações": "impacto_acoes",
        "Ações Dividendos": "impacto_acoes",
        "FIIs": "impacto_fiis",
        "ETFs": "impacto_etfs",
        "BDRs": "impacto_bdrs",
    }

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}

    def run_scenario(
        self,
        scenario_name: str,
        weights: Dict[str, float],
        categories: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Run a single stress scenario on the portfolio.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario from SCENARIOS.
        weights : dict
            Ticker -> weight percentage.
        categories : dict
            Ticker -> category (Ações, FIIs, ETFs, BDRs).

        Returns
        -------
        dict
            Scenario results with total loss, per-asset impact, and recovery.
        """
        scenario = self.SCENARIOS.get(scenario_name)
        if not scenario:
            return {"error": f"Unknown scenario: {scenario_name}"}

        total_loss = 0.0
        asset_impacts: List[Dict[str, Any]] = []
        weighted_loss = 0.0

        for ticker, weight_pct in weights.items():
            category = categories.get(ticker, "Ações")
            impact_key = self.CATEGORY_MAP.get(category, "impacto_acoes")
            impact_pct = scenario.get(impact_key, -0.30)

            loss = weight_pct * impact_pct
            weighted_loss += loss
            total_loss += weight_pct

            asset_impacts.append({
                "ticker": ticker,
                "categoria": category,
                "peso": weight_pct,
                "impacto": impact_pct,
                "contribuicao_perda": round(loss, 2),
            })

        total_loss_pct = round(
            sum(a["contribuicao_perda"] for a in asset_impacts), 2
        ) if total_loss > 0 else 0

        asset_impacts.sort(key=lambda x: x["contribuicao_perda"])

        return {
            "cenario": scenario_name,
            "descricao": scenario["descricao"],
            "perda_estimada": round(total_loss_pct, 2),
            "recuperacao_estimada_dias": scenario["recuperacao_dias"],
            "drawdown_maximo": round(total_loss_pct, 2),
            "impacto_por_ativo": asset_impacts,
        }

    def run_all(
        self,
        weights: Dict[str, float],
        categories: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Run all stress scenarios.

        Parameters
        ----------
        weights : dict
            Ticker -> weight percentage.
        categories : dict
            Ticker -> category string.

        Returns
        -------
        dict
            All scenario results with worst-case summary.
        """
        results: Dict[str, Any] = {}
        for name in self.SCENARIOS:
            results[name] = self.run_scenario(name, weights, categories)

        worst = min(
            (r for r in results.values() if isinstance(r, dict) and "perda_estimada" in r),
            key=lambda x: x["perda_estimada"],
            default=None,
        )

        self.results = {
            "cenarios": results,
            "pior_cenario": worst["cenario"] if worst else None,
            "pior_perda": worst["perda_estimada"] if worst else 0,
        }
        return self.results

    def recovery_analysis(
        self,
        scenario_result: Dict[str, Any],
        daily_return: float = 0.0008,
    ) -> Dict[str, Any]:
        """
        Estimate recovery time after a shock.

        Parameters
        ----------
        scenario_result : dict
            Result from run_scenario.
        daily_return : float, default=0.0008
            Assumed daily return during recovery (0.08%).

        Returns
        -------
        dict
            Recovery time and monthly projections.
        """
        loss = abs(scenario_result.get("perda_estimada", 0))
        if loss <= 0 or daily_return <= 0:
            return {"dias_para_recuperar": 0, "meses_para_recuperar": 0}

        days = int(np.ceil(np.log(1 + loss / 100) / np.log(1 + daily_return)))
        months = round(days / 21, 1)

        return {
            "dias_para_recuperar": days,
            "meses_para_recuperar": months,
            "daily_return_assumido": daily_return,
        }
