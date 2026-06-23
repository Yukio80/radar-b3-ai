import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class AdvancedStressTest:
    """
    Advanced macroeconomic stress scenarios for the Brazilian market.

    Scenarios:
    - Crash de Commodities: -35% commodities, -25% geral
    - Crise Fiscal Brasileira: -30% acoes, +50% volatilidade
    - Choque de Juros: SELIC +10pp, -35% acoes
    - Estagflacao: -15% acoes, +20% inflacao, 0% crescimento
    - Black Monday (1987-style): -20% em um dia
    """

    SCENARIOS = {
        "Crash Commodities": {
            "descricao": "Queda de 35% no preco de commodities (petróleo, minerio), impacto em VALE e PETROBRAS",
            "impacto_base": -0.25,
            "impacto_commodities": -0.40,
            "impacto_financeiro": -0.15,
            "impacto_utilidades": -0.10,
            "impacto_consumo": -0.20,
            "vol_multiplier": 1.5,
            "recuperacao_dias": 540,
        },
        "Crise Fiscal Brasil": {
            "descricao": "Perda do grau de investimento, disparada do risco-pais, fuga de capitais",
            "impacto_base": -0.30,
            "impacto_commodities": -0.35,
            "impacto_financeiro": -0.35,
            "impacto_utilidades": -0.20,
            "impacto_consumo": -0.30,
            "vol_multiplier": 2.0,
            "recuperacao_dias": 900,
        },
        "Choque de Juros": {
            "descricao": "SELIC sobe para 24% aa, credito contrai, consumo cai",
            "impacto_base": -0.25,
            "impacto_commodities": -0.20,
            "impacto_financeiro": -0.35,
            "impacto_utilidades": -0.15,
            "impacto_consumo": -0.30,
            "vol_multiplier": 1.8,
            "recuperacao_dias": 720,
        },
        "Estagflacao": {
            "descricao": "Inflacao alta (15%) + crescimento zero + desemprego alta",
            "impacto_base": -0.20,
            "impacto_commodities": -0.10,
            "impacto_financeiro": -0.30,
            "impacto_utilidades": -0.15,
            "impacto_consumo": -0.25,
            "vol_multiplier": 1.6,
            "recuperacao_dias": 1080,
        },
        "Black Monday Brasil": {
            "descricao": "Queda unica de 20% no Ibovespa com contagio global",
            "impacto_base": -0.20,
            "impacto_commodities": -0.22,
            "impacto_financeiro": -0.22,
            "impacto_utilidades": -0.15,
            "impacto_consumo": -0.20,
            "vol_multiplier": 3.0,
            "recuperacao_dias": 365,
        },
    }

    SECTOR_KEY_MAP = {
        "Petróleo": "impacto_commodities",
        "Mineração": "impacto_commodities",
        "Siderurgia": "impacto_commodities",
        "Agronegócio": "impacto_commodities",
        "Financeiro": "impacto_financeiro",
        "Seguros": "impacto_financeiro",
        "Utilities": "impacto_utilidades",
        "Energia Elétrica": "impacto_utilidades",
        "Saneamento": "impacto_utilidades",
        "Consumo": "impacto_consumo",
        "Varejo": "impacto_consumo",
        "Alimentos": "impacto_consumo",
        "Saúde": "impacto_consumo",
        "Educação": "impacto_consumo",
    }

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}

    def run_scenario(
        self,
        scenario_name: str,
        weights: Dict[str, float],
        sector_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Run an advanced stress scenario.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario.
        weights : dict
            Ticker -> weight percentage.
        sector_map : dict
            Ticker -> sector name.

        Returns
        -------
        dict
            Scenario impact results.
        """
        scenario = self.SCENARIOS.get(scenario_name)
        if not scenario:
            return {"error": f"Scenario not found: {scenario_name}"}

        total = sum(weights.values())
        if total <= 0:
            return {"error": "No weights to stress test"}

        total_loss = 0.0
        asset_impacts = []
        vol_impact = 0.0

        for ticker, w in weights.items():
            sector = sector_map.get(ticker, "Outros")
            impact_key = self.SECTOR_KEY_MAP.get(sector, "impacto_base")
            impact = scenario.get(impact_key, scenario["impacto_base"])

            loss = w * impact
            total_loss += loss
            vol_impact += w * (1 + scenario["vol_multiplier"] * 0.5)

            asset_impacts.append({
                "ticker": ticker,
                "setor": sector,
                "peso": w,
                "impacto": impact,
                "contribuicao": round(loss, 2),
            })

        total_loss_pct = round(total_loss, 2)
        avg_vol = round(vol_impact / len(weights), 2) if weights else 1.0

        asset_impacts.sort(key=lambda x: x["contribuicao"])

        return {
            "cenario": scenario_name,
            "descricao": scenario["descricao"],
            "perda_estimada": total_loss_pct,
            "vol_projetada": avg_vol,
            "recuperacao_estimada_dias": scenario["recuperacao_dias"],
            "recuperacao_meses": round(scenario["recuperacao_dias"] / 21, 1),
            "impacto_por_ativo": asset_impacts,
        }

    def run_all(
        self,
        weights: Dict[str, float],
        sector_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Run all advanced stress scenarios.

        Parameters
        ----------
        weights : dict
            Ticker -> weight percentage.
        sector_map : dict
            Ticker -> sector.

        Returns
        -------
        dict
            All scenario results with worst case.
        """
        results: Dict[str, Any] = {}
        for name in self.SCENARIOS:
            results[name] = self.run_scenario(name, weights, sector_map)

        worst = min(
            (r for r in results.values() if isinstance(r, dict) and "perda_estimada" in r),
            key=lambda x: x["perda_estimada"],
            default=None,
        )

        self.results = {
            "cenarios": results,
            "pior_cenario": worst["cenario"] if worst else None,
            "pior_perda": worst["perda_estimada"] if worst else 0,
            "tipo": "avancado",
            "classificacao_risco": self._classify_risk(results),
        }
        return self.results

    def _classify_risk(self, scenarios: Dict[str, Any]) -> str:
        """Classify portfolio risk based on worst scenario."""
        worst = min(
            (r for r in scenarios.values() if isinstance(r, dict) and "perda_estimada" in r),
            key=lambda x: x["perda_estimada"],
            default=None,
        )
        if worst is None:
            return "Indeterminado"
        pior = abs(worst["perda_estimada"])
        if pior > 35:
            return "Risco Extremo"
        elif pior > 25:
            return "Risco Alto"
        elif pior > 15:
            return "Risco Moderado"
        else:
            return "Risco Baixo"
