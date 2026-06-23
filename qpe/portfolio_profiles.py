from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PortfolioProfile:
    """
    Definition of a portfolio profile with screening criteria and allocation rules.

    Each profile specifies:
    - Objective and description
    - Score thresholds for screening
    - Allocation method (from portfolio_construction.py)
    - Weight and concentration constraints
    - Maximum positions
    """

    name: str
    description: str
    objective: str
    allocation_method: str
    score_thresholds: Dict[str, float]
    sort_key: str
    max_positions: int
    max_asset_weight: float
    max_sector_weight: float
    target_te: Optional[float] = None
    min_irp: float = 0.0
    required_profiles: Optional[List[str]] = None


PROFILES: Dict[str, PortfolioProfile] = {
    "core": PortfolioProfile(
        name="CORE",
        description="Carteira principal de longo prazo com alocação equilibrada",
        objective="Longo prazo, buy and hold, equilíbrio entre fatores",
        allocation_method="risk_parity",
        score_thresholds={"total_score": 60},
        sort_key="total_score",
        max_positions=20,
        max_asset_weight=0.10,
        max_sector_weight=0.25,
        min_irp=65,
    ),
    "alpha": PortfolioProfile(
        name="ALPHA",
        description="Carteira de alta convicção para superar o IBOV",
        objective="Superar o IBOV com gestão ativa de tracking error",
        allocation_method="max_sharpe",
        score_thresholds={"total_score": 50},
        sort_key="alpha_score",
        max_positions=15,
        max_asset_weight=0.08,
        max_sector_weight=0.20,
        target_te=0.08,
        min_irp=0,
    ),
    "dividendos": PortfolioProfile(
        name="DIVIDENDOS",
        description="Carteira focada em geração de renda com altos dividendos",
        objective="Geração de renda, estabilidade de proventos",
        allocation_method="risk_parity",
        score_thresholds={"dividends": 70, "quality": 70},
        sort_key="dividends",
        max_positions=15,
        max_asset_weight=0.10,
        max_sector_weight=0.25,
        min_irp=60,
    ),
    "valor": PortfolioProfile(
        name="VALOR",
        description="Carteira de empresas descontadas com potencial de valorização",
        objective="Empresas negociadas abaixo do valor intrínseco",
        allocation_method="max_sharpe",
        score_thresholds={"valuation": 70, "quality": 60},
        sort_key="valuation",
        max_positions=15,
        max_asset_weight=0.10,
        max_sector_weight=0.25,
        min_irp=0,
    ),
    "crescimento": PortfolioProfile(
        name="CRESCIMENTO",
        description="Carteira de empresas com alto potencial de expansão",
        objective="Expansão de resultados, captura de upside",
        allocation_method="max_sharpe",
        score_thresholds={"growth": 65, "quality": 50},
        sort_key="growth",
        max_positions=15,
        max_asset_weight=0.10,
        max_sector_weight=0.25,
        min_irp=0,
    ),
    "defensiva": PortfolioProfile(
        name="DEFENSIVA",
        description="Carteira de preservação de capital com baixo risco",
        objective="Preservação de capital, baixa volatilidade",
        allocation_method="min_variance",
        score_thresholds={"safety": 70, "quality": 70},
        sort_key="safety",
        max_positions=20,
        max_asset_weight=0.08,
        max_sector_weight=0.20,
        min_irp=70,
    ),
}


def get_profile(name: str) -> PortfolioProfile:
    """Get a portfolio profile by name (case-insensitive)."""
    key = name.lower().strip()
    profile = PROFILES.get(key)
    if profile is None:
        valid = list(PROFILES.keys())
        raise ValueError(f"Unknown profile: {name}. Valid options: {valid}")
    return profile


def list_profiles() -> List[str]:
    """Get list of available profile names."""
    return list(PROFILES.keys())
