import time
from typing import Any, Dict, List, Optional


def save_report(content: str, filename: str, output_dir: str = "reports") -> str:
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _carteira_report(
    portfolio: Any, profile_name: str, regime: str,
) -> str:
    """Generate a single portfolio recommendation report."""
    lines = []
    lines.append(f"# 📊 Carteira Recomendada — {portfolio.profile}")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 📋 Resumo")
    lines.append("")
    lines.append(f"| Indicador | Valor |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Objetivo | {portfolio.objective} |")
    lines.append(f"| Regime Atual | {regime.capitalize()} |")
    lines.append(f"| Market Score | {portfolio.market_score.get('market_score', 0):.1f}/100 ({portfolio.market_score.get('market_label', '-')}) |")
    lines.append(f"| Score Medio | {portfolio.score_medio:.1f}/100 |")
    lines.append(f"| Conviction Media | {portfolio.conviction_media:.1f}/100 |")
    lines.append(f"| Total Ativos | {len(portfolio.positions)} |")
    lines.append("")

    lines.append("## 💼 Composicao da Carteira")
    lines.append("")
    lines.append("| # | Ativo | Peso | Score | Conviction | Qualidade | Valuation | Dividendos | Crescimento | Seguranca |")
    lines.append("|---|-------|------|-------|------------|-----------|-----------|------------|-------------|-----------|")
    for i, p in enumerate(portfolio.positions, 1):
        lines.append(
            f"| {i} | {p['ticker']} | {p['peso']:.1f}% | {p['score']:.1f} | "
            f"{p['conviction']:.0f} ({p['conviction_label']}) | "
            f"{p['qualidade']:.0f} | {p['valuation']:.0f} | {p['dividendos']:.0f} | "
            f"{p['crescimento']:.0f} | {p['seguranca']:.0f} |"
        )
    lines.append("")

    lines.append("## 📈 Metricas de Risco e Retorno")
    lines.append("")
    m = portfolio.metrics
    lines.append("| Metrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Retorno Anualizado | {m.get('retorno_anualizado', 0)*100:.2f}% |")
    lines.append(f"| Volatilidade | {m.get('volatilidade_anualizada', 0)*100:.2f}% |")
    lines.append(f"| Sharpe | {m.get('sharpe_ratio', 0):.2f} |")
    lines.append(f"| Sortino | {m.get('sortino_ratio', 0):.2f} |")
    lines.append(f"| Max Drawdown | {m.get('max_drawdown', 0)*100:.1f}% |")
    lines.append(f"| Alpha | {m.get('alpha', 0)*100:.2f}% |")
    lines.append(f"| Beta | {m.get('beta', 1):.2f} |")
    lines.append(f"| R² | {m.get('r_squared', 0):.2f} |")
    lines.append("")

    irp = portfolio.irp_result
    if irp:
        lines.append("## 🛡️ Indice de Robustez Patrimonial (IRP)")
        lines.append("")
        lines.append(f"| Componente | Score |")
        lines.append(f"|------------|-------|")
        lines.append(f"| IRP Total | {irp.get('IRP', 0):.1f}/100 ({irp.get('classificacao', '-')}) |")
        for k, v in irp.get("sub_scores", {}).items():
            labels = {
                "diversificacao": "Diversificacao",
                "qualidade_media": "Qualidade Media",
                "estabilidade_dividendos": "Estabilidade Dividendos",
                "baixa_alavancagem": "Baixa Alavancagem",
            }
            lines.append(f"| {labels.get(k, k)} | {v:.1f} |")
        lines.append("")

    stress = portfolio.stress_test
    if stress:
        lines.append("## ⚠️ Stress Test")
        lines.append("")
        lines.append("| Cenario | Perda | Recuperacao |")
        lines.append("|---------|-------|-------------|")
        for name, sc in stress.get("cenarios", {}).items():
            if isinstance(sc, dict) and "perda_estimada" in sc:
                rec = f"{sc.get('recuperacao_estimada_dias', 0)} dias"
                lines.append(f"| {name} | {sc['perda_estimada']:.1f}% | {rec} |")
        lines.append("")

    adv = portfolio.advanced_stress
    if adv:
        lines.append("## ⚠️ Stress Test Avancado")
        lines.append("")
        lines.append(f"| Cenario | Perda | Recuperacao |")
        lines.append(f"|---------|-------|-------------|")
        for name, sc in adv.get("cenarios", {}).items():
            if isinstance(sc, dict) and "perda_estimada" in sc:
                rec = f"{sc.get('recuperacao_estimada_dias', 0)} dias"
                lines.append(f"| {name} | {sc['perda_estimada']:.1f}% | {rec} |")
        lines.append("")

    lines.append("## 📋 Explicacoes por Ativo")
    lines.append("")
    for exp in portfolio.explicacoes:
        lines.append(f"### {exp['ticker']} — Peso: {exp['peso']:.1f}% — Conviction: {exp['conviction_label']}")
        if exp.get("motivos"):
            lines.append("**Motivos:**")
            for m in exp["motivos"]:
                lines.append(f"- {m}")
        if exp.get("pontos_fortes"):
            lines.append("**Pontos Fortes:**")
            for p in exp["pontos_fortes"]:
                lines.append(f"- {p}")
        if exp.get("riscos"):
            lines.append("**Riscos:**")
            for r in exp["riscos"]:
                lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


def generate_market_report(
    market_score: Dict[str, Any],
    regime: str,
    regime_description: str,
    consolidated: Dict[str, Any],
) -> str:
    """Generate market overview report."""
    lines = []
    lines.append("# 🌍 Relatorio de Mercado — QPE")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 📊 Score de Mercado")
    lines.append("")
    lines.append(f"| Indicador | Valor |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Market Score | {market_score.get('market_score', 0):.1f}/100 ({market_score.get('market_label', '-')}) |")
    lines.append(f"| Regime Atual | {regime.capitalize()} |")
    lines.append(f"| Ativos Analisados | {market_score.get('total_ativos_analisados', 0)} |")
    lines.append("")
    comp = market_score.get("componentes", {})
    if comp:
        lines.append("| Componente | Score |")
        lines.append("|------------|-------|")
        for k, v in comp.items():
            label = {
                "media_scores_universo": "Media Scores Universo",
                "dispersao_valuation": "Dispersao Valuation",
                "nivel_qualidade": "Nivel Qualidade",
                "regime_score": "Regime",
            }.get(k, k)
            lines.append(f"| {label} | {v:.1f} |")
        lines.append("")

    lines.append("## 🌦️ Regime de Mercado")
    lines.append("")
    lines.append(f"**Regime:** {regime.capitalize()}")
    lines.append("")
    from qpe.regime_detector import RegimeDetector
    lines.append(RegimeDetector.regime_description(regime))
    lines.append("")

    lines.append("## 🏆 Melhores Carteiras por Regime")
    lines.append("")
    regime_recommendations = {
        "bull": {"melhor": "CRESCIMENTO", "segundo": "ALPHA"},
        "bear": {"melhor": "DEFENSIVA", "segundo": "CORE"},
        "crisis": {"melhor": "DEFENSIVA", "segundo": "CORE"},
        "recovery": {"melhor": "CRESCIMENTO", "segundo": "VALOR"},
        "high_rates": {"melhor": "DIVIDENDOS", "segundo": "DEFENSIVA"},
        "low_rates": {"melhor": "VALOR", "segundo": "CRESCIMENTO"},
    }
    rec = regime_recommendations.get(regime, {"melhor": "CORE", "segundo": "ALPHA"})
    lines.append(f"**Recomendacao Principal:** {rec['melhor']}")
    lines.append(f"**Alternativa:** {rec['segundo']}")
    lines.append("")

    lines.append("## 📋 Rank de Carteiras")
    lines.append("")
    lines.append("| Perfil | Score Medio | Conviction | Sharpe | IRP | Ativos |")
    lines.append("|--------|-------------|------------|--------|-----|--------|")
    for pname, pdata in consolidated.items():
        if isinstance(pdata, dict):
            lines.append(
                f"| {pdata.get('profile', pname.upper())} | "
                f"{pdata.get('score_medio', 0):.1f} | "
                f"{pdata.get('conviction_media', 0):.1f} | "
                f"{pdata.get('sharpe', 0):.2f} | "
                f"{pdata.get('irp', 0):.1f} | "
                f"{pdata.get('num_ativos', 0)} |"
            )
    lines.append("")

    lines.append("*Relatorio gerado automaticamente pelo Quantitative Portfolio Engine*")
    return "\n".join(lines)


def generate_validation_report(
    recommendations: Dict[str, Any],
) -> str:
    """Generate recommendation engine validation report."""
    lines = []
    lines.append("# 🔬 Relatorio de Validacao — Recommendation Engine")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    by_alpha = sorted(
        recommendations.items(),
        key=lambda x: x[1].get("alpha", 0) if isinstance(x[1], dict) else 0,
        reverse=True,
    )
    by_sharpe = sorted(
        recommendations.items(),
        key=lambda x: x[1].get("sharpe", 0) if isinstance(x[1], dict) else 0,
        reverse=True,
    )
    by_dd = sorted(
        recommendations.items(),
        key=lambda x: x[1].get("drawdown", 1) if isinstance(x[1], dict) else 1,
    )
    by_cv = sorted(
        recommendations.items(),
        key=lambda x: x[1].get("conviction_media", 0) if isinstance(x[1], dict) else 0,
        reverse=True,
    )

    lines.append("## Respostas")
    lines.append("")

    best_alpha = by_alpha[0] if by_alpha else (None, {})
    lines.append(f"### 1. Qual carteira possui maior alpha esperado?")
    lines.append(f"**{best_alpha[0].upper()}** — Alpha: {best_alpha[1].get('alpha', 0)*100:.2f}%"
                 if best_alpha[0] else "N/A")
    lines.append("")

    best_sharpe = by_sharpe[0] if by_sharpe else (None, {})
    lines.append(f"### 2. Qual possui maior Sharpe?")
    lines.append(f"**{best_sharpe[0].upper()}** — Sharpe: {best_sharpe[1].get('sharpe', 0):.2f}"
                 if best_sharpe[0] else "N/A")
    lines.append("")

    best_dd = by_dd[0] if by_dd else (None, {})
    lines.append(f"### 3. Qual possui menor drawdown?")
    lines.append(f"**{best_dd[0].upper()}** — Drawdown: {best_dd[1].get('drawdown', 0)*100:.1f}%"
                 if best_dd[0] else "N/A")
    lines.append("")

    best_cv = by_cv[0] if by_cv else (None, {})
    lines.append(f"### 4. Qual possui maior Conviction media?")
    lines.append(f"**{best_cv[0].upper()}** — Conviction: {best_cv[1].get('conviction_media', 0):.1f}"
                 if best_cv[0] else "N/A")
    lines.append("")

    risk_return = sorted(
        recommendations.items(),
        key=lambda x: x[1].get("sharpe", 0) / max(x[1].get("drawdown", 0.01), 0.01)
        if isinstance(x[1], dict) else 0,
        reverse=True,
    )
    best_rr = risk_return[0] if risk_return else (None, {})
    lines.append(f"### 5. Qual possui melhor relacao risco-retorno?")
    lines.append(f"**{best_rr[0].upper()}** — Sharpe/Drawdown ratio"
                 if best_rr[0] else "N/A")
    lines.append("")

    regime_recs = {
        "bull": "CRESCIMENTO",
        "bear": "DEFENSIVA",
        "crisis": "DEFENSIVA",
        "recovery": "CRESCIMENTO",
        "high_rates": "DIVIDENDOS",
        "low_rates": "VALOR",
    }
    detected_regime = next(
        (v.get("regime", "unknown") for k, v in recommendations.items()
         if isinstance(v, dict) and v.get("regime")),
        "unknown",
    )
    main_rec = regime_recs.get(detected_regime, "CORE")
    lines.append(f"### 6. Qual e a recomendacao principal do QPE para o regime atual ({detected_regime.capitalize()})?")
    lines.append(f"**{main_rec}**")
    lines.append("")

    lines.append("## Rank Completo")
    lines.append("")
    lines.append("| Perfil | Alpha | Sharpe | Drawdown | Conviction | Score Medio | IRP |")
    lines.append("|--------|-------|--------|----------|------------|-------------|-----|")
    for name, data in sorted(
        recommendations.items(),
        key=lambda x: x[1].get("sharpe", 0) if isinstance(x[1], dict) else 0,
        reverse=True,
    ):
        d = data if isinstance(data, dict) else {}
        lines.append(
            f"| {d.get('profile', name.upper())} | {d.get('alpha', 0)*100:.2f}% | "
            f"{d.get('sharpe', 0):.2f} | {d.get('drawdown', 0)*100:.1f}% | "
            f"{d.get('conviction_media', 0):.1f} | {d.get('score_medio', 0):.1f} | "
            f"{d.get('irp', 0):.1f} |"
        )
    lines.append("")

    return "\n".join(lines)
