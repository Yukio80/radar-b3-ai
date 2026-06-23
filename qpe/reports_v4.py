import time
from typing import Any, Dict, List, Optional


def save_report(content: str, filename: str, output_dir: str = "reports") -> str:
    """Save a report to file."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def generate_alpha_report(validation_data: Dict[str, Any]) -> str:
    """Generate alpha generation report."""
    lines = []
    lines.append("# 🎯 Relatório de Geração de Alfa — QPE v4")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    pf = validation_data.get("performance", {})
    lines.append("## 📊 Performance Final")
    lines.append("")
    lines.append("| Métrica | QPE v4 | IBOV | CDI |")
    lines.append("|---------|--------|------|-----|")
    bench = validation_data.get("benchmark", {})
    ibov = bench.get("ibov", {})
    cdi = bench.get("cdi", {})

    for metric, label, is_pct in [
        ("retorno_anualizado", "Retorno Anualizado", True),
        ("sharpe_ratio", "Sharpe", False),
        ("sortino_ratio", "Sortino", False),
        ("max_drawdown", "Max Drawdown", True),
        ("volatilidade_anualizada", "Volatilidade", True),
    ]:
        val_qpe = pf.get(metric, 0) or 0
        val_ibov = ibov.get(metric, 0) or 0
        val_cdi = cdi.get(metric, 0) or 0
        if is_pct:
            lines.append(f"| {label} | {val_qpe*100:.2f}% | {val_ibov*100:.2f}% | {val_cdi*100:.2f}% |")
        else:
            lines.append(f"| {label} | {val_qpe:.2f} | {val_ibov:.2f} | {val_cdi:.2f} |")

    lines.append("")
    lines.append("## 🧪 Alpha e Risco Relativo")
    lines.append("")
    lines.append(f"| Métrica | Valor |")
    lines.append(f"|---------|-------|")
    lines.append(f"| Alpha | {pf.get('alpha', 0)*100:.2f}% |")
    lines.append(f"| Beta | {pf.get('beta', 1):.2f} |")
    lines.append(f"| Tracking Error | {pf.get('tracking_error', 0)*100:.2f}% |")
    lines.append(f"| Information Ratio | {pf.get('information_ratio', 0):.2f} |")
    lines.append(f"| R² | {pf.get('r_squared', 0):.2f} |")
    lines.append("")

    mc = validation_data.get("monte_carlo", {})
    if mc:
        lines.append("## 🎲 Significância Estatística (Monte Carlo)")
        lines.append("")
        prob_superar_ibov = mc.get("probabilidade_superar_ibov", 0)
        prob_superar_cdi = mc.get("probabilidade_superar_cdi", 0)
        prob_perda = mc.get("probabilidade_perda", 0)
        lines.append(f"| Métrica | Valor |")
        lines.append(f"|---------|-------|")
        lines.append(f"| Prob. > IBOV | {prob_superar_ibov*100:.1f}% |")
        lines.append(f"| Prob. > CDI | {prob_superar_cdi*100:.1f}% |")
        lines.append(f"| Prob. de Perda | {prob_perda*100:.1f}% |")
        lines.append(f"| VaR 95% | {mc.get('var_95', 0)*100:.1f}% |")
        lines.append(f"| VaR 99% | {mc.get('var_99', 0)*100:.1f}% |")
        lines.append("")

    alpha = pf.get("alpha", 0)
    sharpe = pf.get("sharpe_ratio", 0)
    ibov_sharpe = ibov.get("sharpe_ratio", 0)
    info_ratio = pf.get("information_ratio", 0)
    mdd = pf.get("max_drawdown", 0)
    ibov_mdd = ibov.get("max_drawdown", 0)

    lines.append("## ✅ Verificação dos Critérios de Sucesso")
    lines.append("")
    criteria = [
        ("Alpha > 0", alpha > 0, f"{alpha*100:.2f}%"),
        ("Sharpe > IBOV", sharpe > ibov_sharpe, f"QPE={sharpe:.2f} vs IBOV={ibov_sharpe:.2f}"),
        ("Drawdown <= IBOV", mdd <= ibov_mdd * 1.1, f"QPE={mdd*100:.1f}% vs IBOV={ibov_mdd*100:.1f}%"),
        ("Information Ratio > 0.30", info_ratio > 0.30, f"{info_ratio:.2f}"),
    ]
    approved = 0
    for criterion, passed, value in criteria:
        icon = "✅" if passed else "❌"
        lines.append(f"{icon} **{criterion}:** {value}")
        if passed:
            approved += 1
    lines.append("")
    lines.append(f"**Critérios aprovados: {approved}/{len(criteria)}**")
    lines.append("")

    if approved >= 3:
        lines.append("🏆 **Conclusão: QPE v4 apresenta evidência de geração de alfa ajustado ao risco.**")
    elif approved >= 2:
        lines.append("⚠️ **Conclusão: QPE v4 apresenta alfa parcial, mas requer ajustes.**")
    else:
        lines.append("❌ **Conclusão: QPE v4 nao apresenta evidencia significativa de alfa.**")

    return "\n".join(lines)


def generate_optimization_report(
    stage1: Dict[str, Any],
    stage2: Dict[str, Any],
    method_comparison: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate portfolio optimization report."""
    lines = []
    lines.append("# ⚙️ Relatório de Otimização — QPE v4")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 📋 Estágio 1 — Seleção")
    lines.append("")
    lines.append(f"| Indicador | Valor |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Universo Total | {stage1.get('universo_total', 0)} |")
    lines.append(f"| Selecionados (Top-K) | {stage1.get('selecionados', 0)} |")
    lines.append(f"| K Configurado | {stage1.get('top_k', 30)} |")
    lines.append("")

    lines.append("## 📋 Estágio 2 — Otimização")
    lines.append("")
    lines.append(f"| Indicador | Valor |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Método | {stage2.get('metodo', '-')} |")
    lines.append(f"| Covariância | {stage2.get('metodo_covariancia', '-')} |")
    lines.append(f"| Shrinkage | {stage2.get('shrinkage', 0):.3f} |")
    lines.append(f"| Retorno Esperado | {stage2.get('retorno_esperado', 0):.2f}% |")
    lines.append(f"| Vol Esperada | {stage2.get('vol_esperada', 0):.2f}% |")
    lines.append(f"| Sharpe Esperado | {stage2.get('sharpe_esperado', 0):.2f} |")
    lines.append(f"| Ativos Ativos | {stage2.get('ativos_ativos', 0)} |")
    lines.append("")

    pesos = stage2.get("pesos", {})
    if pesos:
        sorted_w = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
        lines.append("## 💼 Alocação Final")
        lines.append("")
        lines.append("| # | Ativo | Peso |")
        lines.append("|---|-------|------|")
        for i, (t, w) in enumerate(sorted_w[:20], 1):
            if w > 0.001:
                lines.append(f"| {i} | {t} | {w*100:.2f}% |")
        lines.append("")

    if method_comparison:
        lines.append("## 📊 Comparação de Métodos")
        lines.append("")
        lines.append("| Método | Retorno | Vol | Sharpe | Ativos |")
        lines.append("|--------|---------|-----|--------|--------|")
        for m in method_comparison:
            lines.append(
                f"| {m.get('metodo', '-')} | {m.get('retorno', 0):.2f}% | "
                f"{m.get('vol', 0):.2f}% | {m.get('sharpe', 0):.2f} | "
                f"{m.get('ativos', 0)} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_regime_report(
    regime_result: Dict[str, Any],
    factor_weights: Dict[str, float],
    base_weights: Dict[str, float],
) -> str:
    """Generate regime-aware allocation report."""
    lines = []
    lines.append("# 🌦️ Relatório de Regime — QPE v4")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    regime = regime_result.get("classificacao", "Indefinido")
    lines.append(f"## Regime Atual: {regime}")
    lines.append(f"**Confianca:** {regime_result.get('confianca', 0)*100:.0f}%")
    lines.append("")

    metricas = regime_result.get("metricas", {})
    if metricas:
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Retorno 12m | {metricas.get('retorno_anualizado', 0)*100:.2f}% |")
        lines.append(f"| Vol 12m | {metricas.get('volatilidade_anualizada', 0)*100:.2f}% |")
        lines.append(f"| Sharpe 12m | {metricas.get('sharpe_12m', 0):.2f} |")
        lines.append(f"| Max Drawdown | {metricas.get('max_drawdown', 0)*100:.2f}% |")
        lines.append("")

    lines.append("## ⚖️ Ajuste de Pesos dos Fatores")
    lines.append("")
    lines.append("| Fator | Peso Base | Peso Ajustado | Delta |")
    lines.append("|-------|-----------|---------------|-------|")
    for factor in ["quality", "valuation", "dividends", "growth", "safety"]:
        base = base_weights.get(factor, 0) * 100
        adj = factor_weights.get(factor, 0) * 100
        delta = adj - base
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        lines.append(f"| {factor.capitalize()} | {base:.1f}% | {adj:.1f}% | {delta_str} |")
    lines.append("")

    from qpe.regime_detector import RegimeDetector
    lines.append("## 📝 Descrição")
    lines.append("")
    lines.append(RegimeDetector.regime_description(regime_result.get("regime", "unknown")))

    return "\n".join(lines)


def generate_attribution_report(attribution_result: Dict[str, Any]) -> str:
    """Generate alpha attribution report."""
    lines = []
    lines.append("# 📊 Relatório de Atribuição — QPE v4")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Fatores de Alpha")
    lines.append("")
    lines.append(f"**Retorno da Carteira:** {attribution_result.get('retorno_real_carteira', 0)*100:.2f}%")
    lines.append(f"**Alpha Total Atribuído:** {attribution_result.get('total_contribuicao_alpha', 0)*100:.2f}%")
    lines.append("")

    lines.append("| Fator | Peso | Contrib. Retorno | Contrib. Alpha | t-stat | Significante |")
    lines.append("|-------|------|-----------------|---------------|--------|-------------|")
    for c in attribution_result.get("fatores", []):
        sig = "✅" if getattr(c, 'significant', False) else "❌"
        weight = getattr(c, 'weight', 0) * 100
        ret_c = getattr(c, 'contribution_return', 0) * 100
        alpha_c = getattr(c, 'contribution_alpha', 0) * 100
        t_stat = getattr(c, 't_stat', 0)
        lines.append(f"| {getattr(c, 'factor', '-').capitalize()} | {weight:.1f}% | {ret_c:.2f}% | {alpha_c:.2f}% | {t_stat:.2f} | {sig} |")
    lines.append("")

    lines.append(f"**Melhor Fator:** {attribution_result.get('melhor_fator', '-')}")
    lines.append(f"**Pior Fator:** {attribution_result.get('pior_fator', '-')}")
    lines.append(f"**Fatores Significativos:** {attribution_result.get('fatores_significativos', 0)}/5")

    return "\n".join(lines)


def generate_v4_validation(validation_data: Dict[str, Any]) -> str:
    """Generate comprehensive QPE v4 validation report."""
    lines = []
    lines.append("# 🔬 Relatório de Validação — QPE v4")
    lines.append("")
    lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    pf = validation_data.get("performance", {})
    lines.append("## 1️⃣ Performance Final")
    lines.append("")
    lines.append("| Métrica | QPE v4 | IBOV | IDIV | CDI |")
    lines.append("|---------|--------|------|------|-----|")
    bench = validation_data.get("benchmark", {})
    for name, metric, is_pct in [
        ("retorno_anualizado", "Retorno Anualizado", True),
        ("sharpe_ratio", "Sharpe", False),
        ("sortino_ratio", "Sortino", False),
        ("max_drawdown", "Max Drawdown", True),
    ]:
        qpe = pf.get(name, 0) or 0
        ibov = bench.get("ibov", {}).get(name, 0) or 0
        idiv = bench.get("idiv", {}).get(name, 0) or 0
        cdi = bench.get("cdi", {}).get(name, 0) or 0
        if is_pct:
            lines.append(f"| {metric} | {qpe*100:.2f}% | {ibov*100:.2f}% | {idiv*100:.2f}% | {cdi*100:.2f}% |")
        else:
            lines.append(f"| {metric} | {qpe:.2f} | {ibov:.2f} | {idiv:.2f} | {cdi:.2f} |")
    lines.append("")

    lines.append("## 2️⃣ Estatísticas de Risco")
    lines.append("")
    lines.append(f"| Métrica | Valor |")
    lines.append(f"|---------|-------|")
    lines.append(f"| Alpha | {pf.get('alpha', 0)*100:.2f}% |")
    lines.append(f"| Beta | {pf.get('beta', 1):.2f} |")
    lines.append(f"| Tracking Error | {pf.get('tracking_error', 0)*100:.2f}% |")
    lines.append(f"| Information Ratio | {pf.get('information_ratio', 0):.2f} |")
    lines.append(f"| R² | {pf.get('r_squared', 0):.2f} |")
    lines.append("")

    wf = validation_data.get("walk_forward", {})
    if wf:
        lines.append("## 3️⃣ Walk-Forward")
        lines.append("")
        lines.append(f"| Indicador | Valor |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Retorno Medio Teste | {wf.get('retorno_medio_teste', 0):.2f}% |")
        lines.append(f"| Taxa de Acerto | {wf.get('taxa_acerto', 0):.1f}% |")
        lines.append(f"| Janelas Positivas | {wf.get('janelas_positivas', 0)}/{validation_data.get('walk_forward', {}).get('total_janelas', 0)} |")
        lines.append("")

    mc = validation_data.get("monte_carlo", {})
    if mc:
        lines.append("## 4️⃣ Monte Carlo")
        lines.append("")
        lines.append(f"| Métrica | Valor |")
        lines.append(f"|---------|-------|")
        lines.append(f"| VaR 95% | {mc.get('var_95', 0)*100:.1f}% |")
        lines.append(f"| VaR 99% | {mc.get('var_99', 0)*100:.1f}% |")
        lines.append(f"| Prob. Perda | {mc.get('probabilidade_perda', 0)*100:.1f}% |")
        lines.append(f"| Prob. > CDI | {mc.get('probabilidade_superar_cdi', 0)*100:.1f}% |")
        lines.append(f"| Prob. > IBOV | {mc.get('probabilidade_superar_ibov', 0)*100:.1f}% |")
        lines.append("")

    regime = validation_data.get("regime", "N/A")
    lines.append(f"## 5️⃣ Regime Atual: {regime}")
    lines.append("")

    stress = validation_data.get("advanced_stress", {})
    if stress:
        lines.append("## 6️⃣ Stress Test Avançado")
        lines.append("")
        lines.append(f"| Cenário | Perda | Recuperação |")
        lines.append(f"|---------|-------|-------------|")
        for name, sc in stress.get("cenarios", {}).items():
            if isinstance(sc, dict) and "perda_estimada" in sc:
                rec = f"{sc.get('recuperacao_estimada_dias', 0)} dias"
                lines.append(f"| {name} | {sc['perda_estimada']:.1f}% | {rec} |")
        lines.append("")

    return "\n".join(lines)
