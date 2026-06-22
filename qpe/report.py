import os
import time
from typing import Any, Dict, List, Optional


class PortfolioReport:
    """
    Generate a detailed markdown report for a QPE portfolio.

    The report includes:
    - Executive summary
    - Portfolio allocation
    - Top assets analysis
    - Sector concentration
    - Risk assessment (stress test)
    - Robustness index
    - Rebalance recommendations
    """

    def __init__(self, output_dir: str = ".") -> None:
        self.output_dir = output_dir

    def generate(
        self,
        portfolio: Dict[str, Any],
        irp_result: Dict[str, Any],
        stress_results: Dict[str, Any],
        explanations: List[Dict[str, Any]],
        profile_name: str = "Personalizado",
    ) -> str:
        """
        Generate a complete portfolio report.

        Parameters
        ----------
        portfolio : dict
            Portfolio data with weights and assets.
        irp_result : dict
            IRP computation result.
        stress_results : dict
            Stress test results.
        explanations : list of dict
            Per-asset explanations.
        profile_name : str
            Portfolio profile name.

        Returns
        -------
        str
            Full markdown report content.
        """
        lines: List[str] = []
        lines.append(f"# 📊 Relatório de Carteira — {profile_name}")
        lines.append("")
        lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## 📋 Resumo Executivo")
        lines.append("")

        weights = portfolio.get("weights", {})
        assets = portfolio.get("assets", [])
        total_assets = len(weights)
        avg_score = portfolio.get("score_medio", 0)
        dy_weighted = portfolio.get("dy_ponderado", 0)

        lines.append(f"| Indicador | Valor |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Total de Ativos | {total_assets} |")
        lines.append(f"| Score Médio | {avg_score:.1f}/100 |")
        lines.append(f"| DY Ponderado | {dy_weighted:.2f}% |")

        if irp_result:
            lines.append(f"| IRP | {irp_result.get('IRP', 0):.1f}/100 |")
            lines.append(f"| Classificação IRP | {irp_result.get('classificacao', '-')} |")

        lines.append("")

        # Top 10 Assets
        lines.append("## 🏆 Top 10 Ativos")
        lines.append("")
        sorted_assets = sorted(
            explanations,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )[:10]

        lines.append("| # | Ticker | Score | Motivos |")
        lines.append("|---|--------|-------|---------|")
        for i, a in enumerate(sorted_assets, 1):
            motivos = a.get("pontos_fortes", a.get("motivos", []))[:3]
            motivos_str = "; ".join(motivos) if motivos else "-"
            lines.append(
                f"| {i} | {a.get('ticker', '-')} | {a.get('score', 0):.1f} | "
                f"{motivos_str} |"
            )
        lines.append("")

        # Allocation
        lines.append("## 💼 Alocação por Classe")
        lines.append("")
        allocation = portfolio.get("alocacao", {})
        if allocation:
            lines.append("| Classe | Percentual |")
            lines.append("|--------|------------|")
            for cat, pct in allocation.items():
                lines.append(f"| {cat} | {pct}% |")
        lines.append("")

        # Sector Concentration
        lines.append("## 🏭 Concentração Setorial")
        lines.append("")
        sectors: Dict[str, float] = {}
        for a in assets:
            sector = a.get("setor", "Outros")
            weight = weights.get(a.get("ticker", ""), 0)
            sectors[sector] = sectors.get(sector, 0) + weight

        if sectors:
            sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
            lines.append("| Setor | Peso |")
            lines.append("|-------|------|")
            for sector, pct in sorted_sectors:
                lines.append(f"| {sector} | {pct:.1f}% |")
        lines.append("")

        # Risk Assessment
        lines.append("## ⚠️ Análise de Risco (Stress Test)")
        lines.append("")
        scenarios = stress_results.get("cenarios", {})
        if scenarios:
            lines.append("| Cenário | Perda Estimada | Recuperação |")
            lines.append("|---------|---------------|-------------|")
            for name, result in scenarios.items():
                if isinstance(result, dict) and "perda_estimada" in result:
                    loss = result["perda_estimada"]
                    recovery = f"{result.get('recuperacao_estimada_dias', 0)} dias"
                    lines.append(f"| {name} | {loss:.1f}% | {recovery} |")

            worst = stress_results.get("pior_cenario", "")
            lines.append("")
            lines.append(f"**Pior cenário:** {worst}")
        lines.append("")

        # Robustness
        lines.append("## 🛡️ Índice de Robustez Patrimonial (IRP)")
        lines.append("")
        if irp_result:
            lines.append(f"**IRP:** {irp_result.get('IRP', 0):.1f}/100")
            lines.append(f"**Classificação:** {irp_result.get('classificacao', '-')}")
            lines.append("")
            sub = irp_result.get("sub_scores", {})
            if sub:
                lines.append("| Componente | Score |")
                lines.append("|------------|-------|")
                for k, v in sub.items():
                    labels = {
                        "diversificacao": "Diversificação",
                        "qualidade_media": "Qualidade Média",
                        "estabilidade_dividendos": "Estabilidade Dividendos",
                        "baixa_alavancagem": "Baixa Alavancagem",
                    }
                    label = labels.get(k, k)
                    lines.append(f"| {label} | {v:.1f} |")
        lines.append("")

        # Risks
        lines.append("## 🔴 Principais Riscos Identificados")
        lines.append("")
        risks = []
        if irp_result:
            sub = irp_result.get("sub_scores", {})
            for k, v in sub.items():
                if v < 50:
                    risks.append(f"- **{k}** com score {v:.1f} — abaixo do ideal")
        if not risks:
            risks.append("- Nenhum risco crítico identificado")
        for r in risks:
            lines.append(r)
        lines.append("")

        # Rebalance
        lines.append("## 🔄 Rebalanceamento Sugerido")
        lines.append("")
        if total_assets > 0:
            max_weight = max(weights.values()) if weights else 0
            min_weight = min(weights.values()) if weights else 0
            if max_weight > 10:
                lines.append(
                    f"- Reduzir exposição em ativos com peso > 10% "
                    f"(atual: {max_weight:.1f}%)"
                )
            if total_assets < 10:
                lines.append(
                    f"- Aumentar número de ativos (atual: {total_assets}, "
                    f"mínimo recomendado: 10)"
                )
            lines.append(
                "- Reavaliar scores a cada 3 meses para capturar mudanças "
                "fundamentalistas"
            )
            lines.append(
                "- Ajustar alocação conforme mudanças no perfil de risco "
                "do investidor"
            )
        lines.append("")

        # Methodology
        lines.append("## 📐 Metodologia")
        lines.append("")
        lines.append(
            "Este relatório foi gerado pelo **Quantitative Portfolio Engine v2**, "
            "que utiliza um modelo multifatorial com 5 pilares:"
        )
        lines.append("")
        lines.append("- **Qualidade** (25%): ROE, ROIC, Margem Líquida")
        lines.append("- **Valuation** (20%): P/L, P/VP, EV/EBITDA")
        lines.append("- **Dividendos** (20%): DY, Consistência")
        lines.append("- **Crescimento** (20%): CAGR Receita, CAGR Lucro")
        lines.append("- **Segurança** (15%): Endividamento, Liquidez")
        lines.append("")
        lines.append(
            "Os scores passam por normalização percentílica para garantir "
            "distribuição adequada entre as categorias."
        )
        lines.append(
            "O IRP combina diversificação, qualidade, estabilidade de "
            "dividendos e alavancagem."
        )

        return "\n".join(lines)

    def save(
        self,
        content: str,
        filename: str = "portfolio_report.md",
    ) -> str:
        """
        Save the report to a file.

        Parameters
        ----------
        content : str
            Markdown content.
        filename : str
            Output filename.

        Returns
        -------
        str
            Full path to the saved file.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
