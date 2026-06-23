import time
from typing import Any, Dict, List, Optional

import numpy as np


class BacktestReport:
    """Generate backtest summary report in markdown."""

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir

    def generate(self, backtest_results: Dict[str, Any], benchmark_results: Optional[Dict[str, Any]] = None) -> str:
        lines: List[str] = []
        lines.append("# 📈 Relatório de Backtest — QPE v3")
        lines.append("")
        lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if "error" in backtest_results:
            lines.append(f"**Erro:** {backtest_results['error']}")
            return "\n".join(lines)

        lines.append("## 📋 Sumário Executivo")
        lines.append("")
        lines.append("| Indicador | Valor |")
        lines.append("|-----------|-------|")
        lines.append(f"| Capital Inicial | R$ {backtest_results.get('capital_inicial', 0):,.2f} |")
        lines.append(f"| Capital Final | R$ {backtest_results.get('capital_final', 0):,.2f} |")
        lines.append(f"| Retorno Total | {backtest_results.get('retorno_total', 0):.2f}% |")
        lines.append(f"| Frequência de Rebalanceamento | {backtest_results.get('frequencia', '-')} |")
        lines.append(f"| Número de Rebalanceamentos | {backtest_results.get('qtd_rebalances', 0)} |")
        lines.append("")

        trades = backtest_results.get("trades", [])
        if trades:
            lines.append("## 💼 Histórico de Trades")
            lines.append("")
            lines.append("| # | Data | Ativos | Capital |")
            lines.append("|---|------|--------|---------|")
            for i, t in enumerate(trades, 1):
                tickers_str = ", ".join(t.get("tickers", [])[:5])
                if len(t.get("tickers", [])) > 5:
                    tickers_str += f" ... (+{len(t['tickers'])-5})"
                lines.append(f"| {i} | {t.get('data', '-')} | {tickers_str} | R$ {t.get('capital', 0):,.2f} |")
            lines.append("")

        if benchmark_results:
            lines.append("## 📊 Comparação com Benchmarks")
            lines.append("")
            lines.append("| Métrica | QPE | IBOV | IDIV | CDI |")
            lines.append("|---------|-----|------|------|-----|")
            qpe = benchmark_results.get("qpe", {})
            ibov = benchmark_results.get("IBOV", {})
            idiv = benchmark_results.get("IDIV", {})
            cdi = benchmark_results.get("CDI", {})

            lines.append(f"| Retorno Acumulado | {qpe.get('retorno_acumulado', 0)*100:.2f}% | {ibov.get('retorno_acumulado', 0)*100:.2f}% | {idiv.get('retorno_acumulado', 0)*100:.2f}% | {cdi.get('retorno_acumulado', 0)*100:.2f}% |")
            lines.append(f"| Sharpe | {qpe.get('sharpe_ratio', 0):.2f} | {ibov.get('sharpe_ratio', 0):.2f} | {idiv.get('sharpe_ratio', 0):.2f} | {cdi.get('sharpe_ratio', 0):.2f} |")
            lines.append(f"| Max Drawdown | {qpe.get('max_drawdown', 0)*100:.2f}% | {ibov.get('max_drawdown', 0)*100:.2f}% | {idiv.get('max_drawdown', 0)*100:.2f}% | {cdi.get('max_drawdown', 0)*100:.2f}% |")
            lines.append(f"| Volatilidade | {qpe.get('volatilidade_anualizada', 0)*100:.2f}% | {ibov.get('volatilidade_anualizada', 0)*100:.2f}% | {idiv.get('volatilidade_anualizada', 0)*100:.2f}% | - |")
            lines.append("")

            if "alpha" in qpe:
                lines.append("## 🎯 Geração de Alfa")
                lines.append("")
                lines.append(f"| Métrica | Valor |")
                lines.append(f"|---------|-------|")
                lines.append(f"| Alpha | {qpe.get('alpha', 0)*100:.2f}% |")
                lines.append(f"| Beta | {qpe.get('beta', 1):.2f} |")
                lines.append(f"| R² | {qpe.get('r_squared', 0):.2f} |")
                lines.append(f"| Tracking Error | {qpe.get('tracking_error', 0)*100:.2f}% |")
                lines.append(f"| Information Ratio | {qpe.get('information_ratio', 0):.2f} |")
                lines.append("")

        lines.append("## ⚠️ Observações")
        lines.append("")
        lines.append("- Resultados baseados em dados históricos disponíveis via Yahoo Finance.")
        lines.append("- Performance passada não garante resultados futuros.")
        lines.append("- Custos de transação estimados em 0.1% por trade.")
        lines.append("- Rebalanceamento segue calendário de dias úteis.")

        return "\n".join(lines)


class ValidationReport:
    """Generate validation report with walk-forward and Monte Carlo results."""

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir

    def generate(self, walk_forward_results: Dict[str, Any], monte_carlo_results: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("# 🔬 Relatório de Validação — QPE v3")
        lines.append("")
        lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## 1️⃣ Walk-Forward Validation")
        lines.append("")
        wf = walk_forward_results.get("resultados_consolidados", {})
        if wf:
            lines.append("| Indicador | Valor |")
            lines.append("|-----------|-------|")
            lines.append(f"| Total de Janelas | {walk_forward_results.get('total_janelas', 0)} |")
            lines.append(f"| Retorno Médio (Treino) | {wf.get('retorno_medio_treino', 0):.2f}% |")
            lines.append(f"| Retorno Médio (Teste) | {wf.get('retorno_medio_teste', 0):.2f}% |")
            lines.append(f"| Mediana Retorno (Teste) | {wf.get('mediana_retorno_teste', 0):.2f}% |")
            lines.append(f"| Desvio Padrão (Teste) | {wf.get('std_retorno_teste', 0):.2f}% |")
            lines.append(f"| Melhor Janela (Teste) | {wf.get('max_retorno_teste', 0):.2f}% |")
            lines.append(f"| Pior Janela (Teste) | {wf.get('min_retorno_teste', 0):.2f}% |")
            lines.append(f"| Janelas Positivas | {wf.get('janelas_positivas', 0)}/{walk_forward_results.get('total_janelas', 0)} |")
            lines.append(f"| Taxa de Acerto | {wf.get('taxa_acerto', 0):.1f}% |")
            lines.append("")

            janelas = walk_forward_results.get("janelas", [])
            if janelas:
                lines.append("### Detalhamento por Janela")
                lines.append("")
                lines.append("| Janela | Treino | Teste | Retorno Treino | Retorno Teste |")
                lines.append("|--------|--------|-------|----------------|---------------|")
                for j in janelas:
                    lines.append(f"| {j['janela']} | {j['treino']['inicio']} → {j['treino']['fim']} | {j['teste']['inicio']} → {j['teste']['fim']} | {j['retorno_treino']:.1f}% | {j['retorno_teste']:.1f}% |")
                lines.append("")

        lines.append("## 2️⃣ Monte Carlo Simulation")
        lines.append("")
        mc = monte_carlo_results
        if mc:
            lines.append(f"| Indicador | Valor |")
            lines.append(f"|-----------|-------|")
            lines.append(f"| Número de Simulações | {mc.get('num_simulacoes', 0):,} |")
            lines.append(f"| Horizonte | {mc.get('horizonte_dias', 0)} dias |")
            lines.append(f"| Retorno Esperado | {mc.get('retorno_esperado', 0)*100:.2f}% |")
            lines.append(f"| Volatilidade Esperada | {mc.get('volatilidade_esperada', 0)*100:.2f}% |")
            lines.append(f"| VaR 95% | {mc.get('var_95', 0)*100:.2f}% |")
            lines.append(f"| VaR 99% | {mc.get('var_99', 0)*100:.2f}% |")
            lines.append(f"| CVaR 95% | {mc.get('cvar_95', 0)*100:.2f}% |")
            lines.append(f"| Probabilidade de Perda | {mc.get('probabilidade_perda', 0)*100:.1f}% |")
            lines.append(f"| Prob. Superar CDI | {mc.get('probabilidade_superar_cdi', 0)*100:.1f}% |")
            lines.append(f"| Prob. Superar IBOV | {mc.get('probabilidade_superar_ibov', 0)*100:.1f}% |")
            lines.append(f"| Retorno Médio Simulado | {mc.get('retorno_medio_simulado', 0)*100:.2f}% |")
            lines.append(f"| Melhor Cenário | {mc.get('melhor_cenario', 0)*100:.2f}% |")
            lines.append(f"| Pior Cenário | {mc.get('pior_cenario', 0)*100:.2f}% |")
            lines.append("")

        lines.append("## 3️⃣ Conclusões")
        lines.append("")
        if wf:
            taxa = wf.get('taxa_acerto', 0)
            ret_teste = wf.get('retorno_medio_teste', 0)
            if taxa > 60 and ret_teste > 0:
                lines.append("✅ **Walk-Forward:** O modelo apresenta consistência fora da amostra, "
                             f"com taxa de acerto de {taxa:.1f}% e retorno médio positivo de {ret_teste:.2f}%.")
            elif ret_teste > 0:
                lines.append("⚠️ **Walk-Forward:** Resultados positivos, mas com baixa consistência. "
                             "Recomenda-se aumentar o período de treino.")
            else:
                lines.append("❌ **Walk-Forward:** O modelo não apresenta robustez fora da amostra. "
                             "Revisar fatores e hiperparâmetros.")

        if mc:
            prob_loss = mc.get('probabilidade_perda', 0)
            var95 = mc.get('var_95', 0)
            if prob_loss < 0.3 and var95 > -0.20:
                lines.append("✅ **Monte Carlo:** Baixa probabilidade de perda e VaR dentro do aceitável.")
            elif prob_loss < 0.5:
                lines.append("⚠️ **Monte Carlo:** Risco moderado. Considere aumentar diversificação.")
            else:
                lines.append("❌ **Monte Carlo:** Alta probabilidade de perda. Revisar alocação.")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Relatório gerado automaticamente pelo QPE v3 — Quantitative Portfolio Engine*")

        return "\n".join(lines)


class PerformanceReport:
    """Generate detailed performance analysis report."""

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir

    def generate(self, portfolio_metrics: Dict[str, Any],
                 regime_analysis: Optional[Dict[str, Any]] = None,
                 correlation_analysis: Optional[Dict[str, Any]] = None) -> str:
        lines: List[str] = []
        lines.append("# 📊 Relatório de Performance — QPE v3")
        lines.append("")
        lines.append(f"**Gerado em:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if portfolio_metrics:
            lines.append("## 📈 Métricas de Performance")
            lines.append("")
            lines.append("| Métrica | Valor |")
            lines.append("|---------|-------|")
            lines.append(f"| Retorno Acumulado | {portfolio_metrics.get('retorno_acumulado', 0)*100:.2f}% |")
            lines.append(f"| Retorno Anualizado | {portfolio_metrics.get('retorno_anualizado', 0)*100:.2f}% |")
            lines.append(f"| Volatilidade Anualizada | {portfolio_metrics.get('volatilidade_anualizada', 0)*100:.2f}% |")
            lines.append(f"| Sharpe Ratio | {portfolio_metrics.get('sharpe_ratio', 0):.2f} |")
            lines.append(f"| Sortino Ratio | {portfolio_metrics.get('sortino_ratio', 0):.2f} |")
            lines.append(f"| Calmar Ratio | {portfolio_metrics.get('calmar_ratio', 0):.2f} |")
            lines.append(f"| Maximum Drawdown | {portfolio_metrics.get('max_drawdown', 0)*100:.2f}% |")

            if "alpha" in portfolio_metrics:
                lines.append(f"| Alpha | {portfolio_metrics.get('alpha', 0)*100:.2f}% |")
                lines.append(f"| Beta | {portfolio_metrics.get('beta', 1):.2f} |")
                lines.append(f"| Tracking Error | {portfolio_metrics.get('tracking_error', 0)*100:.2f}% |")
                lines.append(f"| Information Ratio | {portfolio_metrics.get('information_ratio', 0):.2f} |")
            lines.append("")

        if regime_analysis:
            lines.append("## 🌦️ Análise de Regimes")
            lines.append("")
            regime = regime_analysis.get("classificacao", "Indefinido")
            confianca = regime_analysis.get("confianca", 0)
            lines.append(f"**Regime Atual:** {regime} (confiança: {confianca*100:.0f}%)")
            lines.append("")
            metricas = regime_analysis.get("metricas", {})
            if metricas:
                lines.append("| Métrica | Valor |")
                lines.append("|---------|-------|")
                lines.append(f"| Retorno 12m | {metricas.get('retorno_anualizado', 0)*100:.2f}% |")
                lines.append(f"| Vol 12m | {metricas.get('volatilidade_anualizada', 0)*100:.2f}% |")
                lines.append(f"| Sharpe 12m | {metricas.get('sharpe_12m', 0):.2f} |")
                lines.append(f"| Max Drawdown | {metricas.get('max_drawdown', 0)*100:.2f}% |")
            lines.append("")

        if correlation_analysis:
            lines.append("## 🔗 Análise de Correlação")
            lines.append("")
            avg_corr = correlation_analysis.get("correlacao_media", 0)
            eff_div = correlation_analysis.get("diversificacao_efetiva", 0)
            lines.append(f"| Indicador | Valor |")
            lines.append(f"|-----------|-------|")
            lines.append(f"| Correlação Média | {avg_corr:.2f} |")
            lines.append(f"| Diversificação Efetiva | {eff_div:.1f} ativos |")
            lines.append("")

            fator_corr = correlation_analysis.get("fator_correlacao", {})
            if fator_corr:
                vif = fator_corr.get("vif", {})
                if vif:
                    lines.append("### VIF dos Fatores")
                    lines.append("")
                    lines.append("| Fator | VIF |")
                    lines.append("|-------|-----|")
                    for k, v in vif.items():
                        alerta = " ⚠️" if v > 5 else ""
                        lines.append(f"| {k.capitalize()} | {v:.2f}{alerta} |")
            lines.append("")

        lines.append("## ✅ Avaliação Final")
        lines.append("")
        sharpe = portfolio_metrics.get("sharpe_ratio", 0)
        mdd = portfolio_metrics.get("max_drawdown", 0)

        if sharpe > 1.0 and mdd < 0.20:
            lines.append("✅ **Performance Excelente:** Sharpe > 1.0 e Drawdown controlado.")
        elif sharpe > 0.5:
            lines.append("✅ **Performance Boa:** Sharpe positivo com risco moderado.")
        elif sharpe > 0:
            lines.append("⚠️ **Performance Mediana:** Sharpe positivo, porém baixo. Considere otimizações.")
        else:
            lines.append("❌ **Performance Abaixo do Esperado:** Sharpe negativo. Revisar estratégia.")

        return "\n".join(lines)


def save_report(content: str, filename: str, output_dir: str = "reports") -> str:
    """Save a report to a file."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
