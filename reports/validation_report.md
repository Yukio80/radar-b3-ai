# 🔬 Relatório de Validação — QPE v3

**Gerado em:** 2026-06-23 08:57:22

---

## 1️⃣ Walk-Forward Validation

| Indicador | Valor |
|-----------|-------|
| Total de Janelas | 1 |
| Retorno Médio (Treino) | 19.24% |
| Retorno Médio (Teste) | 86.65% |
| Mediana Retorno (Teste) | 86.65% |
| Desvio Padrão (Teste) | 0.00% |
| Melhor Janela (Teste) | 86.65% |
| Pior Janela (Teste) | 86.65% |
| Janelas Positivas | 1/1 |
| Taxa de Acerto | 100.0% |

### Detalhamento por Janela

| Janela | Treino | Teste | Retorno Treino | Retorno Teste |
|--------|--------|-------|----------------|---------------|
| 1 | 2024-06-23 → 2025-06-23 | 2025-06-24 → 2025-12-21 | 19.2% | 86.7% |

## 2️⃣ Monte Carlo Simulation

| Indicador | Valor |
|-----------|-------|
| Número de Simulações | 5,000 |
| Horizonte | 252 dias |
| Retorno Esperado | 16.29% |
| Volatilidade Esperada | 24.68% |
| VaR 95% | -24.87% |
| VaR 99% | -36.49% |
| CVaR 95% | -32.17% |
| Probabilidade de Perda | 29.4% |
| Prob. Superar CDI | 51.1% |
| Prob. Superar IBOV | 43.8% |
| Retorno Médio Simulado | 16.99% |
| Melhor Cenário | 184.97% |
| Pior Cenário | -55.99% |

## 3️⃣ Conclusões

✅ **Walk-Forward:** O modelo apresenta consistência fora da amostra, com taxa de acerto de 100.0% e retorno médio positivo de 86.65%.
⚠️ **Monte Carlo:** Risco moderado. Considere aumentar diversificação.

---

*Relatório gerado automaticamente pelo QPE v3 — Quantitative Portfolio Engine*