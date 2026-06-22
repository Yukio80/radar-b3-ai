---
name: radar-b3
description: >
  Especialista em análise da bolsa brasileira (B3). Use quando o usuário
  mencionar "radar", "ações", "B3", "bolsa", "investir", "dividendos",
  "magic formula", "PETR4", "VALE3", ou qualquer ticker brasileiro.
  Fornece conhecimento sobre indicadores fundamentalistas e técnicos
  específicos para o mercado brasileiro, fontes de dados e estratégias.
---

# Skill Radar B3 — Análise da Bolsa Brasileira

## Fontes de Dados

1. **brapi.dev** (API REST gratuita)
   - Cotações em tempo real, histórico OHLCV, dividendos
   - Indicadores fundamentalistas: P/L, P/VP, ROE, DY, EV/EBITDA, Margem Líquida
   - Dados macro: SELIC, IPCA, CDI
   - Sem API key obrigatória (rate limit ~15k req/dia)

2. **yfinance** (fallback para preços históricos)
   - Usar sufixo `.SA` para ativos brasileiros (ex: `PETR4.SA`)
   - Períodos: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

3. **openfindata** (MCP Server)
   - Agrega BCB, CVM, B3, IBGE, Tesouro Direto
   - Zero autenticação

## Indicadores Fundamentalistas

### Valuation
| Indicador | O que mede | Faixa Ideal (B3) |
|-----------|-----------|-------------------|
| P/L | Preço / Lucro por Ação | 5-15 (setor dependente) |
| P/VP | Preço / Valor Patrimonial | < 1.5 (ideal < 1.0) |
| EV/EBITDA | Valor da Firma / EBITDA | < 8 (bom), 8-12 (justo) |
| P/Ativo | Preço / Ativo Total | < 1.0 |

### Rentabilidade
| Indicador | O que mede | Faixa Ideal (B3) |
|-----------|-----------|-------------------|
| ROE | Retorno sobre Patrimônio Líquido | > 15% (bom), > 20% (excelente) |
| Margem Líquida | Lucro Líquido / Receita | > 8% (bom), > 15% (excelente) |
| DY | Dividend Yield anual | > 5% (bom), > 8% (excelente) |

### Endividamento
| Indicador | O que mede | Faixa Ideal (B3) |
|-----------|-----------|-------------------|
| Dívida/PL | Dívida Líquida / Patrimônio | < 1.0 (bom), < 0.5 (ótimo) |

## Indicadores Técnicos

| Indicador | Uso | Interpretação |
|-----------|-----|---------------|
| RSI (14) | Força da tendência | < 30: sobrevendido; > 70: sobrecomprado |
| MACD | Cruzamento de médias | Acima da linha de sinal = alta |
| SMA 20 | Tendência curto prazo | Preço acima = tendência altista |
| SMA 200 | Tendência longo prazo | Preço acima = tendência altista |

## Estratégias de Seleção

### Magic Formula (Joel Greenblatt)
1. Rankear empresas por ROIC (retorno sobre capital investido)
2. Rankear por EV/EBITDA (menor = mais barato)
3. Rank combinado = melhor relação qualidade + preço

### Screening Diário (Radar)
- Score fundamentalista (peso 60%): P/L baixo, P/VP < 1, ROE alto, DY consistente, baixo endividamento
- Score técnico (peso 40%): RSI entre 30-50 (comprando), preço > SMA20/SMA200, MACD altista
- Liquidez mínima: evitar microcaps (< R$ 5M volume diário)

### Foco em Dividendos
Buscar DY > 5% com payout sustentável (< 80%), P/L < 15, dívida controlada.

## Script de Análise

O script `scripts/analise.py` está disponível neste skill. Executar com `python scripts/analise.py <comando>` (ou com `.venv/bin/python` se usar venv).
Comandos:
- `radar` — análise completa do mercado (top 20 oportunidades)
- `analisar PETR4` — análise detalhada de um ticker
- `dividendos` — ranking de dividendos
- `magic-formula` — ranking Magic Formula

## Limitações e Cuidados
- Dados via Yahoo Finance podem ter delay de até 15 minutos
- small caps podem ter menos dados disponíveis
- Este skill é informativo, não constitui recomendação de investimento
