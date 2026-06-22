# 📊 Radar B3 AI

Dashboard inteligente para análise da bolsa brasileira (B3) com suporte a ações, FIIs, ETFs, BDRs e carteiras recomendadas.

## Funcionalidades

| Módulo | Descrição |
|--------|-----------|
| 📡 **Radar** | Top 20 ações ranqueadas por score fundamentalista (60%) + técnico (40%) |
| 🔍 **Analisar Ticker** | Análise detalhada de qualquer ativo com gauges, fundamentos e indicadores técnicos (RSI, MACD, SMA) |
| 🏢 **FIIs** | Ranking e análise individual de Fundos Imobiliários |
| 📊 **ETFs** | Ranking de ETFs nacionais e internacionais |
| 🌎 **BDRs** | Ranking de BDRs das maiores empresas globais |
| 💵 **Dividendos** | Ranking dos maiores dividend yields da B3 |
| 🧙 **Magic Formula** | Estratégia Joel Greenblatt (P/L + EV/EBITDA) |
| 💼 **Carteiras** | 4 perfis de investimento com alocação automática |

## Tecnologias

- **Python** — backend de análise
- **Streamlit** — frontend interativo
- **Yahoo Finance (yfinance)** — dados de mercado
- **Plotly** — gráficos interativos
- **Pandas / NumPy** — processamento de dados

## Instalação

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
source venv/bin/activate
streamlit run app.py
```

Acesse em http://localhost:8501

### Comandos via terminal

```bash
python .opencode/skills/radar-b3/scripts/analise.py radar
python .opencode/skills/radar-b3/scripts/analise.py fiis
python .opencode/skills/radar-b3/scripts/analise.py etfs
python .opencode/skills/radar-b3/scripts/analise.py bdrs
python .opencode/skills/radar-b3/scripts/analise.py dividendos
python .opencode/skills/radar-b3/scripts/analise.py magic-formula
python .opencode/skills/radar-b3/scripts/analise.py carteiras
python .opencode/skills/radar-b3/scripts/analise.py analisar PETR4
```

## Perfis de Carteira

| Perfil | Alocação | Foco |
|--------|----------|------|
| 🛡️ Conservador | 40% FIIs · 30% Ações DY · 20% ETFs · 10% BDRs | Preservação e renda |
| ⚖️ Moderado | 35% Ações · 25% ETFs · 25% FIIs · 15% BDRs | Equilíbrio |
| 🚀 Arrojado | 40% BDRs · 30% Ações · 20% ETFs · 10% FIIs | Crescimento |
| 🏖️ Aposentadoria | 45% FIIs · 35% Ações DY · 15% ETFs · 5% BDRs | Renda passiva |

## Requisitos

- Python 3.10+
- yfinance
- fundamentus
- streamlit
- plotly
- pandas
- numpy
- requests
