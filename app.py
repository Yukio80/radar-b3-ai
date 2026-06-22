import sys
import json
import time
import io
import importlib.util
from contextlib import redirect_stdout, redirect_stderr

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_SCRIPT = ".opencode/skills/radar-b3/scripts/analise.py"
_spec = importlib.util.spec_from_file_location("analise", _SCRIPT)
_analise = importlib.util.module_from_spec(_spec)
sys.modules["analise"] = _analise
_spec.loader.exec_module(_analise)

analisar_ticker = _analise.analisar_ticker
analisar_fii = _analise.analisar_fii
analisar_etf = _analise.analisar_etf
analisar_bdr = _analise.analisar_bdr
comando_radar = _analise.comando_radar
comando_dividendos = _analise.comando_dividendos
comando_magic_formula = _analise.comando_magic_formula
comando_fiis = _analise.comando_fiis
comando_etfs = _analise.comando_etfs
comando_bdrs = _analise.comando_bdrs
comando_carteiras = _analise.comando_carteiras
comando_qpe = _analise.comando_qpe

from database import listar_snapshots, carregar_snapshot, historico_portfolio, init_db
init_db()

st.set_page_config(
    page_title="Radar B3 AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .score-high { color: #00c853; font-weight: bold; }
    .score-mid { color: #ff9100; font-weight: bold; }
    .score-low { color: #ff1744; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def format_dy(v):
    if v is None:
        return "-"
    return f"{v*100:.2f}%"

def format_pct(v):
    if v is None:
        return "-"
    return f"{v*100:.2f}%"

def format_brl(v):
    if v is None:
        return "-"
    return f"R$ {v:.2f}"

def score_color(s):
    if s >= 70:
        return "score-high"
    elif s >= 50:
        return "score-mid"
    return "score-low"

def run_radar():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_radar()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao processar radar:\n{err.getvalue()}")
        return None

def run_dividendos():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_dividendos()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao processar dividendos:\n{err.getvalue()}")
        return None

def run_magic():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_magic_formula()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao processar Magic Formula:\n{err.getvalue()}")
        return None

def run_fiis():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_fiis()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao processar FIIs:\n{err.getvalue()}")
        return None

def run_etfs():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_etfs()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao processar ETFs:\n{err.getvalue()}")
        return None

def run_bdrs():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_bdrs()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao processar BDRs:\n{err.getvalue()}")
        return None

def run_carteiras():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_carteiras()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao processar carteiras:\n{err.getvalue()}")
        return None

def run_qpe():
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        comando_qpe()
    out = buf.getvalue()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        st.error(f"Erro ao executar QPE:\n{err.getvalue()}")
        return None


PAGES = {
    "📡 Radar": "radar",
    "🔍 Analisar Ticker": "analisar",
    "🏢 FIIs": "fiis",
    "📊 ETFs": "etfs",
    "🌎 BDRs": "bdrs",
    "💵 Dividendos": "dividendos",
    "🧙 Magic Formula": "magic",
    "💼 Carteiras": "carteiras",
    "🧠 QPE v2": "qpe",
    "📈 Histórico": "historico",
}

st.sidebar.title("📊 Radar B3 AI")
st.sidebar.markdown("Análise da bolsa brasileira")
st.sidebar.divider()

page = st.sidebar.radio("Navegação", list(PAGES.keys()))
page_key = PAGES[page]

st.sidebar.divider()
st.sidebar.markdown("**Sobre**")
st.sidebar.info(
    "Dados via Yahoo Finance. "
    "Indicadores fundamentalistas e técnicos "
    "para auxiliar na análise de ações da B3."
)


if page_key == "radar":
    st.title("📡 Radar de Oportunidades")
    st.markdown("Top 20 ações ranqueadas por score fundamentalista (60%) + técnico (40%)")

    if st.button("🔄 Atualizar Radar", type="primary", use_container_width=True):
        with st.spinner("Analisando ativos da B3..."):
            data = run_radar()
        if data:
            st.session_state["radar_data"] = data

    if "radar_data" not in st.session_state:
        with st.spinner("Primeira análise..."):
            st.session_state["radar_data"] = run_radar()

    data = st.session_state.get("radar_data")
    if not data:
        st.warning("Clique em 'Atualizar Radar' para iniciar.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Data", data.get("data", "-"))
    col2.metric("Ativos Analisados", data.get("total_analisados", 0))
    col3.metric("Top", f"{len(data.get('top_oportunidades', []))}")

    top = data["top_oportunidades"]
    df = pd.DataFrame(top)

    cols_show = {
        "ticker": "Ticker",
        "empresa": "Empresa",
        "preco": "Preço",
        "score_total": "Score",
        "score_fundamental": "Score Fund.",
        "score_tecnico": "Score Tec.",
    }
    show_df = df[[c for c in cols_show if c in df.columns]].copy()
    show_df = show_df.rename(columns=cols_show)
    if "Preço" in show_df.columns:
        show_df["Preço"] = show_df["Preço"].apply(format_brl)

    st.subheader("🏆 Top 20 Ranking")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df.head(10),
        x="ticker",
        y="score_total",
        color="score_total",
        color_continuous_scale="RdYlGn",
        title="Top 10 - Score Total",
        labels={"ticker": "Ticker", "score_total": "Score"},
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Distribuição por Setor")
    if "setor" in df.columns and df["setor"].notna().any():
        setor_counts = df.groupby("setor").size().reset_index(name="count")
        fig2 = px.pie(setor_counts, values="count", names="setor", title="Setores no Top 20")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔬 Detalhamento Fundamentalista")
    fund_rows = []
    for item in top:
        f = item.get("fundamentos", {})
        fund_rows.append({
            "Ticker": item["ticker"],
            "P/L": round(f.get("pl"), 2) if f.get("pl") else "-",
            "P/VP": round(f.get("pvp"), 2) if f.get("pvp") else "-",
            "ROE": format_pct(f.get("roe")),
            "DY": format_dy(f.get("dy")),
            "EV/EBITDA": round(f.get("ev_ebit"), 2) if f.get("ev_ebit") else "-",
            "Margem Líq.": format_pct(f.get("margem_liquida")),
            "Dívida/PL": round(f.get("divida_pl"), 2) if f.get("divida_pl") else "-",
        })
    fund_df = pd.DataFrame(fund_rows)
    st.dataframe(fund_df, use_container_width=True, hide_index=True)


elif page_key == "analisar":
    st.title("🔍 Análise Detalhada de Ticker")
    ticker = st.text_input("Digite o ticker (ex: PETR4, VALE3):", value="PETR4").strip().upper()

    if st.button("Analisar", type="primary"):
        with st.spinner(f"Analisando {ticker}..."):
            result = analisar_ticker(ticker)
        if result:
            st.session_state["analise_result"] = result
            st.session_state["analise_ticker"] = ticker

    result = st.session_state.get("analise_result")
    ticker_analisado = st.session_state.get("analise_ticker")

    if not result:
        st.info("Digite um ticker e clique em 'Analisar'.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Empresa", result.get("empresa", "-"))
    col2.metric("Preço", format_brl(result.get("preco")))
    col3.metric("Setor", result.get("setor", "-"))
    col4.metric("Variação Dia", f'{result.get("variacao_dia", 0):.2f}%' if result.get("variacao_dia") else "-")

    st.subheader("📊 Scores")
    sc1, sc2, sc3 = st.columns(3)
    sf = result.get("score_fundamental", 0)
    stt = result.get("score_tecnico", 0)
    stot = result.get("score_total", 0)

    def gauge(val, title):
        c = "green" if val >= 70 else ("orange" if val >= 50 else "red")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "", "font": {"size": 36, "color": c}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": c},
                   "steps": [
                       {"range": [0, 40], "color": "lightcoral"},
                       {"range": [40, 70], "color": "khaki"},
                       {"range": [70, 100], "color": "lightgreen"},
                   ]},
            title={"text": title},
        ))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    sc1.plotly_chart(gauge(sf, "Fundamental"), use_container_width=True)
    sc2.plotly_chart(gauge(stt, "Técnico"), use_container_width=True)
    sc3.plotly_chart(gauge(stot, "Total"), use_container_width=True)

    st.subheader("📋 Fundamentos")
    f = result.get("fundamentos", {})
    fund_data = {
        "Indicador": ["P/L", "P/VP", "ROE", "Dividend Yield", "EV/EBITDA",
                       "Margem Líquida", "Dívida/PL", "VPA", "EPS", "Payout"],
        "Valor": [
            round(f.get("pl"), 2) if f.get("pl") else "-",
            round(f.get("pvp"), 2) if f.get("pvp") else "-",
            format_pct(f.get("roe")),
            format_dy(f.get("dy")),
            round(f.get("ev_ebit"), 2) if f.get("ev_ebit") else "-",
            format_pct(f.get("margem_liquida")),
            round(f.get("divida_pl"), 2) if f.get("divida_pl") else "-",
            format_brl(f.get("vpa")),
            round(f.get("eps"), 2) if f.get("eps") else "-",
            format_pct(f.get("payout")),
        ],
    }
    st.dataframe(pd.DataFrame(fund_data), use_container_width=True, hide_index=True)

    st.subheader("📈 Análise Técnica")
    t = result.get("tecnicos", {})
    if t:
        tec_cols = {
            "Preço Atual": format_brl(t.get("preco_atual")),
            "SMA 20": format_brl(t.get("sma20")),
            "SMA 200": format_brl(t.get("sma200")),
            "RSI (14)": round(t.get("rsi14"), 2) if t.get("rsi14") else "-",
            "MACD": round(t.get("macd"), 4) if t.get("macd") else "-",
            "Sinal MACD": round(t.get("macd_sinal"), 4) if t.get("macd_sinal") else "-",
            "Máx 6m": format_brl(t.get("max_6m")),
            "Mín 6m": format_brl(t.get("min_6m")),
            "Variação 6m": f'{t.get("variacao_6m", 0):.2f}%' if t.get("variacao_6m") else "-",
        }
        tec_df = pd.DataFrame(list(tec_cols.items()), columns=["Indicador", "Valor"])
        st.dataframe(tec_df, use_container_width=True, hide_index=True)

        rsi = t.get("rsi14")
        if rsi:
            rsi_color = "green" if 30 <= rsi <= 70 else ("red" if rsi > 70 else "orange")
            st.markdown(
                f"**RSI:** <span style='color:{rsi_color};font-size:1.5em;font-weight:bold'>{rsi:.1f}</span> "
                f"{'🔴 Sobrecomprado' if rsi > 70 else '🟢 Neutro' if rsi >= 30 else '🟠 Sobrevendido'}",
                unsafe_allow_html=True,
            )
    else:
        st.info("Dados técnicos indisponíveis")

elif page_key == "fiis":
    st.title("🏢 Fundos Imobiliários (FIIs)")
    st.markdown("Ranking dos melhores FIIs por score fundamentalista + técnico")

    if st.button("🔄 Atualizar FIIs", type="primary", use_container_width=True):
        with st.spinner("Analisando FIIs..."):
            data = run_fiis()
        if data:
            st.session_state["fiis_data"] = data

    if "fiis_data" not in st.session_state:
        with st.spinner("Primeira análise..."):
            st.session_state["fiis_data"] = run_fiis()

    data = st.session_state.get("fiis_data")
    if not data:
        st.warning("Clique em 'Atualizar FIIs' para iniciar.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Data", data.get("data", "-"))
    col2.metric("FIIs Analisados", data.get("total_analisados", 0))
    col3.metric("Top", f"{len(data.get('top_fiis', []))}")

    top = data["top_fiis"]
    df = pd.DataFrame(top)

    cols_show = {
        "ticker": "Ticker",
        "empresa": "Nome",
        "preco": "Preço",
        "score_total": "Score",
        "score_fundamental": "Score Fund.",
        "score_tecnico": "Score Tec.",
    }
    show_df = df[[c for c in cols_show if c in df.columns]].copy()
    show_df = show_df.rename(columns=cols_show)
    if "Preço" in show_df.columns:
        show_df["Preço"] = show_df["Preço"].apply(format_brl)

    st.subheader("🏆 Top 20 FIIs")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df.head(10),
        x="ticker",
        y="score_total",
        color="score_total",
        color_continuous_scale="RdYlGn",
        title="Top 10 FIIs - Score Total",
        labels={"ticker": "Ticker", "score_total": "Score"},
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detalhamento FII")
    fund_rows = []
    for item in top:
        f = item.get("fundamentos", {})
        fund_rows.append({
            "Ticker": item["ticker"],
            "P/VP": round(f.get("pvp"), 2) if f.get("pvp") else "-",
            "DY": format_dy(f.get("dy")),
            "Valor Mercado": format_brl(f.get("valor_mercado")),
            "VPA": format_brl(f.get("vpa")),
        })
    fund_df = pd.DataFrame(fund_rows)
    st.dataframe(fund_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔍 Analisar FII Individual")
    fii_ticker = st.text_input("Digite o ticker do FII (ex: KNRI11, HGLG11):", value="KNRI11").strip().upper()
    if st.button("Analisar FII", type="primary"):
        with st.spinner(f"Analisando FII {fii_ticker}..."):
            result = analisar_fii(fii_ticker)
        if result:
            st.session_state["fii_analise_result"] = result
            st.session_state["fii_analise_ticker"] = fii_ticker

    fii_result = st.session_state.get("fii_analise_result")
    if fii_result:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("FII", fii_result.get("empresa", "-"))
        col2.metric("Preço", format_brl(fii_result.get("preco")))
        col3.metric("Variação Dia", f'{fii_result.get("variacao_dia", 0):.2f}%' if fii_result.get("variacao_dia") else "-")
        col4.metric("Score Total", fii_result.get("score_total", "-"))

        sc1, sc2, sc3 = st.columns(3)
        sf = fii_result.get("score_fundamental", 0)
        stt = fii_result.get("score_tecnico", 0)
        stot = fii_result.get("score_total", 0)

        def gauge(val, title):
            c = "green" if val >= 70 else ("orange" if val >= 50 else "red")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number={"suffix": "", "font": {"size": 36, "color": c}},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": c},
                       "steps": [
                           {"range": [0, 40], "color": "lightcoral"},
                           {"range": [40, 70], "color": "khaki"},
                           {"range": [70, 100], "color": "lightgreen"},
                       ]},
                title={"text": title},
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
            return fig

        sc1.plotly_chart(gauge(sf, "Fundamental"), use_container_width=True)
        sc2.plotly_chart(gauge(stt, "Técnico"), use_container_width=True)
        sc3.plotly_chart(gauge(stot, "Total"), use_container_width=True)

        f = fii_result.get("fundamentos", {})
        fii_fund_data = {
            "Indicador": ["P/VP", "Dividend Yield", "Valor Mercado", "VPA", "P/L"],
            "Valor": [
                round(f.get("pvp"), 2) if f.get("pvp") else "-",
                format_dy(f.get("dy")),
                format_brl(f.get("valor_mercado")),
                format_brl(f.get("vpa")),
                round(f.get("pl"), 2) if f.get("pl") else "-",
            ],
        }
        st.dataframe(pd.DataFrame(fii_fund_data), use_container_width=True, hide_index=True)

        t = fii_result.get("tecnicos", {})
        if t:
            st.subheader("📈 Análise Técnica")
            tec_cols = {
                "Preço Atual": format_brl(t.get("preco_atual")),
                "SMA 20": format_brl(t.get("sma20")),
                "SMA 200": format_brl(t.get("sma200")),
                "RSI (14)": round(t.get("rsi14"), 2) if t.get("rsi14") else "-",
                "MACD": round(t.get("macd"), 4) if t.get("macd") else "-",
                "Sinal MACD": round(t.get("macd_sinal"), 4) if t.get("macd_sinal") else "-",
                "Máx 6m": format_brl(t.get("max_6m")),
                "Mín 6m": format_brl(t.get("min_6m")),
                "Variação 6m": f'{t.get("variacao_6m", 0):.2f}%' if t.get("variacao_6m") else "-",
            }
            tec_df = pd.DataFrame(list(tec_cols.items()), columns=["Indicador", "Valor"])
            st.dataframe(tec_df, use_container_width=True, hide_index=True)


elif page_key == "etfs":
    st.title("📊 ETFs")
    st.markdown("Ranking dos melhores ETFs por score fundamentalista + técnico")

    if st.button("🔄 Atualizar ETFs", type="primary", use_container_width=True):
        with st.spinner("Analisando ETFs..."):
            data = run_etfs()
        if data:
            st.session_state["etfs_data"] = data

    if "etfs_data" not in st.session_state:
        with st.spinner("Primeira análise..."):
            st.session_state["etfs_data"] = run_etfs()

    data = st.session_state.get("etfs_data")
    if not data:
        st.warning("Clique em 'Atualizar ETFs' para iniciar.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Data", data.get("data", "-"))
    col2.metric("ETFs Analisados", data.get("total_analisados", 0))
    col3.metric("Top", f"{len(data.get('top_etfs', []))}")

    top = data["top_etfs"]
    df = pd.DataFrame(top)

    cols_show = {
        "ticker": "Ticker",
        "empresa": "Nome",
        "preco": "Preço",
        "score_total": "Score",
        "score_fundamental": "Score Fund.",
        "score_tecnico": "Score Tec.",
    }
    show_df = df[[c for c in cols_show if c in df.columns]].copy()
    show_df = show_df.rename(columns=cols_show)
    if "Preço" in show_df.columns:
        show_df["Preço"] = show_df["Preço"].apply(format_brl)

    st.subheader("🏆 Top 20 ETFs")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df.head(10),
        x="ticker",
        y="score_total",
        color="score_total",
        color_continuous_scale="RdYlGn",
        title="Top 10 ETFs - Score Total",
        labels={"ticker": "Ticker", "score_total": "Score"},
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detalhamento ETFs")
    fund_rows = []
    for item in top:
        f = item.get("fundamentos", {})
        fund_rows.append({
            "Ticker": item["ticker"],
            "P/VP": round(f.get("pvp"), 2) if f.get("pvp") else "-",
            "DY": format_dy(f.get("dy")),
            "Valor Mercado": format_brl(f.get("valor_mercado")),
            "VPA": format_brl(f.get("vpa")),
        })
    fund_df = pd.DataFrame(fund_rows)
    st.dataframe(fund_df, use_container_width=True, hide_index=True)


elif page_key == "bdrs":
    st.title("🌎 BDRs")
    st.markdown("Ranking dos melhores BDRs por score fundamentalista + técnico")

    if st.button("🔄 Atualizar BDRs", type="primary", use_container_width=True):
        with st.spinner("Analisando BDRs..."):
            data = run_bdrs()
        if data:
            st.session_state["bdrs_data"] = data

    if "bdrs_data" not in st.session_state:
        with st.spinner("Primeira análise..."):
            st.session_state["bdrs_data"] = run_bdrs()

    data = st.session_state.get("bdrs_data")
    if not data:
        st.warning("Clique em 'Atualizar BDRs' para iniciar.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Data", data.get("data", "-"))
    col2.metric("BDRs Analisados", data.get("total_analisados", 0))
    col3.metric("Top", f"{len(data.get('top_bdrs', []))}")

    top = data["top_bdrs"]
    df = pd.DataFrame(top)

    cols_show = {
        "ticker": "Ticker",
        "empresa": "Empresa",
        "preco": "Preço",
        "score_total": "Score",
        "score_fundamental": "Score Fund.",
        "score_tecnico": "Score Tec.",
    }
    show_df = df[[c for c in cols_show if c in df.columns]].copy()
    show_df = show_df.rename(columns=cols_show)
    if "Preço" in show_df.columns:
        show_df["Preço"] = show_df["Preço"].apply(format_brl)

    st.subheader("🏆 Top 20 BDRs")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df.head(10),
        x="ticker",
        y="score_total",
        color="score_total",
        color_continuous_scale="RdYlGn",
        title="Top 10 BDRs - Score Total",
        labels={"ticker": "Ticker", "score_total": "Score"},
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detalhamento BDRs")
    fund_rows = []
    for item in top:
        f = item.get("fundamentos", {})
        fund_rows.append({
            "Ticker": item["ticker"],
            "P/L": round(f.get("pl"), 2) if f.get("pl") else "-",
            "P/VP": round(f.get("pvp"), 2) if f.get("pvp") else "-",
            "DY": format_dy(f.get("dy")),
            "ROE": format_pct(f.get("roe")),
            "Margem Líq.": format_pct(f.get("margem_liquida")),
            "Valor Mercado": format_brl(f.get("valor_mercado")),
        })
    fund_df = pd.DataFrame(fund_rows)
    st.dataframe(fund_df, use_container_width=True, hide_index=True)


elif page_key == "carteiras":
    st.title("💼 Carteiras Recomendadas")
    st.markdown("Portfólios sugeridos conforme seu perfil de investidor")

    if st.button("🔄 Montar Carteiras", type="primary", use_container_width=True):
        with st.spinner("Analisando mercado e montando carteiras..."):
            data = run_carteiras()
        if data:
            st.session_state["carteiras_data"] = data

    if "carteiras_data" not in st.session_state:
        with st.spinner("Primeira análise..."):
            st.session_state["carteiras_data"] = run_carteiras()

    data = st.session_state.get("carteiras_data")
    if not data:
        st.warning("Clique em 'Montar Carteiras' para iniciar.")
        st.stop()

    st.metric("Data", data.get("data", "-"))

    perfis = data.get("perfis", {})
    perfs_order = ["conservador", "moderado", "arrojado", "aposentadoria"]

    for chave in perfs_order:
        p = perfis.get(chave)
        if not p:
            continue

        st.divider()
        st.subheader(f"{p.get('icone', '')} {p['nome']}")
        st.caption(p.get("descricao", ""))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score Médio", f'{p["score_medio"]}/100')
        col2.metric("DY Ponderado", f'{p["dy_ponderado"]}%')
        col3.metric("Ativos", p["total_ativos"])
        col4.metric("Alocação", " | ".join(f'{k} {v}%' for k, v in p["alocacao"].items()))

        ativos = p.get("ativos", [])
        if ativos:
            ativos_df = pd.DataFrame(ativos)
            ativos_df["Preço"] = ativos_df["preco"].apply(format_brl)
            ativos_df["Peso"] = ativos_df["peso"].apply(lambda x: f"{x}%")

            show_cols = ["ticker", "empresa", "tipo", "Preço", "Peso", "score", "dy", "setor"]
            show_df = ativos_df[[c for c in show_cols if c in ativos_df.columns]].rename(columns={
                "ticker": "Ticker", "empresa": "Empresa", "tipo": "Tipo",
                "Preço": "Preço", "Peso": "Peso", "score": "Score",
                "dy": "DY %", "setor": "Setor",
            })
            st.dataframe(show_df, use_container_width=True, hide_index=True)

            fig = px.pie(
                ativos_df,
                values="peso",
                names="tipo",
                title=f"Alocação - {p['nome']}",
                color="tipo",
                color_discrete_map={
                    "FIIs": "#00c853",
                    "Ações Dividendos": "#2979ff",
                    "Ações": "#2979ff",
                    "ETFs": "#ff9100",
                    "BDRs": "#d500f9",
                },
            )
            st.plotly_chart(fig, use_container_width=True)


elif page_key == "qpe":
    st.title("🧠 Quantitative Portfolio Engine v2")
    st.markdown("Score multifatorial (Qualidade + Valuation + Dividendos + Crescimento + Segurança)")

    if st.button("🚀 Executar QPE v2", type="primary", use_container_width=True):
        with st.spinner("Pipeline quantitativa em execução (outliers → CAGR → score → pesos → IRP → stress test)..."):
            data = run_qpe()
        if data:
            st.session_state["qpe_data"] = data

    if "qpe_data" not in st.session_state:
        with st.spinner("Primeira execução..."):
            st.session_state["qpe_data"] = run_qpe()

    data = st.session_state.get("qpe_data")
    if not data:
        st.warning("Clique em 'Executar QPE v2' para iniciar.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data", data.get("data", "-"))
    col2.metric("Ativos Analisados", data.get("total_analisados", 0))
    col3.metric("Score Médio", f'{data.get("score_medio", 0)}/100')
    irp = data.get("irp", {})
    col4.metric("IRP", f'{irp.get("IRP", 0)}/100', irp.get("classificacao", ""))

    if irp:
        sub = irp.get("sub_scores", {})
        if sub:
            st.subheader("🛡️ Índice de Robustez Patrimonial")
            sub_df = pd.DataFrame([{
                "Componente": {"diversificacao": "Diversificação",
                               "qualidade_media": "Qualidade Média",
                               "estabilidade_dividendos": "Estabilidade Dividendos",
                               "baixa_alavancagem": "Baixa Alavancagem"}.get(k, k),
                "Score": v
            } for k, v in sub.items()])
            st.dataframe(sub_df, use_container_width=True, hide_index=True)

    stress = data.get("stress_test", {})
    if stress:
        cenarios = stress.get("cenarios", {})
        if cenarios:
            st.subheader("⚠️ Stress Test")
            stress_rows = []
            for name, result in cenarios.items():
                if isinstance(result, dict) and "perda_estimada" in result:
                    stress_rows.append({
                        "Cenário": name,
                        "Perda Estimada": f'{result["perda_estimada"]:.1f}%',
                        "Recuperação": f'{result.get("recuperacao_estimada_dias", 0)} dias',
                    })
            if stress_rows:
                st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

    top10 = data.get("top_10", [])
    if top10:
        st.subheader("🏆 Top 10 Ativos (Score Multifatorial)")
        top_df = pd.DataFrame(top10)
        fatores = ["qualidade", "valuation", "dividendos", "crescimento", "seguranca"]
        for f in fatores:
            if f in top_df.columns:
                top_df[f] = top_df[f].round(1)
        top_df["classificacao"] = top_df.get("classificacao", "")
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            top_df, x="ticker", y="score",
            color="classificacao",
            title="Top 10 - Score Total",
            color_discrete_map={
                "Elite": "#00c853", "Excelente": "#64dd17",
                "Boa": "#ff9100", "Média": "#ffd600", "Fraca": "#ff1744",
            },
            text_auto=".1f",
        )
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        exps = data.get("explicacoes", [])
        if exps:
            st.subheader("📋 Explicações por Ativo")
            for exp_item in exps[:10]:
                with st.expander(f"{exp_item['ticker']} — Score: {exp_item['score']}"):
                    if exp_item.get("pontos_fortes"):
                        st.markdown("**✅ Pontos Fortes:**")
                        for p in exp_item["pontos_fortes"][:4]:
                            st.markdown(f"- {p}")
                    if exp_item.get("pontos_fracos"):
                        st.markdown("**⚠️ Pontos Fracos:**")
                        for p in exp_item["pontos_fracos"][:4]:
                            st.markdown(f"- {p}")

    outliers = data.get("outliers", {})
    if outliers:
        st.subheader("🔍 Outliers Detectados")
        out_rows = []
        for col, rep in outliers.items():
            if rep.get("outliers", 0) > 0:
                out_rows.append({
                    "Métrica": col,
                    "Total": rep["total"],
                    "Outliers": rep["outliers"],
                    "%": f'{rep["pct_outliers"]}%',
                    "Valores": ", ".join(str(v) for v in rep.get("outlier_values", [])[:5]),
                })
        if out_rows:
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    relatorio = data.get("relatorio", "")
    if relatorio:
        st.subheader("📄 Relatório Gerado")
        st.code(relatorio, language="markdown")


elif page_key == "historico":
    st.title("📈 Histórico de Análises")
    st.markdown("Snapshots salvos automaticamente a cada execução")

    tab1, tab2 = st.tabs(["📸 Snapshots", "💼 Evolução das Carteiras"])

    with tab1:
        tipos = ["radar", "fiis", "etfs", "bdrs", "dividendos", "magic_formula", "carteiras"]
        tipo_filter = st.selectbox("Filtrar por tipo", ["todos"] + tipos)
        filtro = None if tipo_filter == "todos" else tipo_filter

        snapshots = listar_snapshots(filtro, limite=50)
        if not snapshots:
            st.info("Nenhum snapshot encontrado. Execute alguma análise primeiro.")
        else:
            st.caption(f"{len(snapshots)} snapshots encontrados")
            for snap in snapshots:
                label = f"{snap['data_hora']} — {snap['tipo']}"
                if st.button(label, key=f"snap_{snap['id']}", use_container_width=True):
                    dados = carregar_snapshot(snap["id"])
                    if dados:
                        st.session_state["snapshot_view"] = dados
                        st.session_state["snapshot_tipo"] = snap["tipo"]
                        st.session_state["snapshot_data"] = snap["data_hora"]

            if "snapshot_view" in st.session_state:
                st.divider()
                st.subheader(f"{st.session_state['snapshot_tipo']} — {st.session_state['snapshot_data']}")
                dados = st.session_state["snapshot_view"]
                st.json(dados)

    with tab2:
        perfis_opt = ["conservador", "moderado", "arrojado", "aposentadoria"]
        perfil_sel = st.selectbox("Perfil", perfis_opt,
                                  format_func=lambda x: {"conservador": "🛡️ Conservador",
                                                          "moderado": "⚖️ Moderado",
                                                          "arrojado": "🚀 Arrojado",
                                                          "aposentadoria": "🏖️ Aposentadoria"}[x])

        hist = historico_portfolio(perfil_sel, limite=30)
        if not hist:
            st.info("Nenhum histórico para este perfil. Execute 'Carteiras' primeiro.")
        else:
            hist = list(reversed(hist))

            st.subheader("📊 Evolução do Score")
            df_hist = pd.DataFrame(hist)
            fig_score = px.line(
                df_hist, x="data_hora", y="score_medio",
                title=f"Score Médio - {perfil_sel}",
                markers=True,
                labels={"data_hora": "Data", "score_medio": "Score"},
            )
            fig_score.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_score, use_container_width=True)

            st.subheader("💵 Evolução do DY Ponderado")
            fig_dy = px.line(
                df_hist, x="data_hora", y="dy_ponderado",
                title=f"DY Ponderado - {perfil_sel}",
                markers=True,
                labels={"data_hora": "Data", "dy_ponderado": "DY %"},
            )
            st.plotly_chart(fig_dy, use_container_width=True)

            st.subheader("📋 Últimos Registros")
            for item in reversed(hist[-10:]):
                with st.expander(f"{item['data_hora']} — Score: {item['score_medio']}  DY: {item['dy_ponderado']}%  Ativos: {item['total_ativos']}"):
                    aloc = item.get("alocacao", {})
                    if aloc:
                        st.caption("Alocação: " + " · ".join(f"{k} {v}%" for k, v in aloc.items()))
                    ativos = item.get("ativos", [])
                    if ativos:
                        df_at = pd.DataFrame(ativos)
                        df_at["Preço"] = df_at["preco"].apply(lambda x: format_brl(x) if x else "-")
                        df_at["Peso"] = df_at["peso"].apply(lambda x: f"{x}%")
                        show = ["ticker", "tipo", "Preço", "Peso", "score", "dy"]
                        st.dataframe(
                            df_at[[c for c in show if c in df_at.columns]]
                            .rename(columns={"ticker": "Ticker", "tipo": "Tipo", "Preço": "Preço",
                                             "Peso": "Peso", "score": "Score", "dy": "DY %"}),
                            use_container_width=True, hide_index=True,
                        )


elif page_key == "dividendos":
    st.title("💵 Top Dividendos")
    st.markdown("Ranking dos maiores dividend yields da B3")

    if st.button("🔄 Atualizar Dividendos", type="primary", use_container_width=True):
        with st.spinner("Buscando dados de dividendos..."):
            data = run_dividendos()
        if data:
            st.session_state["div_data"] = data

    if "div_data" not in st.session_state:
        with st.spinner("Primeira análise..."):
            st.session_state["div_data"] = run_dividendos()

    data = st.session_state.get("div_data")
    if not data:
        st.warning("Clique em 'Atualizar Dividendos' para iniciar.")
        st.stop()

    st.metric("Data", data.get("data", "-"))

    top = data["top"]
    df = pd.DataFrame(top)
    df["DY"] = df["dy"].apply(lambda x: f"{x*100:.2f}%")
    df["Preço"] = df["preco"].apply(format_brl)
    df["P/L"] = df["pl"].apply(lambda x: round(x, 2) if x else "-")
    df["P/VP"] = df["pvp"].apply(lambda x: round(x, 2) if x else "-")

    show_cols = ["ticker", "empresa", "DY", "Preço", "P/L", "P/VP"]
    show_df = df[[c for c in show_cols if c in df.columns]].rename(columns={
        "ticker": "Ticker", "empresa": "Empresa", "DY": "DY",
        "Preço": "Preço", "P/L": "P/L", "P/VP": "P/VP",
    })
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df.head(10),
        x="ticker",
        y="dy",
        color="dy",
        color_continuous_scale="Greens",
        title="Top 10 - Dividend Yield",
        labels={"ticker": "Ticker", "dy": "Dividend Yield"},
        text=df.head(10)["DY"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


elif page_key == "magic":
    st.title("🧙 Magic Formula")
    st.markdown("Ranking combinado P/L + EV/EBITDA (Joel Greenblatt)")

    if st.button("🔄 Atualizar Magic Formula", type="primary", use_container_width=True):
        with st.spinner("Executando Magic Formula..."):
            data = run_magic()
        if data:
            st.session_state["magic_data"] = data

    if "magic_data" not in st.session_state:
        with st.spinner("Primeira análise..."):
            st.session_state["magic_data"] = run_magic()

    data = st.session_state.get("magic_data")
    if not data:
        st.warning("Clique em 'Atualizar Magic Formula' para iniciar.")
        st.stop()

    st.metric("Data", data.get("data", "-"))

    top = data["top"]
    df = pd.DataFrame(top)
    df["DY"] = df["dy"].apply(lambda x: format_dy(x) if x else "-")
    df["ROE"] = df["roe"].apply(lambda x: format_pct(x) if x else "-")
    df["Preço"] = df["preco"].apply(format_brl)
    df["P/L"] = df["pl"].apply(lambda x: round(x, 2) if x else "-")
    df["EV/EBITDA"] = df["ev_ebit"].apply(lambda x: round(x, 2) if x else "-")

    show_cols = ["ticker", "empresa", "Preço", "P/L", "EV/EBITDA", "ROE", "DY"]
    show_df = df[[c for c in show_cols if c in df.columns]].rename(columns={
        "ticker": "Ticker", "empresa": "Empresa", "Preço": "Preço",
        "P/L": "P/L", "EV/EBITDA": "EV/EBITDA", "ROE": "ROE", "DY": "DY",
    })
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df.head(10),
        x="ticker",
        y="pl",
        color="ev_ebit",
        color_continuous_scale="Blues",
        title="Top 10 - P/L (barras) e EV/EBITDA (cores)",
        labels={"ticker": "Ticker", "pl": "P/L", "ev_ebit": "EV/EBITDA"},
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
