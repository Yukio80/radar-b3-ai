import sys
import json
import time
import os
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), *[".."] * 4))
from database import salvar_snapshot, salvar_portfolio

BRAPI_BASE = "https://brapi.dev/api"
USER_AGENT = "RadarB3/1.0"

TICKERS_FALLBACK = [
    "PETR4","VALE3","ITUB4","BBDC4","ABEV3","WEGE3","BBAS3",
    "B3SA3","RENT3","LREN3","RAIL3","JBSS3","SUZB3","GGBR4",
    "EMBR3","CSNA3","CMIG4","TOTS3","VIVT3","ELET6","SANB11",
    "FLRY3","HAPV3","RADL3","UGPA3","PRIO3","EQTL3","ENGI11",
    "CPFE3","SBSP3","TAEE11","TRPL4","EGIE3","BBSE3","CYRE3",
    "MRVE3","EZTC3","MULT3","IGTI11","BRML3","ALOS3","ARZZ3",
    "TUPY3","FRAS3","KEPL3","POMO4","TEND3","VIVA3","SOMA3",
    "NTCO3","CVCB3","GOLL4","AZUL4","CCRO3","ECOR3","STBP3",
    "BRFS3","MRFG3","BEEF3","AGRO3","SLCE3","TTEN3","UNIP6",
    "TRAD3","MYPK3","RANI3","DOHL4","EALT4","TKNO4","VULC3",
    "WHLM3","WIZC3","BMOB3","ENMT3","LIGT3","AURE3","AFLT3",
    "BRAP4","CLSC4","FESA4","METB3","MTSA4","PDTC3","PTNT4",
    "ROMI3","SHUL4","TECN3","TXRX3","YDUQ3","ZAMP3","AMAR3",
    "ASHI3","BTTL3","CAML3","CBEE3","CEAB3","CIEL3",
]

TICKERS_FII = [
    "KNRI11","HGLG11","XPML11","XPLG11","VISC11","LVBI11",
    "BTLG11","BRCO11","FIIB11","GGRC11","HFOF11","HGRU11",
    "IRDM11","KFOF11","MALL11","PRSN11","RBRR11","RCRB11",
    "RECR11","RZTR11","SNAG11","TORD11","VINO11","VTLT11",
    "HGPO11","HSML11","JSRE11","MXRF11","PVBI11","RBVA11",
    "RNDP11","SDIL11","VRTA11","ALZR11","CPTS11","FAMB11",
    "HABT11","HBCR11","HCHG11","HCRI11","MCCI11","MGFF11",
    "RBBR11","RBRF11","RBRP11","RBRX11","RLOG11",
    "RVBI11","SPTW11","TRXF11","VCJR11","VGHF11",
    "VGIP11","VILG11","VISC11","VVCR11","XPCA11",
]

TICKERS_ETF = [
    "IVVB11","BOVA11","SPXI11","SMAL11","DIVO11",
    "XINA11","NASD11","USDB11","PIBB11","HASH11",
    "QBTC11","BITI11","ETHE11","SOLA11","WEB311",
    "ACWI11","BIZD11","BSCO11","CXSE11","ECOO11",
    "ESGB11","EURP11","FIXA11","GOLD11","JAPO11",
    "KWDB11","MCHF11","MILA11","QDFI11","SCVB11",
    "TECK11","TRIG11","URET11","WRLD11","XFIX11",
    "XMAL11","BOVV11","BRAX11","FIND11","MATB11",
]

TICKERS_BDR = [
    "AAPL34","GOOG34","MSFT34","AMZO34","META34",
    "NVDC34","TSLA34","DISB34","NFLX34","COCA34",
    "PEPB34","MCDB34","SBUB34","JNJB34","PGCO34",
    "BUDD34","BABA34","ITLC34","BIDU34","CSCO34",
    "ORCL34","INTU34","ADBE34","SAPB34","AMGN34",
    "GILD34","JPMC34","BACB34","WALB34","VISA34",
    "MAST34","PYPL34","XPLG34","KOFL34","NVDA34",
    "ABNB34","SPOT34","QCOM34","TXN34","NEEB34",
    "UNH34","HDZB34","LMTB34","BAES34","CATB34",
]

def buscar_tickers():
    try:
        r = requests.get(f"{BRAPI_BASE}/available",
                         headers={"User-Agent": USER_AGENT}, timeout=10)
        if r.status_code == 200:
            return r.json().get("stocks", [])
    except Exception:
        pass
    return TICKERS_FALLBACK

def info_yfinance(ticker):
    try:
        import yfinance as yf
        t = yf.Ticker(ticker + ".SA")
        info = t.info
        if not info or info.get("regularMarketPrice") is None:
            return None
        hist = t.history(period="6mo")
        tec = {}
        if not hist.empty:
            close = hist["Close"]
            c = float(close.iloc[-1])
            def sma(p):
                v = close.rolling(window=p).mean()
                return round(float(v.iloc[-1]), 2) if not v.empty else None
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta.where(delta < 0, 0.0))
            ag = gain.rolling(window=14).mean()
            al = loss.rolling(window=14).mean()
            rs = ag / al.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            sig = macd.ewm(span=9).mean()
            tec = {
                "preco_atual": round(c, 2),
                "sma20": sma(20), "sma200": sma(200),
                "rsi14": round(float(rsi.iloc[-1]), 2) if not rsi.empty else None,
                "macd": round(float(macd.iloc[-1]), 4) if not macd.empty else None,
                "macd_sinal": round(float(sig.iloc[-1]), 4) if not sig.empty else None,
                "max_6m": round(float(close.max()), 2),
                "min_6m": round(float(close.min()), 2),
                "variacao_6m": round((c / float(close.iloc[0]) - 1) * 100, 2),
            }
        dy = _num(info.get("dividendYield"))
        if dy is not None and dy > 1:
            dy = dy / 100.0
        dtop = _num(info.get("debtToEquity"))
        if dtop is not None and dtop > 10:
            dtop = dtop / 100.0
        return {
            "ticker": ticker,
            "empresa": info.get("longName", ""),
            "preco": info.get("regularMarketPrice"),
            "variacao_dia": info.get("regularMarketChangePercent"),
            "setor": info.get("sector"),
            "industria": info.get("industry"),
            "fundamentos": {
                "pl": info.get("trailingPE"),
                "pvp": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "dy": dy,
                "ev_ebit": info.get("enterpriseToEbitda"),
                "ev_receita": info.get("enterpriseToRevenue"),
                "margem_liquida": info.get("profitMargins"),
                "divida_pl": dtop,
                "liquidez_corrente": info.get("currentRatio"),
                "valor_mercado": info.get("marketCap"),
                "beta": info.get("beta"),
                "vpa": info.get("bookValue"),
                "eps": info.get("trailingEps"),
                "p_ativos": info.get("priceToBook"),
                "payout": info.get("payoutRatio"),
            },
            "tecnicos": tec,
        }
    except Exception as e:
        return None

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def score_fundamental(f):
    s = 50
    pl = _num(f.get("pl"))
    if pl is not None and pl > 0:
        s += 20 if pl < 8 else (15 if pl < 12 else (5 if pl < 18 else -5))
    elif pl is not None and pl < 0:
        s -= 10
    pvp = _num(f.get("pvp"))
    if pvp is not None and pvp > 0:
        s += 20 if pvp < 0.8 else (15 if pvp < 1.2 else (10 if pvp < 2 else (5 if pvp < 3 else -10)))
    roe = _num(f.get("roe"))
    if roe is not None:
        s += 20 if roe > 0.20 else (15 if roe > 0.15 else (10 if roe > 0.10 else (5 if roe > 0.05 else 0)))
    dy = _num(f.get("dy"))
    if dy is not None:
        s += 20 if dy > 0.08 else (15 if dy > 0.05 else (10 if dy > 0.03 else (5 if dy > 0.01 else 0)))
    ev_ebit = _num(f.get("ev_ebit"))
    if ev_ebit is not None and ev_ebit > 0:
        s += 15 if ev_ebit < 6 else (10 if ev_ebit < 10 else (5 if ev_ebit < 15 else -5))
    ml = _num(f.get("margem_liquida"))
    if ml is not None:
        s += 10 if ml > 0.15 else (5 if ml > 0.08 else (-5 if ml < 0 else 0))
    dl_pl = _num(f.get("divida_pl"))
    if dl_pl is not None:
        s += 10 if dl_pl < 0.5 else (5 if dl_pl < 1.0 else (-10 if dl_pl > 3 else 0))
    return min(max(s, 0), 100)

def score_tecnico(t):
    if not t:
        return 50
    s = 50
    rsi = t.get("rsi14")
    if rsi:
        s += 20 if 30 <= rsi <= 45 else (15 if 45 < rsi <= 55 else (5 if 55 < rsi <= 65 else (10 if rsi < 25 else (-10 if rsi > 75 else 0))))
    preco = t.get("preco_atual")
    sma20 = t.get("sma20")
    sma200 = t.get("sma200")
    if preco and sma20:
        s += 10 if preco > sma20 else -5
    if preco and sma200:
        s += 10 if preco > sma200 else -5
    macd = t.get("macd")
    macd_s = t.get("macd_sinal")
    if macd is not None and macd_s is not None:
        s += 10 if macd > macd_s else -5
    return min(max(s, 0), 100)

def score_fii(f):
    s = 50
    dy = _num(f.get("dy"))
    if dy is not None:
        s += 30 if dy > 0.10 else (25 if dy > 0.08 else (15 if dy > 0.06 else (10 if dy > 0.04 else 5)))
    pvp = _num(f.get("pvp"))
    if pvp is not None and pvp > 0:
        s += 25 if pvp < 0.85 else (20 if pvp < 1.0 else (10 if pvp < 1.2 else (0 if pvp < 1.5 else -10)))
    valor_mercado = _num(f.get("valor_mercado"))
    if valor_mercado is not None:
        if valor_mercado > 2e9:
            s += 15
        elif valor_mercado > 1e9:
            s += 10
        elif valor_mercado > 500e6:
            s += 5
    pl = f.get("pl")
    if pl and pl > 0:
        s += 10 if pl < 10 else 5
    return min(max(s, 0), 100)

def info_fii(ticker):
    try:
        import yfinance as yf
        t = yf.Ticker(ticker + ".SA")
        info = t.info
        if not info or info.get("regularMarketPrice") is None:
            return None
        hist = t.history(period="6mo")
        tec = {}
        if not hist.empty:
            close = hist["Close"]
            c = float(close.iloc[-1])
            def sma(p):
                v = close.rolling(window=p).mean()
                return round(float(v.iloc[-1]), 2) if not v.empty else None
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta.where(delta < 0, 0.0))
            ag = gain.rolling(window=14).mean()
            al = loss.rolling(window=14).mean()
            rs = ag / al.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            sig = macd.ewm(span=9).mean()
            tec = {
                "preco_atual": round(c, 2),
                "sma20": sma(20), "sma200": sma(200),
                "rsi14": round(float(rsi.iloc[-1]), 2) if not rsi.empty else None,
                "macd": round(float(macd.iloc[-1]), 4) if not macd.empty else None,
                "macd_sinal": round(float(sig.iloc[-1]), 4) if not sig.empty else None,
                "max_6m": round(float(close.max()), 2),
                "min_6m": round(float(close.min()), 2),
                "variacao_6m": round((c / float(close.iloc[0]) - 1) * 100, 2),
            }
        dy = _num(info.get("dividendYield"))
        if dy is not None and dy > 1:
            dy = dy / 100.0
        return {
            "ticker": ticker,
            "empresa": info.get("longName", ""),
            "preco": info.get("regularMarketPrice"),
            "variacao_dia": info.get("regularMarketChangePercent"),
            "setor": "FII",
            "industria": info.get("industry", "Fundos Imobiliários"),
            "fundamentos": {
                "pvp": info.get("priceToBook"),
                "dy": dy,
                "valor_mercado": info.get("marketCap"),
                "vpa": info.get("bookValue"),
                "pl": info.get("trailingPE"),
                "liquidez_corrente": info.get("currentRatio"),
            },
            "tecnicos": tec,
        }
    except Exception as e:
        return None

def analisar_fii(ticker):
    d = info_fii(ticker)
    if d is None:
        return None
    fund = d["fundamentos"]
    tec = d["tecnicos"]
    sf = score_fii(fund)
    st = score_tecnico(tec)
    return {
        "ticker": ticker,
        "empresa": d["empresa"],
        "preco": d["preco"],
        "variacao_dia": d.get("variacao_dia"),
        "setor": "FII",
        "score_fundamental": sf,
        "score_tecnico": st,
        "score_total": round(sf * 0.6 + st * 0.4, 1),
        "fundamentos": {k: v for k, v in fund.items() if v is not None},
        "tecnicos": tec,
    }

def score_etf(f):
    s = 50
    dy = _num(f.get("dy"))
    if dy is not None:
        s += 30 if dy > 0.08 else (20 if dy > 0.05 else (15 if dy > 0.03 else (10 if dy > 0.01 else 5)))
    pvp = _num(f.get("pvp"))
    if pvp is not None and pvp > 0:
        s += 25 if pvp < 1.0 else (20 if pvp < 1.5 else (10 if pvp < 2.0 else 5))
    valor_mercado = _num(f.get("valor_mercado"))
    if valor_mercado is not None:
        if valor_mercado > 5e9:
            s += 15
        elif valor_mercado > 1e9:
            s += 10
        elif valor_mercado > 100e6:
            s += 5
    pl = f.get("pl")
    if pl and pl > 0:
        s += 10 if pl < 10 else 5
    return min(max(s, 0), 100)

def info_etf(ticker):
    try:
        import yfinance as yf
        t = yf.Ticker(ticker + ".SA")
        info = t.info
        if not info or info.get("regularMarketPrice") is None:
            return None
        hist = t.history(period="6mo")
        tec = {}
        if not hist.empty:
            close = hist["Close"]
            c = float(close.iloc[-1])
            def sma(p):
                v = close.rolling(window=p).mean()
                return round(float(v.iloc[-1]), 2) if not v.empty else None
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta.where(delta < 0, 0.0))
            ag = gain.rolling(window=14).mean()
            al = loss.rolling(window=14).mean()
            rs = ag / al.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            sig = macd.ewm(span=9).mean()
            tec = {
                "preco_atual": round(c, 2),
                "sma20": sma(20), "sma200": sma(200),
                "rsi14": round(float(rsi.iloc[-1]), 2) if not rsi.empty else None,
                "macd": round(float(macd.iloc[-1]), 4) if not macd.empty else None,
                "macd_sinal": round(float(sig.iloc[-1]), 4) if not sig.empty else None,
                "max_6m": round(float(close.max()), 2),
                "min_6m": round(float(close.min()), 2),
                "variacao_6m": round((c / float(close.iloc[0]) - 1) * 100, 2),
            }
        dy = _num(info.get("dividendYield"))
        if dy is not None and dy > 1:
            dy = dy / 100.0
        return {
            "ticker": ticker,
            "empresa": info.get("longName", ""),
            "preco": info.get("regularMarketPrice"),
            "variacao_dia": info.get("regularMarketChangePercent"),
            "setor": "ETF",
            "industria": info.get("industry", "Exchange Traded Fund"),
            "fundamentos": {
                "pvp": info.get("priceToBook"),
                "dy": dy,
                "valor_mercado": info.get("marketCap"),
                "vpa": info.get("bookValue"),
                "pl": info.get("trailingPE"),
            },
            "tecnicos": tec,
        }
    except Exception as e:
        return None

def analisar_etf(ticker):
    d = info_etf(ticker)
    if d is None:
        return None
    fund = d["fundamentos"]
    tec = d["tecnicos"]
    sf = score_etf(fund)
    st = score_tecnico(tec)
    return {
        "ticker": ticker,
        "empresa": d["empresa"],
        "preco": d["preco"],
        "variacao_dia": d.get("variacao_dia"),
        "setor": "ETF",
        "score_fundamental": sf,
        "score_tecnico": st,
        "score_total": round(sf * 0.6 + st * 0.4, 1),
        "fundamentos": {k: v for k, v in fund.items() if v is not None},
        "tecnicos": tec,
    }

def score_bdr(f):
    s = 50
    pl = _num(f.get("pl"))
    if pl is not None and pl > 0:
        s += 25 if pl < 10 else (15 if pl < 15 else (10 if pl < 20 else (5 if pl < 30 else -5)))
    elif pl is not None and pl < 0:
        s -= 10
    pvp = _num(f.get("pvp"))
    if pvp is not None and pvp > 0:
        s += 20 if pvp < 2 else (15 if pvp < 5 else (10 if pvp < 10 else (5 if pvp < 20 else 0)))
    roe = _num(f.get("roe"))
    if roe is not None:
        s += 15 if roe > 0.30 else (10 if roe > 0.15 else (5 if roe > 0.05 else 0))
    dy = _num(f.get("dy"))
    if dy is not None:
        s += 15 if dy > 0.03 else (10 if dy > 0.01 else 5)
    valor_mercado = _num(f.get("valor_mercado"))
    if valor_mercado is not None:
        if valor_mercado > 1e12:
            s += 10
        elif valor_mercado > 100e9:
            s += 5
    ml = _num(f.get("margem_liquida"))
    if ml is not None:
        s += 10 if ml > 0.20 else (5 if ml > 0.10 else 0)
    return min(max(s, 0), 100)

def analisar_bdr(ticker):
    d = info_yfinance(ticker)
    if d is None:
        return None
    fund = d["fundamentos"]
    tec = d["tecnicos"]
    sf = score_bdr(fund)
    st = score_tecnico(tec)
    return {
        "ticker": ticker,
        "empresa": d["empresa"],
        "preco": d["preco"],
        "variacao_dia": d.get("variacao_dia"),
        "setor": "BDR",
        "industria": d.get("industria"),
        "score_fundamental": sf,
        "score_tecnico": st,
        "score_total": round(sf * 0.6 + st * 0.4, 1),
        "fundamentos": {k: v for k, v in fund.items() if v is not None},
        "tecnicos": tec,
    }

def analisar_ticker(ticker):
    d = info_yfinance(ticker)
    if d is None:
        return None
    fund = d["fundamentos"]
    tec = d["tecnicos"]
    sf = score_fundamental(fund)
    st = score_tecnico(tec)
    return {
        "ticker": ticker,
        "empresa": d["empresa"],
        "preco": d["preco"],
        "variacao_dia": d.get("variacao_dia"),
        "setor": d.get("setor"),
        "score_fundamental": sf,
        "score_tecnico": st,
        "score_total": round(sf * 0.6 + st * 0.4, 1),
        "fundamentos": {k: v for k, v in fund.items() if v is not None},
        "tecnicos": tec,
    }

def comando_radar():
    print("📡 Buscando tickers da B3...", file=sys.stderr)
    tickers = buscar_tickers()[:80]
    print(f"📊 Analisando {len(tickers)} ativos via Yahoo Finance...", file=sys.stderr)
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_ticker, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futuros), 1):
            res = fut.result()
            if res:
                resultados.append(res)
            if i % 10 == 0:
                print(f"  Progresso: {i}/{len(tickers)}", file=sys.stderr)
    if not resultados:
        print("❌ Nenhum dado obtido.", file=sys.stderr)
        return
    df = pd.DataFrame(resultados)
    df = df.sort_values("score_total", ascending=False)
    top = df.head(20).to_dict(orient="records")
    saida = {
        "tipo": "radar",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_oportunidades": top,
    }
    salvar_snapshot("radar", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

def comando_analisar(ticker):
    ticker = ticker.upper().strip()
    print(f"🔍 Analisando {ticker}...", file=sys.stderr)
    d = info_yfinance(ticker)
    if d is None:
        print(f"❌ Ticker {ticker} não encontrado.", file=sys.stderr)
        return
    fund = d["fundamentos"]
    tec = d["tecnicos"]
    sf = score_fundamental(fund)
    st = score_tecnico(tec)
    result = {
        "tipo": "analise_detalhada",
        "ticker": ticker,
        "empresa": d["empresa"],
        "preco": d["preco"],
        "variacao_dia": d.get("variacao_dia"),
        "setor": d.get("setor"),
        "industria": d.get("industria"),
        "score_fundamental": sf,
        "score_tecnico": st,
        "score_total": round(sf * 0.6 + st * 0.4, 1),
        "fundamentos": {k: v for k, v in fund.items() if v is not None},
        "tecnicos": tec,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

def comando_dividendos():
    print("🔄 Buscando dados de dividendos...", file=sys.stderr)
    tickers = buscar_tickers()[:80]
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(info_yfinance, t): t for t in tickers}
        for fut in as_completed(futuros):
            d = fut.result()
            if d and d["fundamentos"].get("dy") and d["fundamentos"]["dy"] > 0:
                resultados.append({
                    "ticker": d["ticker"],
                    "empresa": d["empresa"],
                    "dy": round(d["fundamentos"]["dy"], 4),
                    "preco": d["preco"],
                    "pl": d["fundamentos"].get("pl"),
                    "pvp": d["fundamentos"].get("pvp"),
                })
    if not resultados:
        print("❌ Nenhum dado de dividendos.", file=sys.stderr)
        return
    df = pd.DataFrame(resultados)
    df = df.sort_values("dy", ascending=False)
    top = df.head(20).to_dict(orient="records")
    saida = {
        "tipo": "top_dividendos",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "top": top,
    }
    salvar_snapshot("dividendos", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

def comando_magic_formula():
    print("🔄 Executando Magic Formula para B3...", file=sys.stderr)
    tickers = buscar_tickers()[:80]
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(info_yfinance, t): t for t in tickers}
        for fut in as_completed(futuros):
            d = fut.result()
            if d is None:
                continue
            f = d["fundamentos"]
            pl = f.get("pl")
            ev_ebit = f.get("ev_ebit")
            if not pl or not ev_ebit or pl <= 0 or ev_ebit <= 0:
                continue
            resultados.append({
                "ticker": d["ticker"],
                "empresa": d["empresa"],
                "preco": d["preco"],
                "pl": pl,
                "ev_ebit": ev_ebit,
                "roe": f.get("roe"),
                "dy": f.get("dy"),
            })
    if not resultados:
        print("❌ Dados insuficientes.", file=sys.stderr)
        return
    df = pd.DataFrame(resultados)
    df["rank_pl"] = df["pl"].rank()
    df["rank_ev_ebit"] = df["ev_ebit"].rank()
    df["magic_score"] = df[["rank_pl", "rank_ev_ebit"]].sum(axis=1).rank()
    df = df.sort_values("magic_score")
    top = df.head(20).to_dict(orient="records")
    saida = {
        "tipo": "magic_formula",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "top": top,
    }
    salvar_snapshot("magic_formula", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

def comando_fiis():
    print("🏢 Buscando FIIs...", file=sys.stderr)
    tickers = TICKERS_FII[:60]
    print(f"📊 Analisando {len(tickers)} FIIs via Yahoo Finance...", file=sys.stderr)
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_fii, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futuros), 1):
            res = fut.result()
            if res:
                resultados.append(res)
            if i % 10 == 0:
                print(f"  Progresso: {i}/{len(tickers)}", file=sys.stderr)
    if not resultados:
        print("❌ Nenhum dado de FII obtido.", file=sys.stderr)
        return
    df = pd.DataFrame(resultados)
    df = df.sort_values("score_total", ascending=False)
    top = df.head(20).to_dict(orient="records")
    saida = {
        "tipo": "fiis",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_fiis": top,
    }
    salvar_snapshot("fiis", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

def comando_analisar_fii(ticker):
    ticker = ticker.upper().strip()
    print(f"🔍 Analisando FII {ticker}...", file=sys.stderr)
    d = info_fii(ticker)
    if d is None:
        print(f"❌ FII {ticker} não encontrado.", file=sys.stderr)
        return
    fund = d["fundamentos"]
    tec = d["tecnicos"]
    sf = score_fii(fund)
    st = score_tecnico(tec)
    result = {
        "tipo": "analise_fii",
        "ticker": ticker,
        "empresa": d["empresa"],
        "preco": d["preco"],
        "variacao_dia": d.get("variacao_dia"),
        "setor": "FII",
        "industria": d.get("industria"),
        "score_fundamental": sf,
        "score_tecnico": st,
        "score_total": round(sf * 0.6 + st * 0.4, 1),
        "fundamentos": {k: v for k, v in fund.items() if v is not None},
        "tecnicos": tec,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

def comando_etfs():
    print("📊 Buscando ETFs...", file=sys.stderr)
    tickers = TICKERS_ETF[:50]
    print(f"Analisando {len(tickers)} ETFs via Yahoo Finance...", file=sys.stderr)
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_etf, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futuros), 1):
            res = fut.result()
            if res:
                resultados.append(res)
            if i % 10 == 0:
                print(f"  Progresso: {i}/{len(tickers)}", file=sys.stderr)
    if not resultados:
        print("❌ Nenhum dado de ETF obtido.", file=sys.stderr)
        return
    df = pd.DataFrame(resultados)
    df = df.sort_values("score_total", ascending=False)
    top = df.head(20).to_dict(orient="records")
    saida = {
        "tipo": "etfs",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_etfs": top,
    }
    salvar_snapshot("etfs", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

def comando_bdrs():
    print("🌎 Buscando BDRs...", file=sys.stderr)
    tickers = TICKERS_BDR[:50]
    print(f"Analisando {len(tickers)} BDRs via Yahoo Finance...", file=sys.stderr)
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_bdr, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futuros), 1):
            res = fut.result()
            if res:
                resultados.append(res)
            if i % 10 == 0:
                print(f"  Progresso: {i}/{len(tickers)}", file=sys.stderr)
    if not resultados:
        print("❌ Nenhum dado de BDR obtido.", file=sys.stderr)
        return
    df = pd.DataFrame(resultados)
    df = df.sort_values("score_total", ascending=False)
    top = df.head(20).to_dict(orient="records")
    saida = {
        "tipo": "bdrs",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_bdrs": top,
    }
    salvar_snapshot("bdrs", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

PERFIS_CARTEIRAS = {
    "conservador": {
        "nome": "Conservador",
        "icone": "🛡️",
        "descricao": "Foco em preservação de capital e renda estável. Maior peso em FIIs e ações de dividendos.",
        "alocacao": {"FIIs": 40, "Ações Dividendos": 30, "ETFs": 20, "BDRs": 10},
        "selecao": {"FIIs": 8, "Ações Dividendos": 6, "ETFs": 4, "BDRs": 2},
    },
    "moderado": {
        "nome": "Moderado",
        "icone": "⚖️",
        "descricao": "Equilíbrio entre segurança e crescimento. Diversificação entre todas as classes.",
        "alocacao": {"Ações": 35, "ETFs": 25, "FIIs": 25, "BDRs": 15},
        "selecao": {"Ações": 7, "ETFs": 5, "FIIs": 5, "BDRs": 3},
    },
    "arrojado": {
        "nome": "Arrojado",
        "icone": "🚀",
        "descricao": "Busca por alta rentabilidade no longo prazo. Maior exposição a BDRs e ações de crescimento.",
        "alocacao": {"BDRs": 40, "Ações": 30, "ETFs": 20, "FIIs": 10},
        "selecao": {"BDRs": 8, "Ações": 6, "ETFs": 4, "FIIs": 2},
    },
    "aposentadoria": {
        "nome": "Aposentadoria",
        "icone": "🏖️",
        "descricao": "Geração de renda passiva consistente. Foco em dividendos e FIIs de alto DY.",
        "alocacao": {"FIIs": 45, "Ações Dividendos": 35, "ETFs": 15, "BDRs": 5},
        "selecao": {"FIIs": 10, "Ações Dividendos": 7, "ETFs": 3, "BDRs": 1},
    },
}

def comando_carteiras():
    print("📊 Montando carteiras recomendadas...", file=sys.stderr)

    print("  Executando análises...", file=sys.stderr)

    acoes = []
    print("  Analisando ações...", file=sys.stderr)
    tickers_acoes = buscar_tickers()[:80]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_ticker, t): t for t in tickers_acoes}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                acoes.append(res)

    fiis = []
    print("  Analisando FIIs...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_fii, t): t for t in TICKERS_FII[:50]}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                fiis.append(res)

    etfs = []
    print("  Analisando ETFs...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_etf, t): t for t in TICKERS_ETF[:40]}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                etfs.append(res)

    bdrs = []
    print("  Analisando BDRs...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_bdr, t): t for t in TICKERS_BDR[:40]}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                bdrs.append(res)

    acoes.sort(key=lambda x: x.get("score_total", 0), reverse=True)
    fiis.sort(key=lambda x: x.get("score_total", 0), reverse=True)
    etfs.sort(key=lambda x: x.get("score_total", 0), reverse=True)
    bdrs.sort(key=lambda x: x.get("score_total", 0), reverse=True)

    acoes_div = [a for a in acoes if a.get("fundamentos", {}).get("dy", 0) or 0 > 0.02]
    acoes_div.sort(key=lambda x: x.get("fundamentos", {}).get("dy", 0) or 0, reverse=True)
    acoes_cresc = [a for a in acoes if a.get("fundamentos", {}).get("dy", 0) or 0 <= 0.02]
    acoes_cresc.sort(key=lambda x: x.get("score_total", 0), reverse=True)

    carteiras = {}
    for chave, perfil in PERFIS_CARTEIRAS.items():
        carteira_ativos = []
        for classe, qtd in perfil["selecao"].items():
            if classe == "FIIs":
                selecionados = fiis[:qtd]
            elif classe == "ETFs":
                selecionados = etfs[:qtd]
            elif classe == "BDRs":
                selecionados = bdrs[:qtd]
            elif classe == "Ações Dividendos":
                selecionados = acoes_div[:qtd]
            elif classe == "Ações":
                selecionados = acoes_cresc[:qtd]
            else:
                continue
            for ativo in selecionados:
                dy = ativo.get("fundamentos", {}).get("dy", 0) or 0
                carteira_ativos.append({
                    "ticker": ativo["ticker"],
                    "empresa": ativo.get("empresa", ""),
                    "tipo": classe,
                    "preco": ativo.get("preco"),
                    "score": ativo.get("score_total", 0),
                    "dy": round(dy * 100, 2),
                    "setor": ativo.get("setor", ""),
                })

        aloc = perfil["alocacao"]
        total_peso = sum(aloc.values())
        pesos_por_tipo = {}
        remaining = total_peso
        tipos = list(aloc.keys())
        for i, tipo in enumerate(tipos):
            count = perfil["selecao"].get(tipo, 1)
            if i == len(tipos) - 1:
                pesos_por_tipo[tipo] = remaining
            else:
                peso = int(aloc[tipo])
                pesos_por_tipo[tipo] = peso
                remaining -= peso

        for ativo in carteira_ativos:
            tipo = ativo["tipo"]
            count = perfil["selecao"].get(tipo, 1)
            ativo["peso"] = round(pesos_por_tipo.get(tipo, 0) / count, 1)

        scores = [a["score"] for a in carteira_ativos if a["score"]]
        score_medio = round(sum(scores) / len(scores), 1) if scores else 0

        dy_total = sum(a["dy"] * a["peso"] for a in carteira_ativos if a["dy"])
        dy_ponderado = round(dy_total / total_peso, 2) if total_peso else 0

        carteiras[chave] = {
            "nome": perfil["nome"],
            "icone": perfil["icone"],
            "descricao": perfil["descricao"],
            "alocacao": aloc,
            "score_medio": score_medio,
            "dy_ponderado": dy_ponderado,
            "total_ativos": len(carteira_ativos),
            "ativos": carteira_ativos,
        }

    saida = {
        "tipo": "carteiras",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "perfis": carteiras,
    }
    salvar_snapshot("carteiras", saida)
    for chave, p in carteiras.items():
        salvar_portfolio(
            chave, p["score_medio"], p["dy_ponderado"],
            p["total_ativos"], p["ativos"], p["alocacao"],
        )
    print(json.dumps(saida, indent=2, ensure_ascii=False))

def comando_qpe():
    print("🧠 Quantitative Portfolio Engine v2", file=sys.stderr)
    print("  Carregando módulos...", file=sys.stderr)
    from qpe.outlier_detection import clean_outliers, outlier_report
    from qpe.multi_factor_score import MultiFactorScore
    from qpe.portfolio_optimizer import PortfolioOptimizer
    from qpe.robustness_index import RobustnessIndex
    from qpe.stress_test import StressTest
    from qpe.explainability import Explainability
    from qpe.report import PortfolioReport

    print("  Analisando ações...", file=sys.stderr)
    tickers = buscar_tickers()[:50]
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_ticker, t): t for t in tickers}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                resultados.append(res)

    if not resultados:
        print("❌ Nenhum dado obtido.", file=sys.stderr)
        return

    df = pd.DataFrame(resultados)

    print("  Tratando outliers...", file=sys.stderr)
    metrics = ["ev_ebit", "margem_liquida"]
    fund_df = df["fundamentos"].apply(pd.Series) if "fundamentos" in df.columns else pd.DataFrame()
    if not fund_df.empty:
        roe_clean = clean_outliers(fund_df, ["roe", "margem_liquida", "ev_ebit", "dy", "divida_pl"], method="iqr", k=1.5)
        outlier_rep = outlier_report(fund_df, ["roe", "dy", "ev_ebit", "margem_liquida"])
    else:
        roe_clean = fund_df
        outlier_rep = {}

    print("  Calculando crescimento (CAGR)...", file=sys.stderr)
    from qpe.growth_factor import GrowthFactor
    gf = GrowthFactor(years=5)
    growth_data = {}
    for t in tickers[:30]:
        growth_data[t] = gf.compute(t)

    print("  Computando score multifatorial...", file=sys.stderr)
    mfs = MultiFactorScore()
    scored_assets = []
    for item in resultados:
        t = item["ticker"]
        f = item.get("fundamentos", {})
        g = growth_data.get(t, {})
        row = {
            "ticker": t,
            "empresa": item.get("empresa", "").strip(),
            "preco": item.get("preco"),
            "setor": item.get("setor", ""),
            "roe": f.get("roe"),
            "roic": f.get("roe"),
            "margem_liquida": f.get("margem_liquida"),
            "pl": f.get("pl"),
            "pvp": f.get("pvp"),
            "ev_ebit": f.get("ev_ebit"),
            "dy": f.get("dy"),
            "dividend_consistency": 0.5,
            "cagr_revenue": g.get("cagr_revenue"),
            "cagr_net_income": g.get("cagr_net_income"),
            "divida_pl": f.get("divida_pl"),
            "liquidez_corrente": f.get("liquidez_corrente"),
        }
        scores = mfs.compute(row)
        row.update(scores)
        scored_assets.append(row)

    score_df = pd.DataFrame(scored_assets)
    if "total_score" in score_df.columns:
        score_df["score_percentil"] = mfs.apply_percentile_ranking(score_df["total_score"])
        score_df["classificacao"] = score_df["score_percentil"].apply(mfs.classify)
        for i, a in enumerate(scored_assets):
            scored_assets[i]["score_percentil"] = float(score_df.iloc[i]["score_percentil"])
            scored_assets[i]["classificacao"] = score_df.iloc[i]["classificacao"]

    print("  Otimizando pesos da carteira...", file=sys.stderr)
    opt = PortfolioOptimizer(peso_min=0.02, peso_max=0.10)
    scores_list = scored_assets[0]["total_score"] if scored_assets else 0
    if not isinstance(scores_list, list):
        scores_list = [a["total_score"] for a in scored_assets]
    tickers_list = [a["ticker"] for a in scored_assets]
    weights_df = opt.optimize(scores_list, tickers_list)
    weights_dict = dict(zip(weights_df["ticker"], weights_df["weight_pct"]))

    categories = {}
    for a in scored_assets:
        s = a.get("setor", "Ações")
        if s in (None, ""):
            s = "Ações"
        categories[a["ticker"]] = s

    print("  Calculando IRP...", file=sys.stderr)
    ri = RobustnessIndex()
    quality_scores = [a.get("quality", 50) for a in scored_assets]
    dy_vals = [a.get("dy") for a in scored_assets]
    debt_vals = [a.get("divida_pl") for a in scored_assets]
    sector_agg = {}
    for t, w in weights_dict.items():
        s = categories.get(t, "Outros")
        sector_agg[s] = sector_agg.get(s, 0) + w
    irp_result = ri.compute(
        num_assets=len(weights_dict),
        quality_scores=quality_scores,
        dy_values=dy_vals,
        debt_values=debt_vals,
        sector_weights=sector_agg,
    )

    print("  Executando stress test...", file=sys.stderr)
    st = StressTest()
    stress_results = st.run_all(weights_dict, categories)

    print("  Gerando explicações...", file=sys.stderr)
    exp = Explainability()
    explanations = exp.batch_explain(scored_assets)

    print("  Gerando relatório...", file=sys.stderr)
    report_gen = PortfolioReport(output_dir=".")
    report_content = report_gen.generate(
        portfolio={
            "weights": weights_dict,
            "assets": scored_assets,
            "score_medio": round(float(np.mean([a["total_score"] for a in scored_assets])), 1),
            "dy_ponderado": round(
                sum(a.get("dy", 0) or 0 * weights_dict.get(a["ticker"], 0)
                    for a in scored_assets) / 100.0, 2
            ) if weights_dict else 0,
        },
        irp_result=irp_result,
        stress_results=stress_results,
        explanations=explanations,
        profile_name="QPE v2",
    )
    report_path = report_gen.save(report_content)

    top10 = sorted(scored_assets, key=lambda x: x["total_score"], reverse=True)[:10]

    saida = {
        "tipo": "qpe_v2",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(scored_assets),
        "outliers": outlier_rep,
        "score_medio": round(float(np.mean([a["total_score"] for a in scored_assets])), 1),
        "carteira": {
            "alocacao": weights_dict,
            "total_ativos": len(weights_dict),
        },
        "irp": irp_result,
        "stress_test": stress_results,
        "top_10": [{
            "ticker": a["ticker"],
            "score": a["total_score"],
            "classificacao": a.get("classificacao", ""),
            "fatores": {
                "qualidade": a.get("quality", 0),
                "valuation": a.get("valuation", 0),
                "dividendos": a.get("dividends", 0),
                "crescimento": a.get("growth", 0),
                "seguranca": a.get("safety", 0),
            },
        } for a in top10],
        "explicacoes": explanations,
        "relatorio": report_path,
    }
    salvar_snapshot("qpe_v2", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

def comando_qpe_validacao():
    """Run QPE v3 validation: backtest, walk-forward, Monte Carlo, regime, correlation."""
    print("🔬 Quantitative Portfolio Engine v3 — Validação", file=sys.stderr)
    print("  1. Obtendo scores QPE v2...", file=sys.stderr)
    from datetime import datetime, timedelta
    import json

    tickers = buscar_tickers()[:50]
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_ticker, t): t for t in tickers}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                resultados.append(res)

    from qpe.multi_factor_score import MultiFactorScore
    from qpe.growth_factor import GrowthFactor
    mfs = MultiFactorScore()
    gf = GrowthFactor(years=5)
    growth_data = {}
    for t in tickers[:30]:
        growth_data[t] = gf.compute(t)

    scored_assets = []
    for item in resultados:
        t = item["ticker"]
        f = item.get("fundamentos", {})
        g = growth_data.get(t, {})
        row = {
            "ticker": t, "empresa": item.get("empresa", "").strip(),
            "preco": item.get("preco"), "setor": item.get("setor", ""),
            "roe": f.get("roe"), "roic": f.get("roe"),
            "margem_liquida": f.get("margem_liquida"),
            "pl": f.get("pl"), "pvp": f.get("pvp"),
            "ev_ebit": f.get("ev_ebit"), "dy": f.get("dy"),
            "dividend_consistency": 0.5,
            "cagr_revenue": g.get("cagr_revenue"),
            "cagr_net_income": g.get("cagr_net_income"),
            "divida_pl": f.get("divida_pl"),
            "liquidez_corrente": f.get("liquidez_corrente"),
        }
        scores = mfs.compute(row)
        row.update(scores)
        scored_assets.append(row)

    ticker_scores = {}
    for a in scored_assets:
        t = a["ticker"]
        if a.get("total_score"):
            ticker_scores[t] = a["total_score"]
            ticker_scores[t + ".SA"] = a["total_score"]
    sectors = {a["ticker"]: a.get("setor", "Outros") for a in scored_assets}

    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("  2. Executando Performance Metrics...", file=sys.stderr)
    from qpe.performance_metrics import PerformanceMetrics
    pm = PerformanceMetrics(risk_free_rate=0.1325)
    pf_metrics = {}

    print("  3. Obtendo dados de benchmark...", file=sys.stderr)
    from qpe.benchmark import BenchmarkEngine
    be = BenchmarkEngine(cdi_rate=0.1325, start_date=start_date)
    bench_data = be.download_all()
    ibov_ret = bench_data.get("IBOV", pd.Series(dtype=float)) if "IBOV" in be._data else be.download("IBOV")
    idiv_ret = bench_data.get("IDIV", pd.Series(dtype=float)) if "IDIV" in be._data else be.download("IDIV")
    cdi_ret = bench_data.get("CDI", pd.Series(dtype=float)) if "CDI" in be._data else be.download("CDI")

    print("  4. Executando Backtest...", file=sys.stderr)
    from qpe.backtesting import BacktestEngine
    bt = BacktestEngine(
        tickers=[t + ".SA" if not t.endswith(".SA") else t for t in ticker_scores.keys()],
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        rebalance_frequency="trimestral",
        top_n=10,
    )
    if bt._prices is None:
        bt.download_data()
    bt_result = bt.run(scores=ticker_scores)

    print("  5. Calculando métricas vs benchmarks...", file=sys.stderr)
    bt_returns = bt_result.get("returns", pd.Series(dtype=float))
    metrics_qpe = pm.all_metrics(bt_returns, ibov_ret) if not bt_returns.empty else {}
    metrics_ibov = pm.all_metrics(ibov_ret) if not ibov_ret.empty else {}
    metrics_idiv = pm.all_metrics(idiv_ret) if not idiv_ret.empty else {}
    metrics_cdi = pm.all_metrics(cdi_ret) if not cdi_ret.empty else {}

    print("  6. Executando Walk-Forward Validation...", file=sys.stderr)
    from qpe.walk_forward import WalkForwardValidator
    wf = WalkForwardValidator(
        tickers=[t + ".SA" if not t.endswith(".SA") else t for t in ticker_scores.keys()],
        start_date=start_date,
        end_date=end_date,
        train_years=1,
        test_months=6,
        top_n=10,
    )
    wf_result = wf.validate()

    print("  7. Executando Monte Carlo...", file=sys.stderr)
    from qpe.monte_carlo import MonteCarloEngine
    mc = MonteCarloEngine(num_simulations=5000, horizon_days=252, seed=42)
    ann_ret = metrics_qpe.get("retorno_anualizado", 0.10)
    ann_vol = metrics_qpe.get("volatilidade_anualizada", 0.20)
    cdi_ann = metrics_cdi.get("retorno_anualizado", 0.1325)
    ibov_ann = metrics_ibov.get("retorno_anualizado", 0.10)
    mc_result = mc.full_analysis(
        annual_return=ann_ret if ann_ret != 0 else 0.10,
        annual_volatility=ann_vol if ann_vol != 0 else 0.20,
        cdi_return=cdi_ann,
        ibov_return=ibov_ann,
    )

    print("  8. Detecting market regime...", file=sys.stderr)
    from qpe.regime_detector import RegimeDetector
    rd = RegimeDetector()
    regime_result = rd.detect(ibov_ret, cdi_rate=0.1325)

    print("  9. Analisando correlações...", file=sys.stderr)
    from qpe.correlation_analysis import CorrelationAnalyzer
    ca = CorrelationAnalyzer()
    fator_df = pd.DataFrame([{
        "quality": a.get("quality", 50), "valuation": a.get("valuation", 50),
        "dividends": a.get("dividends", 50), "growth": a.get("growth", 50),
        "safety": a.get("safety", 50),
    } for a in scored_assets])
    factor_corr = ca.factor_correlation(fator_df)

    corr_matrix = factor_corr.get("correlacao", pd.DataFrame())
    avg_corr = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()) if not corr_matrix.empty and len(corr_matrix) > 1 else 0

    print("  10. Gerando relatório...", file=sys.stderr)
    from qpe.reports import save_report, BacktestReport, ValidationReport, PerformanceReport
    os.makedirs("reports", exist_ok=True)

    bench_metrics = {
        "qpe": metrics_qpe,
        "IBOV": metrics_ibov,
        "IDIV": metrics_idiv,
        "CDI": metrics_cdi,
    }
    bt_report = BacktestReport()
    bt_content = bt_report.generate(bt_result, bench_metrics)
    save_report(bt_content, "backtest_report.md")

    perf_report = PerformanceReport()
    perf_content = perf_report.generate(
        metrics_qpe,
        regime_analysis=regime_result,
        correlation_analysis={
            "correlacao_media": avg_corr,
            "diversificacao_efetiva": ca.effective_diversification() if ca.returns is not None else 0,
            "fator_correlacao": factor_corr,
        },
    )
    save_report(perf_content, "performance_report.md")

    val_report = ValidationReport()
    val_content = val_report.generate(wf_result, mc_result)
    save_report(val_content, "validation_report.md")

    saida = {
        "tipo": "qpe_v3_validacao",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_ativos": len(scored_assets),
        "backtest": {
            "retorno_total": bt_result.get("retorno_total", 0),
            "capital_final": bt_result.get("capital_final", 0),
            "qtd_rebalances": bt_result.get("qtd_rebalances", 0),
        },
        "performance": metrics_qpe,
        "benchmark": {
            "ibov": metrics_ibov,
            "idiv": metrics_idiv,
            "cdi": metrics_cdi,
        },
        "walk_forward": wf_result.get("resultados_consolidados", {}),
        "monte_carlo": {
            "var_95": mc_result.get("var_95", 0),
            "var_99": mc_result.get("var_99", 0),
            "probabilidade_perda": mc_result.get("probabilidade_perda", 0),
            "probabilidade_superar_cdi": mc_result.get("probabilidade_superar_cdi", 0),
            "probabilidade_superar_ibov": mc_result.get("probabilidade_superar_ibov", 0),
        },
        "regime": regime_result.get("classificacao", ""),
        "fatores_vif": factor_corr.get("vif", {}),
        "correlacao_media_fatores": avg_corr,
        "relatorios": ["reports/backtest_report.md", "reports/performance_report.md", "reports/validation_report.md"],
    }

    salvar_snapshot("qpe_v3_validacao", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))
    print("\n✅ Relatórios salvos em reports/", file=sys.stderr)


def comando_qpe_v4():
    """Run QPE v4 full pipeline: regime-aware alpha, covariance shrinkage,
    two-stage selection, MV optimization, Black-Litterman, advanced stress."""
    print("🚀 Quantitative Portfolio Engine v4 — Alpha Generation", file=sys.stderr)
    import json, time, os, numpy as np, pandas as pd
    from datetime import datetime, timedelta
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("  1. Coletando dados fundamentalistas...", file=sys.stderr)
    tickers = buscar_tickers()[:50]
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_ticker, t): t for t in tickers}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                resultados.append(res)

    from qpe.multi_factor_score import MultiFactorScore
    from qpe.growth_factor import GrowthFactor
    from qpe.alpha_engine import AlphaEngine
    from qpe.performance_metrics import PerformanceMetrics
    from qpe.benchmark import BenchmarkEngine
    from qpe.backtesting import BacktestEngine
    from qpe.walk_forward import WalkForwardValidator
    from qpe.monte_carlo import MonteCarloEngine
    from qpe.covariance_models import auto_select_covariance
    from qpe.portfolio_construction import TwoStagePortfolioBuilder, MeanVarianceOptimizer
    from qpe.black_litterman import BlackLittermanOptimizer
    from qpe.enhanced_stress import AdvancedStressTest
    from qpe.regime_detector import RegimeDetector
    from qpe.attribution import AlphaAttributionEngine
    from qpe.reports_v4 import (generate_alpha_report, generate_optimization_report,
                                generate_regime_report, generate_attribution_report,
                                generate_v4_validation, save_report as save_v4_report)
    from qpe.correlation_analysis import CorrelationAnalyzer

    mfs = MultiFactorScore()
    gf = GrowthFactor(years=5)
    growth_data = {}
    for t in tickers[:30]:
        growth_data[t] = gf.compute(t)

    scored_assets, sectors = [], {}
    for item in resultados:
        t = item["ticker"]
        f = item.get("fundamentos", {})
        g = growth_data.get(t, {})
        row = {"ticker": t, "setor": item.get("setor", ""),
               "roe": f.get("roe"), "roic": f.get("roe"),
               "margem_liquida": f.get("margem_liquida"),
               "pl": f.get("pl"), "pvp": f.get("pvp"),
               "ev_ebit": f.get("ev_ebit"), "dy": f.get("dy"),
               "dividend_consistency": 0.5,
               "cagr_revenue": g.get("cagr_revenue"),
               "cagr_net_income": g.get("cagr_net_income"),
               "divida_pl": f.get("divida_pl"),
               "liquidez_corrente": f.get("liquidez_corrente")}
        scores = mfs.compute(row)
        row.update(scores)
        scored_assets.append(row)
        sectors[t] = item.get("setor", "Outros")

    ticker_scores = {a["ticker"]: a["total_score"] for a in scored_assets if a.get("total_score")}

    print("  2. Detectando regime de mercado...", file=sys.stderr)
    start_s = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_s = datetime.now().strftime("%Y-%m-%d")
    be = BenchmarkEngine(cdi_rate=0.1325, start_date=start_s)
    ibov_ret = be.download("IBOV")
    rd = RegimeDetector()
    regime_result = rd.detect(ibov_ret, cdi_rate=0.1325)
    current_regime = regime_result.get("regime", "unknown")

    print("  3. Computando alpha scores com ajuste de regime...", file=sys.stderr)
    ae = AlphaEngine()
    factor_assets = [{"ticker": a["ticker"],
                      "quality": a.get("quality", 50),
                      "valuation": a.get("valuation", 50),
                      "dividends": a.get("dividends", 50),
                      "growth": a.get("growth", 50),
                      "safety": a.get("safety", 50)}
                     for a in scored_assets]
    alpha_assets = ae.compute_alpha_batch(factor_assets, regime=current_regime)
    alpha_scores = {a["ticker"]: a["alpha_score"] for a in alpha_assets}
    factor_weights = ae.get_factor_weights(current_regime)

    print("  4. Baixando dados históricos de preços...", file=sys.stderr)
    price_tickers = [t + ".SA" if not t.endswith(".SA") else t for t in ticker_scores.keys()]
    try:
        import yfinance as yf
        px = yf.download(price_tickers, start=start_s, end=end_s, progress=False, auto_adjust=True)
        if isinstance(px.columns, pd.MultiIndex):
            px = px["Close"] if "Close" in px.columns else px
        if isinstance(px, pd.Series):
            px = px.to_frame()
        returns = px.pct_change().dropna(how="all")
        returns.columns = [c.replace(".SA", "") for c in returns.columns]
    except Exception:
        returns = pd.DataFrame()

    print("  5. Construção two-stage da carteira...", file=sys.stderr)
    builder = TwoStagePortfolioBuilder(optim_method="max_sharpe", max_asset_weight=0.08,
                                       max_sector_weight=0.20, target_te=0.08)
    portfolio_result = builder.build(
        tickers=list(ticker_scores.keys()),
        scores=ticker_scores,
        returns=returns,
        sector_map=sectors if sectors else None,
        top_k=30,
        regime_adjusted_scores=alpha_scores,
    )

    stage1 = portfolio_result.get("stage1", {})
    stage2 = portfolio_result.get("stage2", {})
    optimized_weights = stage2.get("pesos", {})

    if not optimized_weights:
        optimized_weights = {t: 1.0/len(ticker_scores) for t in list(ticker_scores.keys())[:20]}

    print("  6. Black-Litterman...", file=sys.stderr)
    bl_result = {}
    bl_tickers = list(optimized_weights.keys())[:20]
    if len(bl_tickers) >= 2 and not returns.empty:
        bl_ret = returns[[c for c in bl_tickers if c in returns.columns]].dropna()
        if bl_ret.shape[1] >= 2 and bl_ret.shape[0] >= 20:
            cov_result = auto_select_covariance(bl_ret.values)
            bl = BlackLittermanOptimizer(max_weight=0.08)
            bl_result = bl.optimize(cov_result.covariance, bl_tickers,
                                    {t: alpha_scores.get(t, 50) for t in bl_tickers})

    print("  7. Performance metrics vs benchmarks...", file=sys.stderr)
    pm = PerformanceMetrics(risk_free_rate=0.1325)
    bt = BacktestEngine(tickers=price_tickers, start_date=start_s, end_date=end_s,
                        initial_capital=100000, rebalance_frequency="trimestral", top_n=10)
    if bt._prices is None:
        bt.download_data()
    bt_result = bt.run(scores={k + ".SA": v for k, v in alpha_scores.items()})
    bt_returns = bt_result.get("returns", pd.Series(dtype=float))
    metrics_qpe = pm.all_metrics(bt_returns, ibov_ret) if not bt_returns.empty else {}
    idiv_ret = be.download("IDIV") if "IDIV" not in be._data else be._data["IDIV"]
    cdi_ret = be.download("CDI") if "CDI" not in be._data else be._data["CDI"]

    print("  8. Walk-Forward...", file=sys.stderr)
    wf = WalkForwardValidator(tickers=price_tickers, start_date=start_s, end_date=end_s,
                              train_years=1, test_months=6, top_n=10, rebalance_frequency="trimestral")
    wf_result = wf.validate()

    print("  9. Monte Carlo...", file=sys.stderr)
    ann_ret = metrics_qpe.get("retorno_anualizado", 0.10) or 0.10
    ann_vol = metrics_qpe.get("volatilidade_anualizada", 0.20) or 0.20
    mc = MonteCarloEngine(num_simulations=5000, horizon_days=252, seed=42)
    mc_result = mc.full_analysis(annual_return=ann_ret, annual_volatility=ann_vol,
                                 cdi_return=0.1325, ibov_return=ibov_ret.mean()*252 if not ibov_ret.empty else 0.10)

    print("  10. Stress test avançado...", file=sys.stderr)
    astress = AdvancedStressTest()
    stress_result = astress.run_all(optimized_weights, sectors)

    print("  11. Alpha attribution...", file=sys.stderr)
    attrib = AlphaAttributionEngine(risk_free_rate=0.1325)
    fator_df = pd.DataFrame([{"ticker": a["ticker"],
                              "quality": a.get("quality", 50),
                              "valuation": a.get("valuation", 50),
                              "dividends": a.get("dividends", 50),
                              "growth": a.get("growth", 50),
                              "safety": a.get("safety", 50)}
                             for a in scored_assets]).set_index("ticker")
    attribution_result = attrib.attribute(factor_weights, fator_df, returns, ibov_ret, optimized_weights)

    print("  12. Correlation analysis...", file=sys.stderr)
    ca = CorrelationAnalyzer()
    corr_result = ca.factor_correlation(fator_df)

    print("  13. Gerando relatórios...", file=sys.stderr)
    os.makedirs("reports", exist_ok=True)
    validation_data = {
        "performance": metrics_qpe,
        "benchmark": {"ibov": pm.all_metrics(ibov_ret) if not ibov_ret.empty else {},
                      "idiv": pm.all_metrics(idiv_ret) if not idiv_ret.empty else {},
                      "cdi": pm.all_metrics(cdi_ret) if not cdi_ret.empty else {}},
        "walk_forward": wf_result.get("resultados_consolidados", {}),
        "monte_carlo": mc_result,
        "regime": current_regime,
        "advanced_stress": stress_result,
    }

    save_v4_report(generate_alpha_report(validation_data), "alpha_report.md")
    save_v4_report(generate_optimization_report(stage1, stage2), "optimization_report.md")
    save_v4_report(generate_regime_report(regime_result, factor_weights, AlphaEngine.BASE_WEIGHTS), "regime_report.md")
    save_v4_report(generate_attribution_report(attribution_result), "attribution_report.md")
    save_v4_report(generate_v4_validation(validation_data), "qpe_v4_validation.md")

    saida = {
        "tipo": "qpe_v4",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "regime": current_regime,
        "performance": metrics_qpe,
        "benchmark": {"ibov_ret": ibov_ret.mean()*252 if not ibov_ret.empty else 0,
                      "cdi_ret": 0.1325},
        "stage1": stage1,
        "stage2": stage2,
        "black_litterman": {"sharpe": bl_result.get("sharpe_esperado", 0)} if bl_result else {},
        "monte_carlo": {"var_95": mc_result.get("var_95", 0),
                        "prob_perda": mc_result.get("probabilidade_perda", 0),
                        "prob_cdi": mc_result.get("probabilidade_superar_cdi", 0)},
        "stress_test_avancado": {"pior_cenario": stress_result.get("pior_cenario", ""),
                                  "classificacao": stress_result.get("classificacao_risco", "")},
        "attribution": {"melhor_fator": attribution_result.get("melhor_fator", ""),
                         "fatores_significativos": attribution_result.get("fatores_significativos", 0)},
        "relatorios": ["reports/alpha_report.md", "reports/optimization_report.md",
                       "reports/regime_report.md", "reports/attribution_report.md",
                       "reports/qpe_v4_validation.md"],
    }

    salvar_snapshot("qpe_v4", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))
    print("\n✅ QPE v4 completo. Relatorios em reports/", file=sys.stderr)


def comando_qpe_carteiras():
    """Run QPE v5 full pipeline: portfolio profiles, conviction scoring,
    market score, recommendation engine, and reports."""
    print("📊 QPE v5 — Carteiras Recomendadas", file=sys.stderr)
    import json, time, os, numpy as np, pandas as pd
    from datetime import datetime, timedelta
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("  1. Coletando dados fundamentalistas...", file=sys.stderr)
    tickers = buscar_tickers()[:50]
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futuros = {ex.submit(analisar_ticker, t): t for t in tickers}
        for fut in as_completed(futuros):
            res = fut.result()
            if res:
                resultados.append(res)

    from qpe.multi_factor_score import MultiFactorScore
    from qpe.growth_factor import GrowthFactor
    from qpe.alpha_engine import AlphaEngine
    from qpe.regime_detector import RegimeDetector
    from qpe.benchmark import BenchmarkEngine
    from qpe.recommendation_engine import RecommendationEngine
    from qpe.recommendation_reports import _carteira_report, generate_market_report, generate_validation_report, save_report as save_rec_report
    from qpe.portfolio_profiles import PROFILES

    mfs = MultiFactorScore()
    gf = GrowthFactor(years=5)
    growth_data = {}
    for t in tickers[:30]:
        growth_data[t] = gf.compute(t)

    scored_assets, sectors = [], {}
    for item in resultados:
        t = item["ticker"]
        f = item.get("fundamentos", {})
        g = growth_data.get(t, {})
        row = {"ticker": t, "setor": item.get("setor", ""),
               "empresa": item.get("empresa", "").strip(),
               "roe": f.get("roe"), "roic": f.get("roe"),
               "margem_liquida": f.get("margem_liquida"),
               "pl": f.get("pl"), "pvp": f.get("pvp"),
               "ev_ebit": f.get("ev_ebit"), "dy": f.get("dy"),
               "dividend_consistency": 0.5,
               "cagr_revenue": g.get("cagr_revenue"),
               "cagr_net_income": g.get("cagr_net_income"),
               "divida_pl": f.get("divida_pl"),
               "liquidez_corrente": f.get("liquidez_corrente")}
        scores = mfs.compute(row)
        row.update(scores)
        scored_assets.append(row)
        sectors[t] = item.get("setor", "Outros")

    start_s = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_s = datetime.now().strftime("%Y-%m-%d")

    print("  2. Detectando regime de mercado...", file=sys.stderr)
    be = BenchmarkEngine(cdi_rate=0.1325, start_date=start_s)
    ibov_ret = be.download("IBOV")
    rd = RegimeDetector()
    regime_result = rd.detect(ibov_ret, cdi_rate=0.1325)
    current_regime = regime_result.get("regime", "unknown")

    print("  3. Computando alpha scores...", file=sys.stderr)
    ae = AlphaEngine()
    for a in scored_assets:
        alpha = ae.compute_alpha({
            "quality": a.get("quality", 50),
            "valuation": a.get("valuation", 50),
            "dividends": a.get("dividends", 50),
            "growth": a.get("growth", 50),
            "safety": a.get("safety", 50),
        })
        a["alpha_score"] = alpha

    print("  4. Baixando dados históricos...", file=sys.stderr)
    price_tickers = [t + ".SA" if not t.endswith(".SA") else t for t in [a["ticker"] for a in scored_assets]]
    try:
        import yfinance as yf
        px = yf.download(price_tickers, start=start_s, end=end_s, progress=False, auto_adjust=True)
        if isinstance(px.columns, pd.MultiIndex):
            px = px["Close"] if "Close" in px.columns else px
        if isinstance(px, pd.Series):
            px = px.to_frame()
        returns = px.pct_change().dropna(how="all")
        returns.columns = [c.replace(".SA", "") for c in returns.columns]
    except Exception:
        returns = pd.DataFrame()

    print("  5. Gerando recomendações para todos os perfis...", file=sys.stderr)
    engine = RecommendationEngine()
    all_recs = engine.recommend_all(
        scored_assets=scored_assets,
        regime=current_regime,
        returns=returns,
        benchmark_returns=ibov_ret,
    )

    print("  6. Salvando relatórios...", file=sys.stderr)
    os.makedirs("reports", exist_ok=True)
    consolidated = {}
    for name, rec in all_recs.items():
        profile_name = rec.profile.lower().replace(" ", "_")
        content = _carteira_report(rec, profile_name, current_regime)
        save_rec_report(content, f"carteira_{profile_name}.md")
        consolidated[name] = {
            "profile": rec.profile,
            "score_medio": rec.score_medio,
            "conviction_media": rec.conviction_media,
            "sharpe": rec.metrics.get("sharpe_ratio", 0),
            "irp": rec.irp_result.get("IRP", 0),
            "alpha": rec.metrics.get("alpha", 0),
            "drawdown": rec.metrics.get("max_drawdown", 0),
            "num_ativos": len(rec.positions),
            "regime": current_regime,
        }

    market_score = engine._compute_market_score(scored_assets, current_regime)
    market_content = generate_market_report(
        market_score, current_regime,
        RegimeDetector.regime_description(current_regime),
        consolidated,
    )
    save_rec_report(market_content, "market_report.md")
    val_content = generate_validation_report(consolidated)
    save_rec_report(val_content, "qpe_v5_validation.md")

    saida = {
        "tipo": "qpe_v5_carteiras",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "regime": current_regime,
        "market_score": market_score,
        "consolidado": consolidated,
        "relatorios": [
            f"reports/carteira_{p.lower().replace(' ', '_')}.md" for p in PROFILES
        ] + ["reports/market_report.md", "reports/qpe_v5_validation.md"],
    }
    salvar_snapshot("qpe_v5_carteiras", saida)
    print(json.dumps(saida, indent=2, ensure_ascii=False))
    print("\n✅ QPE v5 completo. Relatorios em reports/", file=sys.stderr)


def main():
    args = sys.argv[1:]
    if not args or args[0] == "radar":
        comando_radar()
    elif args[0] == "analisar" and len(args) > 1:
        comando_analisar(args[1])
    elif args[0] == "dividendos":
        comando_dividendos()
    elif args[0] == "magic-formula":
        comando_magic_formula()
    elif args[0] == "fiis":
        comando_fiis()
    elif args[0] == "analisar-fii" and len(args) > 1:
        comando_analisar_fii(args[1])
    elif args[0] == "etfs":
        comando_etfs()
    elif args[0] == "bdrs":
        comando_bdrs()
    elif args[0] == "carteiras":
        comando_carteiras()
    elif args[0] == "qpe":
        comando_qpe()
    elif args[0] == "qpe-validacao":
        comando_qpe_validacao()
    elif args[0] == "qpe-v4":
        comando_qpe_v4()
    elif args[0] == "qpe-carteiras":
        comando_qpe_carteiras()
    else:
        print("Comandos: radar, analisar <ticker>, dividendos, magic-formula, fiis, analisar-fii <ticker>, etfs, bdrs, carteiras, qpe, qpe-validacao, qpe-v4, qpe-carteiras")

if __name__ == "__main__":
    main()
