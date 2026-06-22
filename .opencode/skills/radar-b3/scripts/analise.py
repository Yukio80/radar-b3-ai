import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    print(json.dumps({
        "tipo": "radar",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_oportunidades": top,
    }, indent=2, ensure_ascii=False))

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
    print(json.dumps({
        "tipo": "top_dividendos",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "top": top,
    }, indent=2, ensure_ascii=False))

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
    print(json.dumps({
        "tipo": "magic_formula",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "top": top,
    }, indent=2, ensure_ascii=False))

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
    print(json.dumps({
        "tipo": "fiis",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_fiis": top,
    }, indent=2, ensure_ascii=False))

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
    print(json.dumps({
        "tipo": "etfs",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_etfs": top,
    }, indent=2, ensure_ascii=False))

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
    print(json.dumps({
        "tipo": "bdrs",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "total_analisados": len(resultados),
        "top_bdrs": top,
    }, indent=2, ensure_ascii=False))

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

    print(json.dumps({
        "tipo": "carteiras",
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "perfis": carteiras,
    }, indent=2, ensure_ascii=False))

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
    else:
        print("Comandos: radar, analisar <ticker>, dividendos, magic-formula, fiis, analisar-fii <ticker>, etfs, bdrs, carteiras")

if __name__ == "__main__":
    main()
