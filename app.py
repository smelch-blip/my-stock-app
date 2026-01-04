import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime

# =========================
# UI + CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Wealth Architect Pro (NSE)")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; width: 100%; height: 3em; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Wealth Architect Pro")
st.caption("NSE Portfolio Audit | Version 2.5 (Stable & Rate-Limit Optimized)")

# --- UNIFIED COLUMN CONSTANTS ---
C_COMP, C_LTP = "Company", "LTP"
C_50, C_150, C_200 = "50DMA", "150DMA", "200DMA"
C_SALES, C_PROFIT = "Sales Growth %", "Profit Growth %"
C_ROE, C_ROCE = "ROE %", "ROCE %"
C_PB, C_FAIR, C_MOS, C_METH = "P/B Ratio", "Fair Value", "MoS Buy Price", "Method"
C_MOM, C_REC, C_REAS = "Momentum", "Recommendation", "Reason"

G_TECH, G_FUND, G_VAL, G_DEC, G_NONE = "Technicals", "Fundamentals", "Valuation", "Decision", ""

ALL_COLS = [C_COMP, C_LTP, C_50, C_150, C_200, C_SALES, C_PROFIT, C_ROE, C_ROCE, C_PB, C_FAIR, C_MOS, C_METH, C_MOM, C_REC, C_REAS]
DISPLAY_MAP = [
    (G_NONE, C_COMP), (G_NONE, C_LTP),
    (G_TECH, C_50), (G_TECH, C_150), (G_TECH, C_200),
    (G_FUND, C_SALES), (G_FUND, C_PROFIT), (G_FUND, C_ROE), (G_FUND, C_ROCE),
    (G_VAL, C_PB), (G_VAL, C_FAIR), (G_VAL, C_MOS), (G_VAL, C_METH),
    (G_DEC, C_MOM), (G_DEC, C_REC), (G_DEC, C_REAS)
]

# =========================
# HELPERS
# =========================
def _clean_nse_symbol(sym: str) -> str:
    sym = str(sym or "").strip().upper()
    if not sym: return ""
    if not (sym.endswith(".NS") or sym.endswith(".BO")): sym += ".NS"
    return sym

def _safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except: return None

def _cagr(a, b, years: float):
    a, b = _safe_float(a), _safe_float(b)
    if a is None or b is None or a <= 0 or b <= 0 or years <= 0: return None
    return (b / a) ** (1.0 / years) - 1.0

# =========================
# DATA FETCHING (Rate-Limit & NoneType Safe)
# =========================
def fetch_fundamentals_one(ticker: str):
    time.sleep(0.5) # Anti-Rate Limit Pause
    t = yf.Ticker(ticker)
    info = t.info or {}
    
    res = {
        "roe": _safe_float(info.get("returnOnEquity")), 
        "pb": _safe_float(info.get("priceToBook")),
        "eps": _safe_float(info.get("forwardEps")) or _safe_float(info.get("trailingEps")),
        "book": _safe_float(info.get("bookValue")), 
        "s_3y": None, "p_3y": None, "roce": None,
        "sect": info.get("sector", "Default"), 
        "ind": info.get("industry", "Default")
    }

    try:
        inc = t.income_stmt
        bal = t.balance_sheet
        
        # Safe check for Financial Statements
        if inc is not None and not inc.empty and isinstance(inc, pd.DataFrame):
            if "Total Revenue" in inc.index:
                rev = inc.loc["Total Revenue"].dropna()
                if len(rev) >= 2: res["s_3y"] = _cagr(rev.iloc[-1], rev.iloc[0], len(rev)-1)
            
            if "Net Income" in inc.index:
                ni = inc.loc["Net Income"].dropna()
                if len(ni) >= 2: res["p_3y"] = _cagr(ni.iloc[-1], ni.iloc[0], len(ni)-1)
            
            # Safe ROCE calculation
            if bal is not None and not bal.empty and isinstance(bal, pd.DataFrame):
                if "EBIT" in inc.index and "Total Assets" in bal.index:
                    ebit = _safe_float(inc.loc["EBIT"].iloc[0])
                    assets = _safe_float(bal.loc["Total Assets"].iloc[0])
                    liab = _safe_float(bal.loc["Current Liabilities"].iloc[0]) if "Current Liabilities" in bal.index else 0
                    
                    if ebit is not None and assets is not None:
                        cap_emp = assets - liab
                        if cap_emp > 0: res["roce"] = ebit / cap_emp
    except Exception:
        pass # Silently skip errors for incomplete financial data
    return res

# =========================
# ANALYSIS LOGIC
# =========================
def run_valuation(f, ltp, mos_pct):
    sect, ind = f.get("sect", "Default"), f.get("ind", "Default")
    text = f"{sect} {ind}".lower()
    bucket = "Default"
    if any(k in text for k in ["bank", "nbfc", "financial"]): bucket = "Financial"
    elif any(k in text for k in ["software", "technology"]): bucket = "Technology"

    dcf = (f["eps"] * 3.7908) + (f["eps"] * 10 / (1.1**5)) if f["eps"] else None
    pb_v = (f["book"] * (f["roe"]/0.12)) if f["book"] and f["roe"] else None
    
    if bucket == "Financial": w = {"DCF": 0.05, "PB": 0.95}
    elif bucket == "Technology": w = {"DCF": 0.85, "PB": 0.15}
    else: w = {"DCF": 0.5, "PB": 0.5}
    
    vals = [v * w[k] for k, v in [("DCF", dcf), ("PB", pb_v)] if v]
    weights = [w[k] for k, v in [("DCF", dcf), ("PB", pb_v)] if v]
    fair = (sum(vals) / sum(weights)) if weights else (ltp or 0)
    
    if f["roce"] and f["roce"] > 0.22: fair *= 1.15
    return f["pb"], fair, fair * (1 - mos_pct/100), bucket

# =========================
# MAIN APP
# =========================
with st.sidebar:
    st.header("Settings")
    mos_in = st.slider("Margin of Safety %", 5, 40, 25)
    # REDUCED WORKERS TO PREVENT RATE LIMITING
    work = st.slider("Threads (Keep low to avoid blocks)", 1, 4, 2)

up = st.file_uploader("Upload CSV", type=["csv"])

if up:
    df_in = pd.read_csv(up)
    df_in.columns = [c.lower().strip() for c in df_in.columns]
    
    if "stock symbol" not in df_in.columns:
        st.error("Missing 'stock symbol' column.")
    else:
        ticks = sorted(list(set([_clean_nse_symbol(x) for x in df_in["stock symbol"] if x])))
        if st.button("🚀 RUN AUDIT"):
            results = []
            prog = st.progress(0)
            status = st.empty()
            
            # 1. Batch fetch prices (More efficient)
            prices = yf.download(tickers=ticks, period="2y", group_by="ticker", progress=False)

            # 2. Sequential/Parallel Fundamentals with Thread Management
            with ThreadPoolExecutor(max_workers=work) as exe:
                futures = {exe.submit(fetch_fundamentals_one, t): t for t in ticks}
                for i, fut in enumerate(as_completed(futures)):
                    t = futures[fut]
                    status.text(f"Processing {i+1}/{len(ticks)}: {t}")
                    try:
                        f = fut.result()
                        ltp, d50, d150, d200 = (None,)*4
                        try:
                            s = prices[t]['Close'].dropna() if len(ticks)>1 else prices['Close'].dropna()
                            if len(s) > 200:
                                ltp, d50, d150, d200 = s.iloc[-1], s.rolling(50).mean().iloc[-1], s.rolling(150).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
                        except: pass
                        
                        pb, fair, mos_p, meth = run_valuation(f, ltp, mos_in)
                        mom = "Bullish" if (ltp and d50 and d200 and ltp > d50 > d200) else ("Bearish" if (ltp and d200 and ltp < d200) else "Neutral")
                        rec, reas = ("Buy", "Undervalued") if (ltp and mos_p and ltp <= mos_p) else (("Sell", "Overvalued") if (ltp and fair and ltp > fair * 1.4) else ("Hold", "Fair Value"))

                        results.append({
                            C_COMP: t.split(".")[0], C_LTP: ltp, C_50: d50, C_150: d150, C_200: d200,
                            C_SALES: (f["s_3y"]*100 if f["s_3y"] else None), C_PROFIT: (f["p_3y"]*100 if f["p_3y"] else None),
                            C_ROE: (f["roe"]*100 if f["roe"] else None), C_ROCE: (f["roce"]*100 if f["roce"] else None),
                            C_PB: pb, C_FAIR: fair, C_MOS: mos_p, C_METH: meth, C_MOM: mom, C_REC: rec, C_REAS: reas
                        })
                    except Exception:
                        pass 
                    prog.progress((i+1)/len(ticks))
            
            status.empty()
            if results:
                res_df = pd.DataFrame(results)
                nums = res_df.select_dtypes(include=[np.number]).columns
                res_df[nums] = res_df[nums].round(2)
                res_df = res_df.fillna("NA")

                flat_cols = [f"{g}: {c}".strip(": ") if g else c for g, c in DISPLAY_MAP]
                disp_df = res_df[[c for g, c in DISPLAY_MAP]].copy()
                disp_df.columns = flat_cols

                def style_fn(s):
                    s.applymap(lambda v: "background-color: #dcfce7; color: #166534; font-weight: bold;" if v=="Buy" else ("background-color: #fee2e2; color: #991b1b; font-weight: bold;" if v=="Sell" else ""), subset=[f"{G_DEC}: {C_REC}"])
                    s.set_properties(**{'background-color': '#eff6ff', 'font-weight': 'bold'}, subset=[f"{G_VAL}: {C_MOS}"])
                    return s

                st.dataframe(style_fn(disp_df.style).format(precision=2), use_container_width=True, hide_index=True)
                st.download_button("📥 Download", res_df.to_csv(index=False), "Audit.csv")
            else:
                st.error("No data could be retrieved. Please wait 5 minutes and try again (Yahoo Rate Limit).")
