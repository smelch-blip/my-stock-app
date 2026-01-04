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

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; width: 100%; height: 3em; border-radius: 8px; }
    .small-note { color:#6b7280; font-size: 0.85rem; margin-top: -10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Wealth Architect Pro")
st.caption("NSE Portfolio Audit | Sector-Weighted Intelligence | ROCE & MoS Enabled")

# Column Definitions
COL_COMPANY, COL_LTP = "Company", "LTP"
COL_50, COL_150, COL_200 = "50DMA", "150DMA", "200DMA"
COL_SALES_3Y, COL_PROFIT_3Y = "Sales Growth %", "Profit Growth %"
COL_ROE, COL_ROCE = "ROE %", "ROCE %"
COL_VAL_PB, COL_VAL_FAIR, COL_VAL_MOS, COL_VAL_METHOD = "P/B Ratio", "Fair Value", "MoS Buy Price", "Method"
COL_MOM, COL_REC, COL_REASON = "Momentum", "Recommendation", "Reason"

GROUP_TECH, GROUP_FUND, GROUP_VAL, GROUP_FINAL, GROUP_LEFT = "Technicals", "Fundamentals", "Valuation", "Decision", ""

# MultiIndex Setup for display
DISPLAY_COLUMNS = pd.MultiIndex.from_tuples([
    (GROUP_LEFT, COL_COMPANY), (GROUP_LEFT, COL_LTP),
    (GROUP_TECH, COL_50), (GROUP_TECH, COL_150), (GROUP_TECH, COL_200),
    (GROUP_FUND, COL_SALES_3Y), (GROUP_FUND, COL_PROFIT_3Y), (GROUP_FUND, COL_ROE), (GROUP_FUND, COL_ROCE),
    (GROUP_VAL, COL_VAL_PB), (GROUP_VAL, COL_VAL_FAIR), (GROUP_VAL, COL_VAL_MOS), (GROUP_VAL, COL_VAL_METHOD),
    (GROUP_FINAL, COL_MOM), (GROUP_FINAL, COL_REC), (GROUP_FINAL, COL_REASON)
])

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

def _sector_bucket(sector: str, industry: str) -> str:
    text = f"{str(sector)} {str(industry)}".lower()
    if any(k in text for k in ["bank", "nbfc", "insurance", "financial"]): return "Financial"
    if any(k in text for k in ["software", "it services", "technology"]): return "Technology"
    if any(k in text for k in ["fmcg", "retail", "consumer", "food"]): return "Consumer"
    if any(k in text for k in ["auto", "automobile"]): return "Auto"
    return "Default"

# =========================
# DATA FETCHING (CAGR + ROCE)
# =========================
def fetch_fundamentals_one(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info or {}
    res = {
        "sector": info.get("sector"), "industry": info.get("industry"),
        "roe": _safe_float(info.get("returnOnEquity")),
        "pb": _safe_float(info.get("priceToBook")),
        "eps": _safe_float(info.get("forwardEps")) or _safe_float(info.get("trailingEps")),
        "book_value": _safe_float(info.get("bookValue")),
        "sales_3y": None, "profit_3y": None, "roce": None
    }
    try:
        inc = t.income_stmt
        bal = t.balance_sheet
        if inc is not None and not inc.empty:
            if "Total Revenue" in inc.index:
                rev = inc.loc["Total Revenue"].dropna()
                if len(rev) >= 2: res["sales_3y"] = _cagr(rev.iloc[-1], rev.iloc[0], len(rev)-1)
            if "Net Income" in inc.index:
                ni = inc.loc["Net Income"].dropna()
                if len(ni) >= 2: res["profit_3y"] = _cagr(ni.iloc[-1], ni.iloc[0], len(ni)-1)
            
            # ROCE Calculation
            if bal is not None and not bal.empty:
                ebit = inc.loc["EBIT"].iloc[0] if "EBIT" in inc.index else None
                assets = bal.loc["Total Assets"].iloc[0] if "Total Assets" in bal.index else None
                liab = bal.loc["Current Liabilities"].iloc[0] if "Current Liabilities" in bal.index else 0
                if ebit and assets:
                    res["roce"] = (ebit / (assets - liab))
    except: pass
    return res

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_batch(tickers: list):
    try:
        return yf.download(tickers=tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: return None

# =========================
# VALUATION ENGINE
# =========================
def run_valuation_engine(fund, ltp, mos_pct):
    sector = fund.get("sector", "")
    bucket = _sector_bucket(sector, fund.get("industry", ""))
    roce = fund.get("roce", 0)
    
    # 1. DCF Model (3.79 Factor + Terminal)
    dcf_val = None
    if fund["eps"]:
        dcf_val = (fund["eps"] * 3.7908) + (fund["eps"] * 10 / (1.1**5))
        
    # 2. P/B Graham Model
    pb_val = (fund["book_value"] * (fund["roe"]/0.12)) if fund["book_value"] and fund["roe"] else None

    # 3. Industry Specific Weighting
    if bucket == "Financial":
        weights = {"DCF": 0.05, "PB": 0.95}
    elif bucket == "Technology":
        weights = {"DCF": 0.85, "PB": 0.15}
    else:
        weights = {"DCF": 0.50, "PB": 0.50}
    
    # Quality Premium for High ROCE (>22%)
    quality_mult = 1.15 if (roce and roce > 0.22) else 1.0
    
    vals, w_sum = [], 0
    if dcf_val: 
        vals.append(dcf_val * weights["DCF"]); w_sum += weights["DCF"]
    if pb_val: 
        vals.append(pb_val * weights["PB"]); w_sum += weights["PB"]
        
    fair = ((sum(vals)/w_sum) if w_sum > 0 else (ltp or 0)) * quality_mult
    mos_buy = fair * (1 - mos_pct/100) if fair else None
    
    return fund["pb"], fair, mos_buy, f"{bucket} Model"

# =========================
# MAIN APP
# =========================
with st.sidebar:
    st.header("Control Panel")
    mos_input = st.slider("Margin of Safety %", 5, 40, 25)
    workers = st.slider("Threads", 1, 10, 4)
    st.info("Valuation shifts automatically based on industry (e.g., P/B for Banks).")

uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    
    if "stock symbol" not in df.columns:
        st.error("Missing 'Stock Symbol' column.")
    else:
        tickers = sorted(list(set([_clean_nse_symbol(x) for x in df["stock symbol"] if x])))
        
        if st.button("🚀 EXECUTE AUDIT"):
            results = []
            progress = st.progress(0)
            status = st.empty()
            
            batch_data = fetch_prices_batch(tickers)
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_t = {executor.submit(fetch_fundamentals_one, t): t for t in tickers}
                for i, future in enumerate(as_completed(future_to_t)):
                    t = future_to_t[future]
                    try:
                        f = future.result()
                        # Technicals
                        ltp, d50, d150, d200 = (None,)*4
                        try:
                            series = batch_data[t]['Close'].dropna() if len(tickers)>1 else batch_data['Close'].dropna()
                            if len(series) > 200:
                                ltp, d50, d150, d200 = series.iloc[-1], series.rolling(50).mean().iloc[-1], series.rolling(150).mean().iloc[-1], series.rolling(200).mean().iloc[-1]
                        except: pass
                        
                        pb, fair, mos_buy, v_meth = run_valuation_engine(f, ltp, mos_input)
                        
                        mom = "Neutral"
                        if ltp and d50 and d200:
                            if ltp > d50 > d200: mom = "Bullish"
                            elif ltp < d200: mom = "Bearish"
                        
                        reco, reason = "Hold", "Fairly Valued"
                        if ltp and mos_buy and ltp <= mos_buy: reco, reason = "Buy", "Undervalued"
                        elif ltp and fair and ltp > fair * 1.4: reco, reason = "Sell", "Expensive"

                        results.append({
                            COL_COMPANY: t.split(".")[0], COL_LTP: ltp,
                            COL_50: d50, COL_150: d150, COL_200: d200,
                            COL_SALES_3Y: (f["sales_3y"]*100 if f["sales_3y"] else None),
                            COL_PROFIT_3Y: (f["profit_3y"]*100 if f["profit_3y"] else None),
                            COL_ROE: (f["roe"]*100 if f["roe"] else None),
                            COL_ROCE: (f["roce"]*100 if f["roce"] else None),
                            COL_VAL_PB: pb, COL_VAL_FAIR: fair, COL_VAL_MOS: mos_buy, COL_VAL_METHOD: v_meth,
                            COL_MOM: mom, COL_REC: reco, COL_REASON: reason
                        })
                    except: pass
                    progress.progress((i+1)/len(tickers))
            
            # Process & Style
            res_df = pd.DataFrame(results)
            # 1. Process Data & Rounding
            res_df = pd.DataFrame(results)
            num_cols = res_df.select_dtypes(include=[np.number]).columns
            res_df[num_cols] = res_df[num_cols].round(2)
            res_df = res_df.fillna("NA")

            # 2. Flatten MultiIndex for Streamlit Compatibility
            # We combine ('Valuation', 'MoS Buy Price') into 'Valuation: MoS Buy Price'
            flat_cols = [f"{grp}: {col}".strip(": ") if grp else col for grp, col in DISPLAY_COLUMNS]
            disp_df = pd.DataFrame(res_df[CSV_COLUMNS].values, columns=flat_cols)
            
            # 3. Updated Styling Function for Flattened Columns
            def style_audit(styler):
                # Target the flattened column name 'Decision: Recommendation'
                styler.applymap(lambda v: "background-color: #dcfce7; color: #166534; font-weight: bold;" if v=="Buy" else 
                                ("background-color: #fee2e2; color: #991b1b; font-weight: bold;" if v=="Sell" else ""), 
                                subset=[f"{GROUP_FINAL}: {COL_REC}"])
                
                # Target the flattened column name 'Valuation: MoS Buy Price'
                styler.set_properties(**{'background-color': '#eff6ff', 'font-weight': 'bold'}, 
                                     subset=[f"{GROUP_VAL}: {COL_VAL_MOS}"])
                return styler

            st.subheader("📊 Portfolio Report")
            
            # 4. Final Dataframe Render with Flattened Keys
            st.dataframe(
                style_audit(disp_df.style).format(precision=2), 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    f"{GROUP_VAL}: {COL_VAL_MOS}": st.column_config.NumberColumn("MoS Buy Price 🎯", help="Target entry price after Margin of Safety"),
                    f"{GROUP_FUND}: {COL_ROCE}": st.column_config.NumberColumn("ROCE %", help="Capital efficiency indicator")
                }
            )

            st.dataframe(
                style_audit(disp_df.style).format(precision=2), 
                use_container_width=True, hide_index=True,
                column_config={
                    (GROUP_VAL, COL_VAL_MOS): st.column_config.NumberColumn("MoS Buy Price", help="Target entry price after Margin of Safety"),
                    (GROUP_FUND, COL_ROCE): st.column_config.NumberColumn("ROCE %", help="Capital efficiency indicator")
                }
            )
            st.download_button("📥 Download Report", res_df.to_csv(index=False), "WealthAudit.csv", "text/csv")
