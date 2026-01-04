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
    .small-note { color:#6b7280; font-size: 0.85rem; margin-top: -10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Wealth Architect Pro")
st.caption("NSE Portfolio Audit | Data: Yahoo Finance | Version: 2.0 Stable")

# Column Definitions
COL_COMPANY, COL_LTP = "Company", "LTP"
COL_50, COL_150, COL_200 = "50DMA", "150DMA", "200DMA"
COL_SALES_3Y, COL_PROFIT_3Y = "Sales growth % (YOY-3 years)", "Profit growth % (YOY 3years)"
COL_ROE, COL_ROCE = "ROE", "ROCE"
COL_VAL_PB, COL_VAL_FAIR, COL_VAL_MOS, COL_VAL_METHOD = "VAL: PB", "VAL: Fair", "VAL: MoS Buy", "VAL: Method"
COL_MOM, COL_REC, COL_REASON = "Momentum", "Recommendation", "Reason"

GROUP_TECH, GROUP_FUND, GROUP_VAL, GROUP_FINAL, GROUP_LEFT = "Technicals", "Fundamentals", "Valuation", "Final", ""

DISPLAY_COLUMNS = pd.MultiIndex.from_tuples([
    (GROUP_LEFT, COL_COMPANY), (GROUP_LEFT, COL_LTP),
    (GROUP_TECH, COL_50), (GROUP_TECH, COL_150), (GROUP_TECH, COL_200),
    (GROUP_FUND, COL_SALES_3Y), (GROUP_FUND, COL_PROFIT_3Y), (GROUP_FUND, COL_ROE), (GROUP_FUND, COL_ROCE),
    (GROUP_VAL, COL_VAL_PB), (GROUP_VAL, COL_VAL_FAIR), (GROUP_VAL, COL_VAL_MOS), (GROUP_VAL, COL_VAL_METHOD),
    (GROUP_FINAL, COL_MOM), (GROUP_FINAL, COL_REC), (GROUP_FINAL, COL_REASON)
])

CSV_COLUMNS = [COL_COMPANY, COL_LTP, COL_50, COL_150, COL_200, COL_SALES_3Y, COL_PROFIT_3Y, 
               COL_ROE, COL_ROCE, COL_VAL_PB, COL_VAL_FAIR, COL_VAL_MOS, COL_VAL_METHOD, 
               COL_MOM, COL_REC, COL_REASON]

SECTOR_BANDS = {
    "Financial": {"method": "P/B", "pb_min": 1.0, "pb_max": 3.5, "pb_mid": 2.0},
    "IT": {"method": "P/E", "pe_min": 18.0, "pe_max": 40.0, "pe_mid": 26.0},
    "Consumer": {"method": "P/E", "pe_min": 18.0, "pe_max": 40.0, "pe_mid": 24.0},
    "Auto": {"method": "P/E", "pe_min": 10.0, "pe_max": 22.0, "pe_mid": 16.0},
    "Industrials": {"method": "P/E", "pe_min": 12.0, "pe_max": 28.0, "pe_mid": 18.0},
    "Healthcare": {"method": "P/E", "pe_min": 18.0, "pe_max": 38.0, "pe_mid": 24.0},
    "Energy": {"method": "CYCLICAL", "pe_min": 6.0, "pe_max": 14.0, "pe_mid": 10.0},
    "Metals": {"method": "CYCLICAL", "pe_min": 5.0, "pe_max": 12.0, "pe_mid": 8.0},
    "Default": {"method": "P/E", "pe_min": 12.0, "pe_max": 28.0, "pe_mid": 18.0},
}

# =========================
# HELPERS
# =========================
def _clean_nse_symbol(sym: str) -> str:
    sym = str(sym or "").strip().upper()
    if not sym: return ""
    if "." not in sym: sym += ".NS"
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
    if any(k in text for k in ["bank", "nbfc", "insurance", "financial", "capital markets"]): return "Financial"
    if any(k in text for k in ["software", "it services", "technology"]): return "IT"
    if any(k in text for k in ["fmcg", "retail", "consumer", "food"]): return "Consumer"
    if any(k in text for k in ["auto", "tyre", "automobile"]): return "Auto"
    if any(k in text for k in ["industrial", "engineering", "defense", "construction"]): return "Industrials"
    if any(k in text for k in ["pharma", "hospital", "healthcare"]): return "Healthcare"
    if any(k in text for k in ["oil", "gas", "power", "energy", "utilities"]): return "Energy"
    if any(k in text for k in ["metal", "steel", "mining"]): return "Metals"
    return "Default"

def compute_dmas(close: pd.Series):
    if close is None or len(close) < 200: return None, None, None, None
    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])
    d150 = float(close.rolling(150).mean().iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])
    return ltp, d50, d150, d200

def momentum_label(ltp, d50, d150, d200):
    if any(v is None for v in [ltp, d50, d200]): return "NA"
    if ltp > d50 > (d150 or 0) > d200: return "Bullish"
    if ltp < d200 and d50 < d200: return "Bearish"
    return "Neutral"

# =========================
# DATA FETCHING
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_batch(tickers: list):
    try:
        data = yf.download(tickers=tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        return data
    except: return None

def _extract_close_series(batch_df, ticker):
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            if ticker in batch_df.columns.levels[0]:
                return batch_df[ticker]['Close'].dropna()
        else:
            if 'Close' in batch_df.columns:
                return batch_df['Close'].dropna()
        return None
    except: return None

def fetch_fundamentals_one(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info or {}
    
    # Fundamental Metrics
    res = {
        "sector": info.get("sector"), "industry": info.get("industry"),
        "bucket": _sector_bucket(info.get("sector"), info.get("industry")),
        "roe": (_safe_float(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") else None),
        "pb": _safe_float(info.get("priceToBook")),
        "eps": _safe_float(info.get("forwardEps")) or _safe_float(info.get("trailingEps")),
        "book_value": _safe_float(info.get("bookValue")),
        "sales_cagr": None, "profit_cagr": None, "roce": None
    }

    # Best-effort Financial Statements
    try:
        inc = t.income_stmt
        if inc is not None and not inc.empty:
            rev = inc.loc["Total Revenue"].dropna() if "Total Revenue" in inc.index else None
            if rev is not None and len(rev) >= 4: res["sales_cagr"] = _cagr(rev.iloc[-1], rev.iloc[0], 3.0) * 100
            
            ni = inc.loc["Net Income"].dropna() if "Net Income" in inc.index else None
            if ni is not None and len(ni) >= 4: res["profit_cagr"] = _cagr(ni.iloc[-1], ni.iloc[0], 3.0) * 100
    except: pass
    return res

def compute_valuation(fund, mos_pct):
    bucket = fund.get("bucket", "Default")
    cfg = SECTOR_BANDS.get(bucket, SECTOR_BANDS["Default"])
    fair, method = None, f"{cfg['method']} | {bucket}"
    
    if cfg["method"] == "P/B" and fund["book_value"]:
        fair = fund["book_value"] * cfg["pb_mid"]
    elif fund["eps"]:
        pe = cfg["pe_mid"]
        if fund["roe"] and fund["roe"] > 18: pe *= 1.1
        fair = fund["eps"] * pe
    
    mos_buy = fair * (1 - mos_pct/100) if fair else None
    return fund.get("pb"), fair, mos_buy, method

# =========================
# MAIN APP LOGIC
# =========================
with st.sidebar:
    st.header("Settings")
    mos = st.slider("Margin of Safety %", 5, 40, 20)
    workers = st.slider("Speed (Workers)", 1, 8, 4)
    st.markdown('<div class="small-note">Lower workers if Yahoo blocks you.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    
    if "stock symbol" not in df.columns:
        st.error("Missing 'stock symbol' column.")
        st.stop()

    unique_tickers = sorted(list(set([_clean_nse_symbol(x) for x in df["stock symbol"] if x])))
    company_map = dict(zip(df["stock symbol"].apply(_clean_nse_symbol), df.get("company name", df["stock symbol"])))

    if st.button("🚀 RUN ANALYSIS"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.info("Fetching price history...")
        batch_data = fetch_prices_batch(unique_tickers)
        
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_t = {executor.submit(fetch_fundamentals_one, t): t for t in unique_tickers}
            
            for i, future in enumerate(as_completed(future_to_t)):
                t = future_to_t[future]
                try:
                    fund = future.result()
                    close = _extract_close_series(batch_data, t)
                    ltp, d50, d150, d200 = compute_dmas(close)
                    mom = momentum_label(ltp, d50, d150, d200)
                    pb, fair, mos_buy, v_method = compute_valuation(fund, mos)
                    
                    reco, reason = "Hold", "Fairly valued"
                    if ltp and mos_buy and ltp <= mos_buy: reco, reason = "Buy", "Deep Discount"
                    elif ltp and fair and ltp > fair * 1.3: reco, reason = "Sell", "Significant Overvalue"

                    results.append({
                        COL_COMPANY: company_map.get(t, t), COL_LTP: ltp,
                        COL_50: d50, COL_150: d150, COL_200: d200,
                        COL_SALES_3Y: fund["sales_cagr"], COL_PROFIT_3Y: fund["profit_cagr"],
                        COL_ROE: fund["roe"], COL_ROCE: fund["roce"],
                        COL_VAL_PB: pb, COL_VAL_FAIR: fair, COL_VAL_MOS: mos_buy, COL_VAL_METHOD: v_method,
                        COL_MOM: mom, COL_REC: reco, COL_REASON: reason
                    })
                except Exception as e:
                    results.append({COL_COMPANY: t, COL_REC: "Error", COL_REASON: str(e)})
                
                progress_bar.progress((i + 1) / len(unique_tickers))
                status_text.text(f"Analyzed {i+1}/{len(unique_tickers)}: {t}")

        # =========================
        # 3. DISPLAY & STYLING
        # =========================
        res_df = pd.DataFrame(results)

        # A. ROUNDING: Apply 2 decimal rounding to all numeric columns
        num_cols = res_df.select_dtypes(include=[np.number]).columns
        res_df[num_cols] = res_df[num_cols].round(2)
        res_df = res_df.fillna("NA")

        # B. MULTI-INDEX PREP
        display_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
        for (grp, col) in DISPLAY_COLUMNS: 
            display_df[(grp, col)] = res_df[col]
        
        # C. STYLING FUNCTION
        def apply_audit_styles(styler):
            # Recommendation colors
            def color_reco(val):
                if val == "Buy": return "background-color: #dcfce7; color: #166534; font-weight: bold;"
                if val == "Sell": return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
                return ""
            
            styler.applymap(color_reco, subset=[(GROUP_FINAL, COL_REC)])
            
            # MoS Buy Price Highlight (Blue)
            styler.set_properties(**{'background-color': '#eff6ff', 'font-weight': 'bold'}, 
                                 subset=[(GROUP_VAL, COL_VAL_MOS)])
            return styler

        st.subheader("Analysis Results")
        
        # Render with precision 2 to ensure UI consistency
        st.dataframe(
            apply_audit_styles(display_df.style).format(precision=2), 
            use_container_width=True, 
            hide_index=True
        )

        st.download_button("Download CSV", res_df.to_csv(index=False), "results.csv", "text/csv")
        status_text.success("Done!")
