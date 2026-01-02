---

## 3) `app.py`
```python
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# UI + CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Wealth Architect Pro (NSE)")
st.title("🏛️ Wealth Architect Pro")
st.caption("NSE-only portfolio audit using Yahoo Finance (free). Includes technicals, fundamentals (best-effort), and sector-aware valuation bands.")

# IMPORTANT: You asked for exact column names + order
OUTPUT_COLUMNS = [
    "Company",
    "LTP",
    "50DMA",
    "150DMA",
    "200DMA",
    "Sales growth % (YOY-3 years)",
    "Profit growth % (YOY 3years)",
    "ROE",
    "ROCE",
    "VAL: PB",
    "VAL: Fair",
    "VAL: MoS Buy",
    "VAL: Method",
    "Momentum",
    "Recommendation",
    "Reason",
]

# Sector-aware valuation bands (India-leaning heuristics)
# You can fine-tune these later once you see results.
SECTOR_BANDS = {
    # Banks/NBFCs typically better handled by P/B
    "Financial Services": {"method": "PB", "min": 1.0, "max": 3.0},
    # IT/services can carry higher P/E
    "Technology": {"method": "PE", "min": 18, "max": 40},
    "Communication Services": {"method": "PE", "min": 15, "max": 35},
    # Staples / FMCG often higher
    "Consumer Defensive": {"method": "PE", "min": 25, "max": 55},
    # Cyclical consumer (auto, discretionary): keep tighter than 30 blanket
    "Consumer Cyclical": {"method": "PE", "min": 10, "max": 22},
    # Cyclicals/materials: use normalized PE, lower band
    "Basic Materials": {"method": "CYCLICAL_PE", "min": 6, "max": 14},
    "Energy": {"method": "CYCLICAL_PE", "min": 6, "max": 14},
    "Industrials": {"method": "PE", "min": 12, "max": 25},
    "Utilities": {"method": "PE", "min": 10, "max": 18},
    "Healthcare": {"method": "PE", "min": 18, "max": 35},
    "Real Estate": {"method": "CYCLICAL_PE", "min": 8, "max": 18},
    "Default": {"method": "PE", "min": 12, "max": 22},
}

# =========================
# HELPERS
# =========================
def _as_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, float, int)):
            return float(x)
        return float(str(x))
    except Exception:
        return None

def _clean_nse_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if not s:
        return ""
    # NSE-only:
    if "." not in s:
        s += ".NS"
    # If someone gives .BO etc, keep it but this app is meant for NSE
    return s

def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b

def _cagr(v0, v1, years: float):
    if v0 is None or v1 is None or v0 <= 0 or v1 <= 0 or years <= 0:
        return None
    return (v1 / v0) ** (1.0 / years) - 1.0

def _pick_ticker_col(df: pd.DataFrame) -> str | None:
    cols = [c.lower().strip() for c in df.columns]
    for key in ["stock symbol", "ticker", "symbol"]:
        if key in cols:
            return df.columns[cols.index(key)]
    # fallback: any column containing symbol/ticker
    for c in df.columns:
        cl = c.lower()
        if "symbol" in cl or "ticker" in cl:
            return c
    return None

def _get_company_name(row, fallback):
    for k in ["company", "company name", "name"]:
        if k in row.index:
            v = row.get(k)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
    return fallback

# =========================
# DATA FETCH (CACHED)
# =========================
@st.cache_data(ttl=60 * 30)  # 30 mins
def fetch_yahoo_bundle(symbol: str):
    """
    Returns a dict with:
    - info (dict)
    - hist (DataFrame)
    - financials (annual income statement)
    - balance_sheet (annual balance sheet)
    """
    t = yf.Ticker(symbol)

    # history: we need >= 200 trading days for 200DMA; keep 2y buffer.
    hist = t.history(period="2y", auto_adjust=False)

    # info: can be slow, but required for sector/roe/pb/eps
    info = t.info or {}

    # annual statements (best-effort)
    # yfinance provides annual financials/balance_sheet in many cases, not always for NSE.
    financials = getattr(t, "financials", None)
    balance_sheet = getattr(t, "balance_sheet", None)

    return {
        "info": info,
        "hist": hist,
        "financials": financials,
        "balance_sheet": balance_sheet,
    }

# =========================
# METRIC COMPUTATION
# =========================
def compute_dmas(hist: pd.DataFrame):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None, None, None, None

    close = hist["Close"].dropna()
    if len(close) < 210:
        return None, None, None, None

    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])
    d150 = float(close.rolling(150).mean().iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])
    return ltp, d50, d150, d200

def compute_3y_growth_from_financials(financials: pd.DataFrame):
    """
    Returns (sales_cagr_3y_pct, profit_cagr_3y_pct)
    - Uses Total Revenue and Net Income if available
    - Needs at least 4 annual points to compute 3-year CAGR cleanly.
    """
    if financials is None or not isinstance(financials, pd.DataFrame) or financials.empty:
        return None, None

    # financials are typically rows x columns (years)
    cols = list(financials.columns)
    if len(cols) < 4:
        # not enough years for 3y CAGR
        return None, None

    # Ensure chronological order (oldest -> newest)
    cols_sorted = sorted(cols)
    fin = financials[cols_sorted]

    def _get_row_any(names):
        for n in names:
            if n in fin.index:
                return fin.loc[n].astype(float).values
        return None

    rev = _get_row_any(["Total Revenue", "TotalRevenue", "totalRevenue"])
    pat = _get_row_any(["Net Income", "NetIncome", "netIncome"])

    # pick 3-year span: oldest among last 4 vs newest
    # last 4 points => years diff ~3
    def _cagr_last4(arr):
        if arr is None or len(arr) < 4:
            return None
        v0 = _as_float(arr[-4])
        v1 = _as_float(arr[-1])
        g = _cagr(v0, v1, 3.0)
        return None if g is None else g * 100

    sales = _cagr_last4(rev)
    profit = _cagr_last4(pat)
    return sales, profit

def compute_roce(financials: pd.DataFrame, balance_sheet: pd.DataFrame):
    """
    Approx ROCE (%) = EBIT / (Total Assets - Current Liabilities) * 100
    Best-effort; returns None if data lines missing.
    """
    if financials is None or balance_sheet is None:
        return None
    if not isinstance(financials, pd.DataFrame) or not isinstance(balance_sheet, pd.DataFrame):
        return None
    if financials.empty or balance_sheet.empty:
        return None

    # Align to latest common column (year)
    fin_cols = sorted(list(financials.columns))
    bs_cols = sorted(list(balance_sheet.columns))
    if not fin_cols or not bs_cols:
        return None

    latest = min(fin_cols[-1], bs_cols[-1])  # safest intersection-ish
    if latest not in financials.columns or latest not in balance_sheet.columns:
        return None

    fin = financials[latest]
    bs = balance_sheet[latest]

    ebit = None
    for k in ["EBIT", "Ebit", "Operating Income", "OperatingIncome"]:
        if k in fin.index:
            ebit = _as_float(fin.loc[k])
            break

    total_assets = None
    for k in ["Total Assets", "TotalAssets"]:
        if k in bs.index:
            total_assets = _as_float(bs.loc[k])
            break

    current_liab = None
    for k in ["Total Current Liabilities", "TotalCurrentLiabilities"]:
        if k in bs.index:
            current_liab = _as_float(bs.loc[k])
            break

    capital_employed = None
    if total_assets is not None and current_liab is not None:
        capital_employed = total_assets - current_liab

    roce = _safe_div(ebit, capital_employed)
    return None if roce is None else roce * 100

def sector_cfg_from_info(info: dict):
    sector = info.get("sector") or "Default"
    cfg = SECTOR_BANDS.get(sector, SECTOR_BANDS["Default"])
    return sector, cfg

def compute_valuation(info: dict, sector: str, cfg: dict, sales_g_3y, profit_g_3y, roe_pct, ltp, mos_pct):
    """
    Returns:
    - pb
    - fair
    - mos_buy
    - method_str
    """
    pb = _as_float(info.get("priceToBook"))
    book_value = _as_float(info.get("bookValue"))

    trailing_eps = _as_float(info.get("trailingEps"))
    forward_eps = _as_float(info.get("forwardEps"))
    eps = forward_eps if (forward_eps is not None and forward_eps > 0) else trailing_eps

    method = cfg["method"]
    fair = None
    method_str = ""

    # Small “quality tilt” for multiple within band:
    # use ROE + growth (3y if available) but keep bounded.
    growth_used = None
    if profit_g_3y is not None and np.isfinite(profit_g_3y):
        growth_used = profit_g_3y
    elif sales_g_3y is not None and np.isfinite(sales_g_3y):
        growth_used = sales_g_3y

    base_mult = (cfg["min"] + cfg["max"]) / 2

    # Adjust multiple modestly (avoid huge skews)
    adj = 1.0
    if roe_pct is not None:
        if roe_pct >= 18:
            adj *= 1.10
        elif roe_pct <= 10:
            adj *= 0.90

    if growth_used is not None:
        if growth_used >= 20:
            adj *= 1.10
        elif growth_used <= 8:
            adj *= 0.90

    mult = base_mult * adj
    mult = max(cfg["min"], min(cfg["max"], mult))

    if method == "PB":
        # For PB, need book value
        if book_value is not None and book_value > 0:
            fair = book_value * mult
            method_str = f"P/B (band {cfg['min']}-{cfg['max']}, used {mult:.2f})"
        else:
            fair = None
            method_str = "P/B (book value missing)"
    elif method == "CYCLICAL_PE":
        # For cyclicals: normalize EPS with haircut
        if eps is not None and eps > 0:
            norm_eps = eps * 0.8
            fair = norm_eps * mult
            method_str = f"CYCLICAL_PE (0.8×EPS, band {cfg['min']}-{cfg['max']}, used {mult:.2f})"
        else:
            fair = None
            method_str = "CYCLICAL_PE (EPS missing)"
    else:
        # Standard PE
        if eps is not None and eps > 0:
            fair = eps * mult
            method_str = f"P/E (band {cfg['min']}-{cfg['max']}, used {mult:.2f})"
        else:
            fair = None
            method_str = "P/E (EPS missing)"

    mos_buy = None
    if fair is not None:
        mos_buy = fair * (1 - mos_pct / 100.0)

    # If pb missing, try compute from book value
    if pb is None and book_value is not None and book_value > 0 and ltp is not None:
        pb = ltp / book_value

    return pb, fair, mos_buy, method_str

def compute_momentum(ltp, d50, d200):
    if ltp is None or d50 is None or d200 is None:
        return "Neutral"
    if ltp > d50 > d200:
        return "Bullish"
    if (ltp < d200) and (d50 < d200):
        return "Bearish"
    return "Neutral"

def final_reco(momentum, ltp, fair, mos_buy):
    """
    Keep it simple for business usage:
    - Buy if below MoS and not bearish
    - Hold if under fair (or near)
    - Sell/Avoid if far above fair or bearish + above fair
    """
    if ltp is None:
        return "Hold", "Price missing"

    if fair is None or mos_buy is None:
        return "Hold", "Insufficient valuation data (EPS/book value missing)"

    if (ltp <= mos_buy) and (momentum != "Bearish"):
        return "Buy", "Price below MoS buy + momentum not bearish"
    if ltp <= fair:
        return "Hold", "Near/under fair value; not at MoS yet"
    if momentum == "Bearish" and ltp > fair:
        return "Sell", "Bearish trend + above fair value"
    return "Hold", "Above fair value (wait for better entry)"

# =========================
# UI
# =========================
with st.sidebar:
    st.header("Settings")
    mos = st.slider("Margin of Safety %", 5, 40, 20)
    st.caption("Note: Sales/Profit growth are computed as 3-year CAGR if annual statements exist on Yahoo; otherwise left blank.")

uploaded = st.file_uploader("Upload Portfolio CSV (NSE)", type=["csv"])

if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

tick_col = _pick_ticker_col(df)
if not tick_col:
    st.error("Your CSV must contain a column named like: stock symbol / ticker / symbol")
    st.stop()

run = st.button("🚀 RUN ANALYSIS")
if not run:
    st.stop()

tickers = df[tick_col].dropna().astype(str).tolist()
tickers = [_clean_nse_symbol(t) for t in tickers if str(t).strip()]

if not tickers:
    st.error("No tickers found in your file.")
    st.stop()

status = st.empty()
progress = st.progress(0)

results = []

# Small parallelism to reduce "stuck" feeling; keep low to avoid Yahoo throttling
MAX_WORKERS = 4

def process_one(idx, raw_sym):
    sym = _clean_nse_symbol(raw_sym)
    bundle = fetch_yahoo_bundle(sym)
    info = bundle["info"] or {}
    hist = bundle["hist"]
    financials = bundle["financials"]
    balance_sheet = bundle["balance_sheet"]

    ltp, d50, d150, d200 = compute_dmas(hist)

    sector, cfg = sector_cfg_from_info(info)

    # Growth (3y CAGR) from annual financials (best effort)
    sales_g3, profit_g3 = compute_3y_growth_from_financials(financials)

    # ROE (Yahoo is decimal)
    roe = info.get("returnOnEquity")
    roe_pct = None if roe is None else float(roe) * 100

    roce_pct = compute_roce(financials, balance_sheet)

    pb, fair, mos_buy, method_str = compute_valuation(
        info=info,
        sector=sector,
        cfg=cfg,
        sales_g_3y=sales_g3,
        profit_g_3y=profit_g3,
        roe_pct=roe_pct,
        ltp=ltp,
        mos_pct=mos,
    )

    momentum = compute_momentum(ltp, d50, d200)
    reco, reason = final_reco(momentum, ltp, fair, mos_buy)

    # Company name (Yahoo longName sometimes missing for NSE)
    company = info.get("longName") or info.get("shortName") or sym.replace(".NS", "")
    return {
        "Company": company,
        "LTP": None if ltp is None else round(ltp, 2),
        "50DMA": None if d50 is None else round(d50, 2),
        "150DMA": None if d150 is None else round(d150, 2),
        "200DMA": None if d200 is None else round(d200, 2),
        "Sales growth % (YOY-3 years)": None if sales_g3 is None else round(sales_g3, 2),
        "Profit growth % (YOY 3years)": None if profit_g3 is None else round(profit_g3, 2),
        "ROE": None if roe_pct is None else round(roe_pct, 2),
        "ROCE": None if roce_pct is None else round(roce_pct, 2),
        "VAL: PB": None if pb is None else round(pb, 2),
        "VAL: Fair": None if fair is None else round(fair, 2),
        "VAL: MoS Buy": None if mos_buy is None else round(mos_buy, 2),
        "VAL: Method": method_str,
        "Momentum": momentum,
        "Recommendation": reco,
        "Reason": reason + (
            "" if (sales_g3 is not None or profit_g3 is not None) else " | Growth(3Y) not available on Yahoo"
        ) + (
            "" if roce_pct is not None else " | ROCE not available from statements"
        ),
    }

# Run
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = []
    for i, sym in enumerate(tickers):
        futures.append(ex.submit(process_one, i, sym))

    done = 0
    total = len(futures)
    for fut in as_completed(futures):
        done += 1
        progress.progress(done / total)
        status.info(f"Processing… {done}/{total}")
        try:
            results.append(fut.result())
        except Exception as e:
            results.append({
                "Company": "UNKNOWN",
                "LTP": None, "50DMA": None, "150DMA": None, "200DMA": None,
                "Sales growth % (YOY-3 years)": None,
                "Profit growth % (YOY 3years)": None,
                "ROE": None, "ROCE": None,
                "VAL: PB": None, "VAL: Fair": None, "VAL: MoS Buy": None,
                "VAL: Method": "Error",
                "Momentum": "Neutral",
                "Recommendation": "Hold",
                "Reason": f"Error fetching data: {e}",
            })

status.empty()

# Order columns EXACTLY as you requested
res_df = pd.DataFrame(results)
for c in OUTPUT_COLUMNS:
    if c not in res_df.columns:
        res_df[c] = None
res_df = res_df[OUTPUT_COLUMNS]

# --------- Render with grouped headers like your screenshot (HTML table) ----------
# Streamlit doesn't support merged header cells in st.dataframe, so we render HTML from pandas Styler.
top = [
    "", "Technicals", "Technicals", "Technicals",
    "Fundamentals", "Fundamentals", "Fundamentals", "Fundamentals", "Fundamentals",
    "Valuation", "Valuation", "Valuation", "Valuation",
    "Final", "Final", "Final"
]
bottom = OUTPUT_COLUMNS

multi_cols = pd.MultiIndex.from_arrays([top, bottom])
display_df = res_df.copy()
display_df.columns = multi_cols

st.subheader("Results")
styler = display_df.style.set_table_styles([
    {"selector": "th", "props": [("background-color", "#e6f0ff"), ("color", "#000"), ("border", "1px solid #ddd"), ("text-align", "center")]},
    {"selector": "td", "props": [("border", "1px solid #eee"), ("padding", "6px 8px")]},
]).set_properties(**{"text-align": "left"})

st.markdown(styler.to_html(), unsafe_allow_html=True)

# Download
csv_bytes = res_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="wealth_architect_results.csv", mime="text/csv")
