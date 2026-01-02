# app.py
# Wealth Architect Pro (NSE) — Sector-aware valuation + MoS + Momentum gate
# Added metrics: 50DMA, 150DMA, 200DMA, Sales growth % (YoY-3 years), Profit growth % (YoY-3 years),
# ROCE, ROE, PB

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# -------------------------
# UI SETUP
# -------------------------
st.set_page_config(layout="wide", page_title="Wealth Architect Pro — NSE")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; width: 100%; height: 3em; border-radius: 8px; }
    .smallnote { font-size: 12px; color: #334155; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏛️ Wealth Architect Pro — NSE")
st.caption("Sector-aware valuation + Margin of Safety + 50/150/200 DMA + fundamentals (best-effort from yfinance)")

# -------------------------
# SECTOR METHOD CONFIG
# -------------------------
METHOD_BY_SECTOR = {
    "Financial Services": {"method": "PB", "pb_range": (1.0, 4.0)},
    "Industrials": {"method": "EV_EBITDA", "ev_ebitda_range": (10.0, 18.0)},
    "Utilities": {"method": "EV_EBITDA", "ev_ebitda_range": (8.0, 14.0)},
    "Basic Materials": {"method": "CYCLICAL_EV_EBITDA", "ev_ebitda_range": (4.0, 8.0)},
    "Energy": {"method": "CYCLICAL_EV_EBITDA", "ev_ebitda_range": (4.0, 8.0)},
    "Technology": {"method": "PE", "pe_range": (18.0, 45.0)},
    "Consumer Defensive": {"method": "PE", "pe_range": (28.0, 65.0)},
    "Consumer Cyclical": {"method": "PE", "pe_range": (18.0, 45.0)},
    "Healthcare": {"method": "PE", "pe_range": (22.0, 50.0)},
    "Communication Services": {"method": "PE", "pe_range": (15.0, 30.0)},
    "Real Estate": {"method": "PE", "pe_range": (12.0, 28.0)},
    "Default": {"method": "PE", "pe_range": (15.0, 30.0)},
}

def infer_sector_from_industry(industry: str) -> str:
    ind = (industry or "").lower()
    if any(k in ind for k in ["bank", "nbfc", "finance", "financial", "insurance", "brok", "asset management"]):
        return "Financial Services"
    if any(k in ind for k in ["software", "it services", "information technology", "technology"]):
        return "Technology"
    if any(k in ind for k in ["pharma", "pharmaceutical", "hospital", "health", "diagnostic"]):
        return "Healthcare"
    if any(k in ind for k in ["power", "utility", "electric", "water utility", "gas utility"]):
        return "Utilities"
    if any(k in ind for k in ["oil", "gas", "refining", "energy", "exploration"]):
        return "Energy"
    if any(k in ind for k in ["steel", "metal", "mining", "paper", "chemical", "materials", "cement"]):
        return "Basic Materials"
    if any(k in ind for k in ["telecom", "media", "entertainment", "communication"]):
        return "Communication Services"
    if any(k in ind for k in ["real estate", "reit", "property"]):
        return "Real Estate"
    if any(k in ind for k in ["fmcg", "beverage", "foods", "consumer", "retail", "apparel"]):
        return "Consumer Defensive"
    if any(k in ind for k in ["auto", "automobile", "hotel", "travel", "leisure"]):
        return "Consumer Cyclical"
    if any(k in ind for k in ["industrial", "engineering", "capital goods", "construction", "infrastructure", "logistics"]):
        return "Industrials"
    return "Default"

def normalize_symbol(sym: str, default_suffix: str = ".NS") -> str:
    s = str(sym).strip().upper()
    if s == "" or s == "NAN":
        return ""
    if s.endswith((".NS", ".BO")):
        return s
    if "." in s:
        return s
    return s + default_suffix

def r2(x, nd=2):
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
        return round(float(x), nd)
    except Exception:
        return None

def pct(x, nd=1):
    if x is None:
        return None
    try:
        return round(float(x) * 100.0, nd)
    except Exception:
        return None

def cagr_3y(end_val, start_val):
    # 3-year CAGR from start to end: (end/start)^(1/3) - 1
    try:
        end_val = float(end_val)
        start_val = float(start_val)
        if end_val <= 0 or start_val <= 0:
            return None
        return (end_val / start_val) ** (1.0 / 3.0) - 1.0
    except Exception:
        return None

def find_row(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    idx = [str(i).strip().lower() for i in df.index]
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(idx):
            if c == name:
                return df.iloc[i]
    # fuzzy contains
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(idx):
            if c in name:
                return df.iloc[i]
    return None

def get_financial_frames(ticker: yf.Ticker):
    # yfinance provides these depending on version/coverage.
    # We'll try multiple attributes safely.
    income = None
    balance = None
    try:
        income = getattr(ticker, "income_stmt", None)
        if income is None or income.empty:
            income = getattr(ticker, "financials", None)
    except Exception:
        income = None

    try:
        balance = getattr(ticker, "balance_sheet", None)
    except Exception:
        balance = None

    return income, balance

@st.cache_data(ttl=1800)
def fetch_yf(symbol: str):
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    hist = t.history(period="5y", auto_adjust=False)  # enough for 200DMA + stability
    income, balance = get_financial_frames(t)
    return info, hist, income, balance

# -------------------------
# CORE ANALYSIS ENGINE
# -------------------------
def analyze_one(symbol: str, mos_pct: float, coe: float, g: float,
                ttm_pe_haircut: float, cyclical_norm_haircut: float, debt_gate: float):

    info, hist, income, balance = fetch_yf(symbol)

    if hist is None or hist.empty or "Close" not in hist:
        return None

    close = hist["Close"].dropna()
    if len(close) < 210:
        return {
            "Ticker": symbol,
            "Data Status": "Insufficient price history (<210 trading days)",
            "Recommendation": "NO DATA",
        }

    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])
    d150 = float(close.rolling(150).mean().iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])

    # Momentum regime (death cross/weakness captured here)
    if ltp > d50 > d200:
        momentum = "Bullish"
    elif ltp < d200 or d50 < d200:
        momentum = "Bearish"
    else:
        momentum = "Neutral"

    quote_type = info.get("quoteType")
    if quote_type == "ETF":
        return {
            "Ticker": symbol,
            "Sector": "ETF",
            "Method": "ETF",
            "LTP": r2(ltp),
            "50DMA": r2(d50),
            "150DMA": r2(d150),
            "200DMA": r2(d200),
            "Momentum": momentum,
            "Sales growth % (YoY- 3 years)": None,
            "Profit growth % (YoY 3years)": None,
            "ROCE": None,
            "ROE": None,
            "PB": None,
            "Fair Value": None,
            "MoS Buy": None,
            "Valuation Verdict": "Not Applicable",
            "Recommendation": "ETF (Skip valuation)",
            "Confidence": 95,
            "Data Coverage": "High",
            "Notes": "ETF: NAV-tracking instrument; company financials not applicable."
        }

    # Sector detection
    sector = info.get("sector")
    industry = info.get("industry")
    if not sector:
        sector = infer_sector_from_industry(industry)
    cfg = METHOD_BY_SECTOR.get(sector, METHOD_BY_SECTOR["Default"])
    method = cfg["method"]

    # Core fundamentals from info
    roe = info.get("returnOnEquity")  # fraction
    book = info.get("bookValue")
    pb = info.get("priceToBook")      # P/B ratio

    eps_fwd = info.get("forwardEps")
    eps_ttm = info.get("trailingEps")
    peg = info.get("pegRatio")

    ebitda = info.get("ebitda")
    total_debt = info.get("totalDebt")
    cash = info.get("totalCash")

    revenue_growth = info.get("revenueGrowth")
    op_margin = info.get("operatingMargins")
    fcf = info.get("freeCashflow")
    payout = info.get("payoutRatio")

    # NetDebt/EBITDA
    net_debt_to_ebitda = None
    if ebitda and float(ebitda) > 0 and total_debt is not None and cash is not None:
        try:
            net_debt_to_ebitda = (float(total_debt) - float(cash)) / float(ebitda)
        except Exception:
            net_debt_to_ebitda = None

    # ---- 3-year Sales/Profit growth (best-effort from annual income statement)
    sales_cagr_3y = None
    profit_cagr_3y = None
    ebit_latest = None
    capital_employed = None
    roce = None

    # Income statement: rows vary by ticker/provider; we use fuzzy candidates
    if isinstance(income, pd.DataFrame) and not income.empty:
        # We want annual series: columns are years. Take latest and 3 years ago.
        rev_series = find_row(income, ["total revenue", "totalrevenue", "revenue"])
        ni_series = find_row(income, ["net income", "netincome", "net income common stockholders", "profit after tax"])
        ebit_series = find_row(income, ["ebit", "operating income", "income from operations"])

        def get_latest_and_3y_ago(series):
            if series is None:
                return None, None
            s = series.dropna()
            if len(s) < 4:
                return None, None
            # income stmt columns typically newest first; sort by column name/date if possible
            try:
                s = s.sort_index(axis=0)  # no-op for Series, kept safe
            except Exception:
                pass
            # If index is years/dates as columns, series index are columns already.
            # We'll just take last and 4th from last based on positional order after sorting by index string.
            try:
                idx_sorted = sorted(list(s.index), key=lambda x: str(x))
                s2 = s.loc[idx_sorted]
                latest = s2.iloc[-1]
                three_ago = s2.iloc[-4]
                return latest, three_ago
            except Exception:
                # fallback: assume as-is order is newest->oldest
                latest = s.iloc[0]
                three_ago = s.iloc[3]
                return latest, three_ago

        rev_latest, rev_3ago = get_latest_and_3y_ago(rev_series)
        ni_latest, ni_3ago = get_latest_and_3y_ago(ni_series)

        sales_cagr_3y = cagr_3y(rev_latest, rev_3ago)
        profit_cagr_3y = cagr_3y(ni_latest, ni_3ago)

        if ebit_series is not None:
            s = ebit_series.dropna()
            if len(s) >= 1:
                try:
                    # take the newest value (best effort)
                    ebit_latest = float(s.iloc[0])
                except Exception:
                    ebit_latest = None

    # Balance sheet for ROCE (best-effort)
    if isinstance(balance, pd.DataFrame) and not balance.empty:
        assets_series = find_row(balance, ["total assets"])
        cl_series = find_row(balance, ["total current liabilities", "current liabilities"])
        # capital employed approx = Total Assets - Total Current Liabilities
        if assets_series is not None and cl_series is not None:
            try:
                assets_latest = float(assets_series.dropna().iloc[0])
                cl_latest = float(cl_series.dropna().iloc[0])
                capital_employed = assets_latest - cl_latest
            except Exception:
                capital_employed = None

    # ROCE approx
    if ebit_latest is not None and capital_employed is not None and capital_employed > 0:
        try:
            roce = ebit_latest / capital_employed
        except Exception:
            roce = None

    # ---- Data coverage + confidence
    missing = 0
    for k in [roe, pb, sales_cagr_3y, profit_cagr_3y, roce]:
        if k is None:
            missing += 1
    data_coverage = {0: "High", 1: "Medium", 2: "Low", 3: "Very Low", 4: "Very Low", 5: "Very Low"}.get(missing, "Very Low")

    conf = 90
    conf -= 8 * missing
    if method == "PB" and (book is None or roe is None):
        conf -= 15
    if method in ["EV_EBITDA", "CYCLICAL_EV_EBITDA"] and (ebitda is None):
        conf -= 15
    if method == "PE" and (eps_fwd is None and eps_ttm is None):
        conf -= 15
    conf = max(0, min(100, conf))

    # ---- Hard stops
    hard_stop = False
    reasons = []

    if payout is not None and payout > 1.0:
        hard_stop = True
        reasons.append("Dividend > Earnings (payoutRatio > 1)")

    if net_debt_to_ebitda is not None and net_debt_to_ebitda > debt_gate and sector not in ["Utilities"]:
        hard_stop = True
        reasons.append(f"NetDebt/EBITDA > {debt_gate:g}")

    # ---- Quality score
    q_score = 0
    if roe is not None and roe > 0.15: q_score += 25
    if roce is not None and roce > 0.15: q_score += 25
    if sales_cagr_3y is not None and sales_cagr_3y > 0.12: q_score += 20
    if profit_cagr_3y is not None and profit_cagr_3y > 0.12: q_score += 20
    if fcf is not None and fcf > 0: q_score += 10

    # -------------------------
    # VALUATION
    # -------------------------
    fair_val = None
    basis = None

    if method == "PB":
        # justified PB ~ (ROE - g) / (CoE - g) clamped to pb_range
        if book and float(book) > 0 and roe and float(roe) > 0:
            coe_f = max(0.06, float(coe))
            g_f = min(float(g), coe_f - 0.01)
            denom = (coe_f - g_f)
            justified_pb = (float(roe) - g_f) / denom if denom > 0 else None
            if justified_pb is None or not np.isfinite(justified_pb) or justified_pb <= 0:
                justified_pb = 2.0
            pb_min, pb_max = cfg.get("pb_range", (1.0, 4.0))
            justified_pb = max(pb_min, min(pb_max, justified_pb))
            fair_val = float(book) * float(justified_pb)
            basis = f"P/B (justified PB≈{r2(justified_pb,2)})"

    elif method == "EV_EBITDA":
        if ebitda and float(ebitda) > 0:
            lo, hi = cfg.get("ev_ebitda_range", (10.0, 18.0))
            mid = (lo + hi) / 2.0
            if net_debt_to_ebitda is not None and net_debt_to_ebitda > 3:
                mid *= 0.85
            fair_ev = float(ebitda) * float(mid)
            net_debt = (float(total_debt) - float(cash)) if (total_debt is not None and cash is not None) else 0.0
            fair_equity = fair_ev - float(net_debt)
            market_cap = info.get("marketCap")
            shares = (float(market_cap) / float(ltp)) if (market_cap and ltp > 0) else None
            if shares and shares > 0:
                fair_val = fair_equity / shares
                basis = f"EV/EBITDA (fair≈{r2(mid,2)}x)"

    elif method == "CYCLICAL_EV_EBITDA":
        if ebitda and float(ebitda) > 0:
            lo, hi = cfg.get("ev_ebitda_range", (4.0, 8.0))
            mid = (lo + hi) / 2.0
            norm_ebitda = float(ebitda) * (1.0 - float(cyclical_norm_haircut))
            fair_ev = norm_ebitda * float(mid)
            net_debt = (float(total_debt) - float(cash)) if (total_debt is not None and cash is not None) else 0.0
            fair_equity = fair_ev - float(net_debt)
            market_cap = info.get("marketCap")
            shares = (float(market_cap) / float(ltp)) if (market_cap and ltp > 0) else None
            if shares and shares > 0:
                fair_val = fair_equity / shares
                basis = f"Norm EV/EBITDA (EBITDA -{int(cyclical_norm_haircut*100)}%, fair≈{r2(mid,2)}x)"

    else:  # PE
        pe_min, pe_max = cfg.get("pe_range", (15.0, 30.0))
        base_pe = (pe_min + pe_max) / 2.0
        if peg is not None:
            try:
                if float(peg) < 1.2: base_pe *= 1.10
                elif float(peg) > 2.0: base_pe *= 0.85
            except Exception:
                pass
        base_pe = max(pe_min, min(pe_max, base_pe))

        eps_used = None
        eff_pe = base_pe

        if eps_fwd is not None and eps_fwd > 0:
            eps_used = float(eps_fwd)
            basis = f"P/E (Forward EPS, fairPE≈{r2(eff_pe,2)})"
        elif eps_ttm is not None and eps_ttm > 0:
            eps_used = float(eps_ttm)
            eff_pe = base_pe * float(ttm_pe_haircut)
            basis = f"P/E (TTM EPS, fairPE≈{r2(eff_pe,2)} after haircut)"

        if eps_used is not None and eff_pe > 0:
            fair_val = eps_used * eff_pe

    if fair_val is None or fair_val <= 0:
        return {
            "Ticker": symbol,
            "Sector": sector,
            "Industry": industry,
            "Method": method,
            "LTP": r2(ltp),
            "50DMA": r2(d50),
            "150DMA": r2(d150),
            "200DMA": r2(d200),
            "Momentum": momentum,
            "Sales growth % (YoY- 3 years)": pct(sales_cagr_3y, 1),
            "Profit growth % (YoY 3years)": pct(profit_cagr_3y, 1),
            "ROCE": pct(roce, 1),
            "ROE": pct(roe, 1),
            "PB": r2(pb, 2),
            "Fair Value": None,
            "MoS Buy": None,
            "Valuation Basis": None,
            "Recommendation": "HOLD / WATCH (Valuation NA)",
            "Confidence": conf,
            "Data Coverage": data_coverage,
            "Notes": "Missing valuation inputs for this method (common for some NSE tickers on yfinance)."
        }

    mos_price = fair_val * (1.0 - mos_pct / 100.0)

    # Recommendation rules (no strong buy in bearish trend)
    if hard_stop:
        rec = f"AVOID ({'; '.join(reasons)})"
    else:
        if momentum == "Bearish":
            if ltp <= mos_price:
                rec = "WATCHLIST (Cheap but trend bearish)"
            elif ltp <= fair_val:
                rec = "HOLD / WATCH (Trend bearish)"
            else:
                rec = "AVOID (Expensive + bearish)"
        else:
            if ltp <= mos_price and q_score >= 60:
                rec = "STRONG BUY / ADD"
            elif ltp <= fair_val and q_score >= 45:
                rec = "HOLD / ACCUMULATE ON DIPS"
            elif ltp <= fair_val:
                rec = "HOLD / WATCH"
            else:
                rec = "WAIT / AVOID"

    return {
        "Ticker": symbol,
        "Sector": sector,
        "Industry": industry,
        "Method": method,
        "LTP": r2(ltp),
        "50DMA": r2(d50),
        "150DMA": r2(d150),
        "200DMA": r2(d200),
        "Momentum": momentum,
        "Sales growth % (YoY- 3 years)": pct(sales_cagr_3y, 1),
        "Profit growth % (YoY 3years)": pct(profit_cagr_3y, 1),
        "ROCE": pct(roce, 1),
        "ROE": pct(roe, 1),
        "PB": r2(pb, 2),
        "Fair Value": r2(fair_val, 2),
        "MoS Buy": r2(mos_price, 2),
        "Valuation Basis": basis,
        "Quality Score": q_score,
        "Confidence": conf,
        "Data Coverage": data_coverage,
        "Recommendation": rec,
        "Notes": ""
    }

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("Controls")

    suffix_choice = st.selectbox("Default ticker suffix", [".NS (NSE)", ".BO (BSE)"], index=0)
    default_suffix = ".NS" if suffix_choice.startswith(".NS") else ".BO"

    mos = st.slider("Margin of Safety (MoS) %", 5, 40, 20)

    st.subheader("Financials (P/B) inputs")
    coe = st.slider("Cost of Equity (CoE)", 0.08, 0.22, 0.14, 0.01)
    g = st.slider("Long-term Growth (g)", 0.03, 0.14, 0.08, 0.01)
    if g >= coe:
        st.warning("Keep g < CoE for justified P/B to behave sensibly.")

    st.subheader("P/E controls")
    ttm_haircut = st.slider("TTM EPS haircut on fair P/E", 0.60, 1.00, 0.85, 0.05)

    st.subheader("Cyclical controls")
    cyc_norm = st.slider("Cyclical normalization haircut on EBITDA", 0.10, 0.60, 0.35, 0.05)

    st.subheader("Risk gate")
    debt_gate = st.slider("Hard stop NetDebt/EBITDA (non-utilities)", 2.0, 6.0, 3.0, 0.5)

    st.markdown(
        '<div class="smallnote">'
        "<b>Note:</b> Sales/Profit 3-year growth, ROCE are computed from annual statements when available. "
        "If missing on yfinance, they will be blank and Confidence will drop."
        "</div>",
        unsafe_allow_html=True
    )

# -------------------------
# UPLOAD + RUN
# -------------------------
uploaded = st.file_uploader("Upload your portfolio (CSV or XLSX)", type=["csv", "xlsx"])

if uploaded:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    df.columns = [c.lower().strip() for c in df.columns]

    symbol_candidates = [c for c in df.columns if ("symbol" in c) or ("ticker" in c)]
    if not symbol_candidates:
        st.error("No symbol/ticker column found. Add a column like 'stock symbol' or 'ticker'.")
        st.stop()

    symbol_col = st.selectbox("Which column has the ticker?", symbol_candidates)

    if st.button("🚀 RUN ANALYSIS"):
        results = []
        status = st.empty()
        progress = st.progress(0)

        tickers = df[symbol_col].dropna().astype(str).tolist()
        total = len(tickers)

        for i, raw in enumerate(tickers):
            sym = normalize_symbol(raw, default_suffix)
            if not sym:
                continue

            status.info(f"Analyzing {sym} ({i+1}/{total}) ...")
            out = analyze_one(sym, mos, coe, g, ttm_haircut, cyc_norm, debt_gate)
            if out:
                results.append(out)

            progress.progress((i + 1) / total)

        status.empty()

        if not results:
            st.error("No results generated. Check symbols and try again.")
            st.stop()

        res_df = pd.DataFrame(results)

        # Force a clean column order with your requested metrics prominent
        preferred_order = [
            "Ticker", "Sector", "Industry", "Method",
            "LTP", "50DMA", "150DMA", "200DMA", "Momentum",
            "Sales growth % (YoY- 3 years)", "Profit growth % (YoY 3years)", "ROCE", "ROE", "PB",
            "Fair Value", "MoS Buy", "Valuation Basis",
            "Quality Score", "Confidence", "Data Coverage", "Recommendation", "Notes"
        ]
        ordered = [c for c in preferred_order if c in res_df.columns] + [c for c in res_df.columns if c not in preferred_order]
        res_df = res_df[ordered]

        st.subheader("Results")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results (CSV)",
            data=csv_bytes,
            file_name="wealth_architect_results.csv",
            mime="text/csv",
        )

st.markdown(
    "### How to run (local)\n"
    "1) Install:\n"
    "```bash\n"
    "pip install streamlit yfinance pandas numpy openpyxl\n"
    "```\n"
    "2) Run:\n"
    "```bash\n"
    "streamlit run app.py\n"
    "```\n"
)
