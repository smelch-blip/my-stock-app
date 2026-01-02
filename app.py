import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(layout="wide", page_title="Stock Analyst")
st.title("📈 Portfolio Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your portfolio (Excel or CSV)", type=["csv", "xlsx"])

# ---------------- Helpers ---------------- #
def normalize_symbol(sym: str, default_suffix: str = ".NS") -> str:
    sym = str(sym).strip().upper()
    if sym == "" or sym == "NAN":
        return ""
    if "." in sym:  # user already provided .NS / .BO
        return sym
    return sym + default_suffix

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None

def calc_cagr(old, new, years):
    old = safe_float(old); new = safe_float(new)
    if old is None or new is None or old <= 0 or years <= 0:
        return None
    try:
        return (new / old) ** (1 / years) - 1
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_yf_data(symbol: str):
    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # price history for DMAs
    hist = t.history(period="2y", auto_adjust=False)

    # financial statements
    try:
        fin = t.financials  # annual income statement
    except Exception:
        fin = pd.DataFrame()

    try:
        cf = t.cashflow     # annual cashflow
    except Exception:
        cf = pd.DataFrame()

    try:
        bs = t.balance_sheet  # annual balance sheet
    except Exception:
        bs = pd.DataFrame()

    return info, hist, fin, cf, bs

def compute_metrics(symbol: str, fair_pe: float, mos_pct: float):
    # Returns a dict with all columns (tech + fundamentals + valuation + rec)
    info, hist, fin, cf, bs = fetch_yf_data(symbol)

    # ---------------- Technicals ---------------- #
    ltp = d50 = d150 = d200 = None
    tech_rec = "Data Missing"
    if hist is not None and not hist.empty and "Close" in hist.columns and len(hist) >= 210:
        close = hist["Close"].dropna()
        if len(close) >= 210:
            ltp = float(close.iloc[-1])
            d50 = float(close.rolling(50).mean().iloc[-1])
            d150 = float(close.rolling(150).mean().iloc[-1])
            d200 = float(close.rolling(200).mean().iloc[-1])

            # Technical recommendation
            if ltp > d50 > d150 > d200:
                tech_rec = "Strong Buy (Trend)"
            elif ltp < d200:
                tech_rec = "Sell/Caution (Below 200DMA)"
            elif ltp > d200 and ltp < d50:
                tech_rec = "Hold (Pullback)"
            else:
                tech_rec = "Hold"

    # ---------------- Fundamentals (5 core) ---------------- #
    # 1) ROCE (approx) = EBIT / (Total Assets - Current Liabilities)
    roce = None
    try:
        # EBIT sometimes present in income statement
        ebit = None
        if fin is not None and not fin.empty:
            # last column is most recent year
            if "EBIT" in fin.index:
                ebit = safe_float(fin.loc["EBIT"].iloc[0])
            elif "Operating Income" in fin.index:
                ebit = safe_float(fin.loc["Operating Income"].iloc[0])

        capital_employed = None
        if bs is not None and not bs.empty:
            total_assets = safe_float(bs.loc["Total Assets"].iloc[0]) if "Total Assets" in bs.index else None
            current_liab = safe_float(bs.loc["Total Liab"].iloc[0]) if "Total Liab" in bs.index else None
            # Better approx if current liabilities exist
            if "Total Current Liabilities" in bs.index:
                current_liab = safe_float(bs.loc["Total Current Liabilities"].iloc[0])
            if total_assets is not None and current_liab is not None:
                capital_employed = total_assets - current_liab

        if ebit is not None and capital_employed is not None and capital_employed > 0:
            roce = ebit / capital_employed
    except Exception:
        roce = None

    # 2) Sales growth (approx 3Y CAGR) from annual "Total Revenue"
    sales_cagr_3y = None
    # 3) PAT growth (approx 3Y CAGR) from annual "Net Income"
    pat_cagr_3y = None
    try:
        if fin is not None and not fin.empty:
            # yfinance financials columns are years; sometimes 4 columns available
            # We’ll use latest and 3-years-back if available
            if "Total Revenue" in fin.index and fin.shape[1] >= 2:
                rev_series = fin.loc["Total Revenue"]
                # most recent is first column for yfinance (often)
                rev_latest = safe_float(rev_series.iloc[0])
                rev_old = safe_float(rev_series.iloc[min(3, len(rev_series)-1)])
                years = min(3, len(rev_series)-1)
                sales_cagr_3y = calc_cagr(rev_old, rev_latest, years)

            if "Net Income" in fin.index and fin.shape[1] >= 2:
                ni_series = fin.loc["Net Income"]
                ni_latest = safe_float(ni_series.iloc[0])
                ni_old = safe_float(ni_series.iloc[min(3, len(ni_series)-1)])
                years = min(3, len(ni_series)-1)
                pat_cagr_3y = calc_cagr(ni_old, ni_latest, years)
    except Exception:
        pass

    # 4) Free Cash Flow (FCF) = CFO - CapEx (annual latest)
    fcf = None
    try:
        if cf is not None and not cf.empty:
            cfo = safe_float(cf.loc["Total Cash From Operating Activities"].iloc[0]) if "Total Cash From Operating Activities" in cf.index else None
            capex = safe_float(cf.loc["Capital Expenditures"].iloc[0]) if "Capital Expenditures" in cf.index else None
            if cfo is not None and capex is not None:
                fcf = cfo - capex
    except Exception:
        pass

    # 5) Net Debt / EBITDA (from info if available)
    net_debt_to_ebitda = None
    net_debt = None
    ebitda = safe_float(info.get("ebitda"))
    total_debt = safe_float(info.get("totalDebt"))
    cash = safe_float(info.get("totalCash"))
    if total_debt is not None and cash is not None:
        net_debt = total_debt - cash
    if net_debt is not None and ebitda is not None and ebitda > 0:
        net_debt_to_ebitda = net_debt / ebitda

    # Fundamental score (simple: count “green”)
    fund_score = 0
    fund_score += 1 if (roce is not None and roce >= 0.15) else 0
    fund_score += 1 if (sales_cagr_3y is not None and sales_cagr_3y >= 0.12) else 0
    fund_score += 1 if (pat_cagr_3y is not None and pat_cagr_3y >= 0.12) else 0
    fund_score += 1 if (fcf is not None and fcf > 0) else 0
    fund_score += 1 if (net_debt_to_ebitda is not None and net_debt_to_ebitda <= 2) else 0

    # ---------------- Valuation ---------------- #
    mcap = safe_float(info.get("marketCap"))
    ev = safe_float(info.get("enterpriseValue"))
    trailing_pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))
    peg = safe_float(info.get("pegRatio"))
    eps_ttm = safe_float(info.get("trailingEps"))
    eps_fwd = safe_float(info.get("forwardEps"))
    ev_to_ebitda = safe_float(info.get("enterpriseToEbitda"))

    fair_price = None
    mos_price = None
    val_flag = "NA"

    if eps_fwd is not None and fair_pe is not None and fair_pe > 0:
        fair_price = eps_fwd * fair_pe
        mos_price = fair_price * (1 - mos_pct/100.0)

        if ltp is not None:
            if ltp <= mos_price:
                val_flag = "Cheap (>= MoS)"
            elif ltp <= fair_price:
                val_flag = "Fair"
            else:
                val_flag = "Expensive"

    # ---------------- Final Recommendation ---------------- #
    overall = "Hold"
    # Combine technical + fundamentals + valuation
    if tech_rec.startswith("Sell/Caution"):
        overall = "Sell/Caution"
    else:
        if fund_score >= 4 and val_flag in ["Cheap (>= MoS)", "Fair"] and tech_rec in ["Strong Buy (Trend)", "Hold (Pullback)", "Hold"]:
            overall = "Buy / Add in Tranches"
        elif fund_score >= 3:
            overall = "Hold / Watch"
        else:
            overall = "Hold (Weak Fundamentals)"

    return {
        # Identity
        "Ticker Used": symbol,

        # -------- TECHNICALS --------
        "TECH: LTP": fmt(ltp),
        "TECH: Prev Day SMA50": fmt(d50),
        "TECH: Prev Day SMA150": fmt(d150),
        "TECH: Prev Day SMA200": fmt(d200),
        "TECH: Signal": tech_rec,

        # -------- FUNDAMENTALS (5) --------
        "FND: ROCE % (approx)": fmt(roce * 100 if roce is not None else None),
        "FND: Sales CAGR 3Y % (approx)": fmt(sales_cagr_3y * 100 if sales_cagr_3y is not None else None),
        "FND: PAT CAGR 3Y % (approx)": fmt(pat_cagr_3y * 100 if pat_cagr_3y is not None else None),
        "FND: FCF Latest (₹)": fmt(fcf, 0),
        "FND: NetDebt/EBITDA": fmt(net_debt_to_ebitda),
        "FND: Score (0-5)": fund_score,

        # -------- VALUATION --------
        "VAL: MarketCap (₹)": fmt(mcap, 0),
        "VAL: Enterprise Value (₹)": fmt(ev, 0),
        "VAL: EV/EBITDA": fmt(ev_to_ebitda),
        "VAL: Trailing P/E": fmt(trailing_pe),
        "VAL: Forward P/E": fmt(forward_pe),
        "VAL: PEG": fmt(peg),
        "VAL: EPS (TTM)": fmt(eps_ttm),
        "VAL: EPS (Forward)": fmt(eps_fwd),
        "VAL: Fair P/E (Input)": fair_pe,
        "VAL: Fair Price": fmt(fair_price),
        "VAL: MoS % (Input)": mos_pct,
        "VAL: MoS Buy Price": fmt(mos_price),
        "VAL: Valuation Flag": val_flag,

        # -------- FINAL --------
        "FINAL: Recommendation": overall
    }

# ---------------- Main ---------------- #
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.success("File Uploaded Successfully!")
    df.columns = [c.lower().strip() for c in df.columns]

    # choose symbol column
    candidate_cols = [c for c in df.columns if "symbol" in c or "ticker" in c]
    if not candidate_cols:
        st.error("Could not find a symbol/ticker column. Add a column like 'stock symbol' or 'ticker'.")
        st.stop()

    symbol_col = st.selectbox("Select Symbol/Ticker column", candidate_cols)
    exchange = st.selectbox("Exchange suffix", options=[".NS (NSE)", ".BO (BSE)", "None"])
    suffix = ".NS" if exchange.startswith(".NS") else (".BO" if exchange.startswith(".BO") else "")

    st.markdown("### Valuation Inputs")
    fair_pe = st.number_input("Fair P/E to compute fair price (used only if Forward EPS exists)", min_value=1.0, value=30.0, step=1.0)
    mos_pct = st.number_input("Margin of Safety %", min_value=0.0, max_value=50.0, value=15.0, step=1.0)

    if st.button("Run Analysis"):
        with st.spinner("Analyzing stocks..."):
            results = []
            debug_rows = []

            prog = st.progress(0)
            total = len(df)

            for i, (_, row) in enumerate(df.iterrows()):
                raw_sym = row.get(symbol_col, "")
                sym = normalize_symbol(raw_sym, suffix)

                if not sym:
                    debug_rows.append({"Row": i, "Issue": "Blank symbol"})
                    continue

                out = compute_metrics(sym, fair_pe=fair_pe, mos_pct=mos_pct)

                out["Share Name"] = row.get("company name", sym)
                out["Qty"] = row.get("qnty", row.get("qty", 0))
                out["Avg Cost"] = row.get("avg cost", row.get("cost", None))

                results.append(out)
                prog.progress(int((i + 1) / total * 100))

            if results:
                res_df = pd.DataFrame(results)

                # Column ordering: Identity -> Technicals -> Fundamentals -> Valuation -> Final
                ordered_cols = (
                    ["Share Name", "Ticker Used", "Qty", "Avg Cost"] +
                    [c for c in res_df.columns if c.startswith("TECH:")] +
                    [c for c in res_df.columns if c.startswith("FND:")] +
                    [c for c in res_df.columns if c.startswith("VAL:")] +
                    [c for c in res_df.columns if c.startswith("FINAL:")]
                )
                ordered_cols = [c for c in ordered_cols if c in res_df.columns]

                st.dataframe(res_df[ordered_cols], use_container_width=True)

            if debug_rows:
                st.warning("Some rows/tickers had issues:")
                st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
