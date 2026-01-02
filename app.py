# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(layout="wide", page_title="Wealth Architect Pro")
st.title("🏛️ Wealth Architect Pro — Sector-Aware Valuation + Quality + MoS")

# -----------------------------
# 1) Sector configuration
# -----------------------------
SECTOR_BANDS = {
    "Financial Services": {"type": "PB", "min": 1.0, "max": 4.0},
    "Technology": {"type": "PE", "min": 18, "max": 45},
    "Consumer Defensive": {"type": "PE", "min": 30, "max": 65},
    "Consumer Cyclical": {"type": "PE", "min": 20, "max": 45},
    "Industrials": {"type": "PE", "min": 15, "max": 35},
    "Basic Materials": {"type": "PE", "min": 6, "max": 18},
    "Healthcare": {"type": "PE", "min": 25, "max": 50},
    "Utilities": {"type": "PE", "min": 10, "max": 20},
    "Energy": {"type": "PE", "min": 8, "max": 15},
    "Communication Services": {"type": "PE", "min": 15, "max": 30},
    "Real Estate": {"type": "PE", "min": 15, "max": 35},
    "Default": {"type": "PE", "min": 15, "max": 30},
}

# Industry keyword-based fallback mapping (yfinance sector is often missing for NSE)
def infer_sector_from_industry(industry: str) -> str:
    ind = (industry or "").lower()
    if any(k in ind for k in ["bank", "nbfc", "finance", "financial", "insurance", "asset management", "broking"]):
        return "Financial Services"
    if any(k in ind for k in ["software", "it services", "information technology", "technology", "semiconductor"]):
        return "Technology"
    if any(k in ind for k in ["pharma", "pharmaceutical", "hospital", "healthcare", "diagnostic", "medtech"]):
        return "Healthcare"
    if any(k in ind for k in ["power", "utility", "electric", "water utility", "gas utility"]):
        return "Utilities"
    if any(k in ind for k in ["oil", "gas", "refining", "energy", "exploration"]):
        return "Energy"
    if any(k in ind for k in ["cement", "steel", "aluminium", "metal", "mining", "paper", "chemical", "materials"]):
        return "Basic Materials"
    if any(k in ind for k in ["telecom", "media", "entertainment", "communication"]):
        return "Communication Services"
    if any(k in ind for k in ["real estate", "reit", "property"]):
        return "Real Estate"
    if any(k in ind for k in ["fmcg", "beverage", "foods", "consumer", "retail", "apparel"]):
        return "Consumer Defensive"
    if any(k in ind for k in ["auto", "automobile", "hotel", "travel", "leisure", "consumer cyclical"]):
        return "Consumer Cyclical"
    if any(k in ind for k in ["industrial", "engineering", "capital goods", "construction", "infrastructure", "logistics"]):
        return "Industrials"
    return "Default"

# -----------------------------
# 2) Styling
# -----------------------------
def color_decision(val):
    s = str(val)
    if "Strong Buy" in s or "Add" in s:
        return "color:#28a745;font-weight:700"
    if "Avoid" in s or "Sell" in s or "Bearish" in s:
        return "color:#dc3545;font-weight:700"
    if "Hold" in s or "Watch" in s:
        return "color:#ffc107;font-weight:700"
    return "color:#6c757d;font-weight:700"

def safe_num(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def r2(x, nd=2):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None

def normalize_symbol(sym: str, default_suffix: str = ".NS") -> str:
    s = str(sym).strip().upper()
    if s == "" or s == "NAN":
        return ""
    if s.endswith((".NS", ".BO")):
        return s
    # If user typed something like "INFY.NS " with spaces, handled above
    if "." in s and (s.endswith(".NS") or s.endswith(".BO")):
        return s
    return s + default_suffix

# -----------------------------
# 3) Core data fetch (cached)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_yf(symbol: str):
    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    hist = t.history(period="2y", auto_adjust=False)
    return info, hist

# -----------------------------
# 4) Valuation + Quality engine
# -----------------------------
def analyze_wealth_engine(symbol: str, mos_pct: float, ttm_pe_haircut: float, coe: float, g: float):
    """
    Returns a dict of metrics or None if prices missing.
    - mos_pct: margin of safety percentage
    - ttm_pe_haircut: e.g., 0.85 means fair PE is reduced when using TTM EPS instead of forward EPS
    - coe: cost of equity (for bank P/B justification optional)
    - g: long-term growth assumption (for bank P/B justification optional)
    """
    info, hist = fetch_yf(symbol)

    if hist is None or hist.empty:
        return None

    close = hist.get("Close")
    if close is None:
        return None

    close = close.dropna()
    if len(close) < 210:
        # Not enough data for 200DMA
        return {
            "Ticker Used": symbol,
            "Data Status": "Insufficient Price History (<210 trading days)",
        }

    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])

    # Momentum state
    if ltp > d50 > d200:
        momentum = "Bullish"
    elif ltp < d200 or d50 < d200:
        momentum = "Bearish"
    else:
        momentum = "Neutral"

    # Sector selection
    sector = info.get("sector")
    industry = info.get("industry")
    if not sector:
        sector = infer_sector_from_industry(industry)
    sector_cfg = SECTOR_BANDS.get(sector, SECTOR_BANDS["Default"])
    val_method = sector_cfg["type"]

    # Key fundamentals (yfinance may be sparse for NSE; track coverage)
    revenue_growth = safe_num(info.get("revenueGrowth"))
    op_margins = safe_num(info.get("operatingMargins"))
    fcf = safe_num(info.get("freeCashflow"))
    roe = safe_num(info.get("returnOnEquity"))
    payout = safe_num(info.get("payoutRatio"))
    pe_ttm = safe_num(info.get("trailingPE"))
    pe_fwd = safe_num(info.get("forwardPE"))
    eps_ttm = safe_num(info.get("trailingEps"))
    eps_fwd = safe_num(info.get("forwardEps"))
    peg = safe_num(info.get("pegRatio"))
    pb = safe_num(info.get("priceToBook"))
    book = safe_num(info.get("bookValue"))
    ev_to_ebitda = safe_num(info.get("enterpriseToEbitda"))

    ebitda = safe_num(info.get("ebitda"))
    total_debt = safe_num(info.get("totalDebt"))
    cash = safe_num(info.get("totalCash"))
    mcap = safe_num(info.get("marketCap"))
    ev = safe_num(info.get("enterpriseValue"))

    # Data coverage (important for business confidence)
    missing_q = 0
    if revenue_growth is None:
        missing_q += 1
    if op_margins is None:
        missing_q += 1
    if fcf is None:
        missing_q += 1

    coverage = {0: "High", 1: "Medium", 2: "Low", 3: "Very Low"}.get(missing_q, "Very Low")

    # Net debt / EBITDA
    net_debt_to_ebitda = None
    if ebitda is not None and ebitda > 0 and total_debt is not None and cash is not None:
        net_debt_to_ebitda = (total_debt - cash) / ebitda

    # ---------------- Hard stops (risk gates) ----------------
    hard_stop = False
    reasons = []

    # Leverage gate (stricter than 4; you can tune in sidebar)
    if net_debt_to_ebitda is not None and net_debt_to_ebitda > 3.0 and sector not in ["Utilities"]:
        hard_stop = True
        reasons.append("NetDebt/EBITDA > 3")

    if payout is not None and payout > 1.0:
        hard_stop = True
        reasons.append("Dividend > Earnings")

    # If momentum is bearish, we won't “Strong Buy/Add” (but we may still “Hold/Watch”)
    # ---------------- Quality score (0–100) ----------------
    q_score = 0
    roe_pct = (roe * 100) if roe is not None else None

    if roe_pct is not None and roe_pct > 15:
        q_score += 30
    if revenue_growth is not None and revenue_growth > 0.12:
        q_score += 20
    if op_margins is not None and op_margins > 0.15:
        q_score += 20
    if fcf is not None and fcf > 0:
        q_score += 30

    # Confidence score (0–100)
    conf = 85
    conf -= 10 * missing_q
    if eps_fwd is None and eps_ttm is None and val_method != "PB":
        conf -= 25
    if val_method == "PB" and (book is None or book <= 0):
        conf -= 25
    if len(close) < 260:
        conf -= 5
    if net_debt_to_ebitda is not None and net_debt_to_ebitda > 3:
        conf -= 10
    conf = max(0, min(100, conf))

    # ---------------- Valuation (sector-aware) ----------------
    fair_val = None
    fair_band_low = None
    fair_band_high = None
    valuation_basis = None

    if val_method == "PB":
        # Financials: use book value × justified PB (simple version)
        # justified PB ≈ (ROE - g) / (CoE - g), clamped to sector PB band
        if book is not None and book > 0 and roe is not None and roe > 0:
            coe_f = max(0.05, coe)  # guardrails
            g_f = min(g, coe_f - 0.01)  # ensure coe > g
            justified_pb = (roe - g_f) / (coe_f - g_f) if (coe_f - g_f) > 0 else None

            # fallback to midpoint PB if formula fails
            if justified_pb is None or not np.isfinite(justified_pb) or justified_pb <= 0:
                justified_pb = (sector_cfg["min"] + sector_cfg["max"]) / 2

            # adjust slightly for very high/low ROE
            if roe * 100 > 18:
                justified_pb *= 1.05
            elif roe * 100 < 10:
                justified_pb *= 0.90

            # clamp
            justified_pb = max(sector_cfg["min"], min(sector_cfg["max"], justified_pb))
            fair_val = book * justified_pb
            valuation_basis = f"P/B (justified PB={r2(justified_pb,2)})"

            # range
            fair_band_low = fair_val * 0.90
            fair_band_high = fair_val * 1.10
    else:
        # Non-financials: Forward EPS preferred, else TTM EPS with haircut
        eps = None
        used_eps = None
        base_pe = (sector_cfg["min"] + sector_cfg["max"]) / 2

        # PEG moderation (only if PEG available)
        if peg is not None:
            if peg < 1.2:
                base_pe *= 1.10
            elif peg > 2.0:
                base_pe *= 0.85

        # clamp PE in band
        base_pe = max(sector_cfg["min"], min(sector_cfg["max"], base_pe))

        if eps_fwd is not None and eps_fwd > 0:
            eps = eps_fwd
            used_eps = "Forward EPS"
            eff_pe = base_pe
        elif eps_ttm is not None and eps_ttm > 0:
            eps = eps_ttm
            used_eps = "TTM EPS"
            eff_pe = base_pe * float(ttm_pe_haircut)  # haircut for using TTM
        else:
            eps = None
            used_eps = None
            eff_pe = None

        if eps is not None and eff_pe is not None and eff_pe > 0:
            fair_val = eps * eff_pe
            valuation_basis = f"P/E ({used_eps}, fairPE={r2(eff_pe,2)})"
            # range: widen a bit for cyclicals/materials/energy
            if sector in ["Basic Materials", "Energy"]:
                fair_band_low = fair_val * 0.80
                fair_band_high = fair_val * 1.20
            else:
                fair_band_low = fair_val * 0.90
                fair_band_high = fair_val * 1.10

    # If no fair value, return with status
    if fair_val is None or fair_val <= 0:
        return {
            "Ticker Used": symbol,
            "Sector": sector,
            "Industry": industry,
            "Data Coverage": coverage,
            "Confidence": conf,
            "TECH: LTP": r2(ltp),
            "TECH: 50DMA": r2(d50),
            "TECH: 200DMA": r2(d200),
            "TECH: Momentum": momentum,
            "FND: ROE %": r2(roe_pct, 1) if roe_pct is not None else None,
            "FND: NetDebt/EBITDA": r2(net_debt_to_ebitda, 2),
            "FND: RevGrowth": r2(revenue_growth * 100, 1) if revenue_growth is not None else None,
            "FND: OpMargin": r2(op_margins * 100, 1) if op_margins is not None else None,
            "FND: FCF": r2(fcf, 0),
            "FND: Quality Score": q_score,
            "VAL: Basis": None,
            "VAL: Fair Value": None,
            "VAL: Fair Low": None,
            "VAL: Fair High": None,
            "VAL: MoS Buy": None,
            "VAL: EV/EBITDA": r2(ev_to_ebitda, 2),
            "VAL: P/E (TTM)": r2(pe_ttm, 2),
            "VAL: P/E (Fwd)": r2(pe_fwd, 2),
            "VAL: PEG": r2(peg, 2),
            "VAL: P/B": r2(pb, 2),
            "FINAL: Verdict": "Insufficient Data (Valuation NA)",
            "FINAL: Recommendation": "Hold / Watch",
            "FINAL: Reasons": "Missing EPS/BookValue needed for valuation"
        }

    mos_price = fair_val * (1 - mos_pct / 100)

    # Valuation verdict
    if ltp <= mos_price:
        verdict = "Cheap (>= MoS)"
    elif ltp <= fair_val:
        verdict = "Fair"
    else:
        verdict = "Expensive"

    # Mispricing %
    mispricing = (fair_val / ltp - 1) * 100 if ltp > 0 else None

    # Final recommendation logic
    if hard_stop:
        rec = f"Avoid ({', '.join(reasons)})"
    elif q_score >= 70 and ltp <= mos_price and momentum != "Bearish":
        rec = "Strong Buy / Add"
    elif q_score >= 55 and verdict in ["Cheap (>= MoS)", "Fair"] and momentum != "Bearish":
        rec = "Hold / Accumulate on dips"
    elif momentum == "Bearish":
        rec = "Hold / Watch (Bearish trend)"
    else:
        rec = "Neutral / Wait"

    return {
        "Ticker Used": symbol,
        "Sector": sector,
        "Industry": industry,
        "Data Coverage": coverage,
        "Confidence": conf,

        # TECH
        "TECH: LTP": r2(ltp),
        "TECH: 50DMA": r2(d50),
        "TECH: 200DMA": r2(d200),
        "TECH: Momentum": momentum,

        # FUNDAMENTALS / RISK
        "FND: ROE %": r2(roe_pct, 1) if roe_pct is not None else None,
        "FND: NetDebt/EBITDA": r2(net_debt_to_ebitda, 2),
        "FND: RevGrowth %": r2(revenue_growth * 100, 1) if revenue_growth is not None else None,
        "FND: OpMargin %": r2(op_margins * 100, 1) if op_margins is not None else None,
        "FND: FCF": r2(fcf, 0),
        "FND: Quality Score": q_score,

        # VALUATION
        "VAL: Basis": valuation_basis,
        "VAL: Fair Value": r2(fair_val),
        "VAL: Fair Low": r2(fair_band_low),
        "VAL: Fair High": r2(fair_band_high),
        "VAL: MoS Buy": r2(mos_price),
        "VAL: % Cheap/Exp": r2(mispricing, 1),
        "VAL: EV/EBITDA": r2(ev_to_ebitda, 2),
        "VAL: P/E (TTM)": r2(pe_ttm, 2),
        "VAL: P/E (Fwd)": r2(pe_fwd, 2),
        "VAL: PEG": r2(peg, 2),
        "VAL: P/B": r2(pb, 2),

        # SIZE (optional)
        "SIZE: MarketCap": r2(mcap, 0),
        "SIZE: EnterpriseValue": r2(ev, 0),

        # FINAL
        "FINAL: Verdict": verdict,
        "FINAL: Recommendation": rec,
        "FINAL: Reasons": "; ".join(reasons) if reasons else ""
    }

# -----------------------------
# 5) UI
# -----------------------------
uploaded_file = st.file_uploader("Upload Portfolio (CSV or XLSX)", type=["csv", "xlsx"])

st.sidebar.header("Controls")
suffix = st.sidebar.selectbox("Default exchange suffix", [".NS (NSE)", ".BO (BSE)"], index=0)
default_suffix = ".NS" if suffix.startswith(".NS") else ".BO"

mos = st.sidebar.slider("Margin of Safety %", 5, 40, 20)
ttm_haircut = st.sidebar.slider("TTM EPS haircut on Fair P/E (when forward EPS missing)", 0.60, 1.00, 0.85, 0.05)

st.sidebar.subheader("Financials (P/B Justification) Inputs")
coe = st.sidebar.slider("Cost of Equity (CoE)", 0.08, 0.20, 0.14, 0.01)
g = st.sidebar.slider("Long-term Growth (g)", 0.03, 0.14, 0.08, 0.01)
if g >= coe:
    st.sidebar.warning("Keep g < CoE for justified P/B to behave sensibly.")

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
    st.success("File uploaded.")

    df.columns = [c.lower().strip() for c in df.columns]

    # Pick symbol column
    symbol_candidates = [c for c in df.columns if "symbol" in c or "ticker" in c]
    if not symbol_candidates:
        st.error("No symbol column found. Add a column like 'stock symbol' or 'ticker'.")
        st.stop()

    symbol_col = st.selectbox("Select the column that contains the ticker/symbol", symbol_candidates)

    name_candidates = [c for c in df.columns if "company" in c or "name" in c]
    name_col = st.selectbox("Select company name column (optional)", ["(none)"] + name_candidates)

    qty_candidates = [c for c in df.columns if c in ["qty", "qnty", "quantity", "shares"]]
    qty_col = st.selectbox("Select quantity column (optional)", ["(none)"] + qty_candidates)

    cost_candidates = [c for c in df.columns if "avg" in c or "cost" in c or "buy" in c]
    cost_col = st.selectbox("Select avg cost column (optional)", ["(none)"] + cost_candidates)

    if st.button("Run Robust Multi-Factor Analysis"):
        results = []
        status = st.empty()
        prog = st.progress(0)
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            raw_sym = row.get(symbol_col, "")
            sym = normalize_symbol(raw_sym, default_suffix)

            if not sym:
                continue

            status.text(f"Analyzing: {sym} ...")
            out = analyze_wealth_engine(sym, mos, ttm_haircut, coe, g)

            if out:
                # Attach portfolio columns
                out["Company"] = (row.get(name_col) if name_col != "(none)" else None) or sym
                out["Portfolio Qty"] = (row.get(qty_col) if qty_col != "(none)" else None)
                out["Avg Cost"] = (row.get(cost_col) if cost_col != "(none)" else None)
                results.append(out)

            prog.progress(int((i + 1) / total * 100))

        if not results:
            st.error("No results generated. Check that symbols are valid and include NSE tickers.")
            st.stop()

        res_df = pd.DataFrame(results)

        # Make sure columns are ordered in a business-friendly way
        ordered = (
            ["Company", "Ticker Used", "Portfolio Qty", "Avg Cost", "Sector", "Industry", "Data Coverage", "Confidence"] +
            [c for c in res_df.columns if c.startswith("TECH:")] +
            [c for c in res_df.columns if c.startswith("FND:")] +
            [c for c in res_df.columns if c.startswith("VAL:")] +
            [c for c in res_df.columns if c.startswith("SIZE:")] +
            [c for c in res_df.columns if c.startswith("FINAL:")]
        )
        ordered = [c for c in ordered if c in res_df.columns]
        res_df = res_df[ordered]

        st.subheader("Results")
        st.dataframe(
            res_df.style.applymap(color_decision, subset=["FINAL: Recommendation"]),
            use_container_width=True
        )

        # Download output
        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="wealth_architect_results.csv",
            mime="text/csv"
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
    "\n"
    "**Tip:** For NSE symbols, use `INFY` / `TCS` etc. The app auto-appends `.NS` by default.\n")
