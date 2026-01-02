# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# =========================
# UI + CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Wealth Architect Pro (NSE)")
st.title("🏛️ Wealth Architect Pro")
st.caption("NSE-only portfolio audit using Yahoo Finance (free). Includes technicals, fundamentals (best-effort), and sector-aware valuation.")

# EXACT output columns & order (must be a LIST, not a set)
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

# Sector valuation defaults (simple + explainable)
SECTOR_DEFAULTS = {
    "Financial Services": {"method": "P/B", "target_pb": 2.2},
    "Technology": {"method": "P/E", "target_pe": 28.0},
    "Consumer Defensive": {"method": "P/E", "target_pe": 40.0},
    "Consumer Cyclical": {"method": "P/E", "target_pe": 30.0},
    "Healthcare": {"method": "P/E", "target_pe": 35.0},
    "Industrials": {"method": "DEBT_ADJ_PE", "target_pe": 22.0},
    "Utilities": {"method": "CYCLICAL_PE", "target_pe": 14.0},
    "Energy": {"method": "CYCLICAL_PE", "target_pe": 10.0},
    "Basic Materials": {"method": "CYCLICAL_PE", "target_pe": 12.0},
    "Real Estate": {"method": "DEBT_ADJ_PE", "target_pe": 18.0},
    "Communication Services": {"method": "P/E", "target_pe": 20.0},
    "Default": {"method": "P/E", "target_pe": 20.0},
}

# =========================
# Helpers
# =========================
def _as_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        x = str(x).replace(",", "").strip()
        if x == "" or x.lower() in {"nan", "none", "na"}:
            return None
        return float(x)
    except Exception:
        return None

def _fmt(x, nd=2):
    v = _as_float(x)
    if v is None or np.isnan(v):
        return "NA"
    return round(v, nd)

def _pct_fmt(x, nd=2):
    v = _as_float(x)
    if v is None or np.isnan(v):
        return "NA"
    return round(v, nd)

def _clean_nse_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if s == "" or s.lower() == "nan":
        return ""
    # If user already passed .NS or .BO, keep it
    if "." not in s:
        s = s + ".NS"
    return s

def _pick_col(df: pd.DataFrame, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    # fallback: any column containing keyword
    for c in df.columns:
        cl = c.lower().strip()
        for cand in candidates:
            if cand in cl:
                return c
    return None

def _safe_div(a, b):
    a = _as_float(a)
    b = _as_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b

def _cagr(v0, v1, years: float):
    v0 = _as_float(v0)
    v1 = _as_float(v1)
    if v0 is None or v1 is None or v0 <= 0 or v1 <= 0 or years <= 0:
        return None
    return (v1 / v0) ** (1.0 / years) - 1.0

def _latest_from_stmt(stmt: pd.DataFrame, row_names):
    """Return latest value for any of row_names from financial statement dataframe."""
    if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.empty:
        return None
    for rn in row_names:
        if rn in stmt.index:
            series = stmt.loc[rn]
            if hasattr(series, "dropna"):
                series = series.dropna()
            if len(series) == 0:
                continue
            # columns are dates, pick latest by column
            try:
                # ensure columns sortable
                cols = list(series.index)
                # series is indexed by columns (dates)
                # take first element from the left? safer: take max column label
                # But yfinance often orders newest->oldest; still, choose the first non-null after sorting
                return float(series.iloc[0])
            except Exception:
                try:
                    return float(series.values[0])
                except Exception:
                    return None
    return None

def _get_annual_series(stmt: pd.DataFrame, row_names):
    """Return a list of (date, value) for the first matching row_name."""
    if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.empty:
        return []
    for rn in row_names:
        if rn in stmt.index:
            s = stmt.loc[rn]
            if isinstance(s, pd.Series):
                s = s.dropna()
                # s index = columns (dates)
                out = []
                for k, v in s.items():
                    try:
                        out.append((k, float(v)))
                    except Exception:
                        continue
                # try to sort by date descending
                try:
                    out = sorted(out, key=lambda x: str(x[0]), reverse=True)
                except Exception:
                    pass
                return out
    return []

def _compute_3y_growth_from_stmt(stmt: pd.DataFrame, row_names):
    """
    Best-effort 3Y CAGR using annual statement:
    need >= 4 annual points (current vs 3 years ago).
    Returns percent (0-100) or None.
    """
    series = _get_annual_series(stmt, row_names)
    if len(series) < 4:
        return None
    # Use latest and the 4th point as ~3 years ago
    latest_val = series[0][1]
    old_val = series[3][1]
    g = _cagr(old_val, latest_val, 3.0)
    if g is None:
        return None
    return g * 100

def _momentum_state(ltp, d50, d200):
    ltp = _as_float(ltp)
    d50 = _as_float(d50)
    d200 = _as_float(d200)
    if ltp is None or d50 is None or d200 is None:
        return "NA"
    if ltp > d50 > d200:
        return "Bullish"
    if d50 < d200 or ltp < d200:
        return "Bearish"
    return "Neutral"

def _valuation(symbol, info, roe_pct, pb):
    """
    Returns:
      fair_price, method_str, rationale_str
    """
    sector = (info.get("sector") or "Default")
    cfg = SECTOR_DEFAULTS.get(sector, SECTOR_DEFAULTS["Default"])

    # EPS choices
    eps_fwd = _as_float(info.get("forwardEps"))
    eps_ttm = _as_float(info.get("trailingEps"))
    eps_use = eps_fwd if (eps_fwd is not None and eps_fwd > 0) else eps_ttm

    # Book value
    book_value = _as_float(info.get("bookValue"))

    # Net debt / EBITDA (for debt adjusted)
    total_debt = _as_float(info.get("totalDebt"))
    total_cash = _as_float(info.get("totalCash"))
    ebitda = _as_float(info.get("ebitda"))
    net_debt_to_ebitda = None
    if total_debt is not None and total_cash is not None and ebitda is not None and ebitda > 0:
        net_debt_to_ebitda = (total_debt - total_cash) / ebitda

    method = cfg["method"]
    rationale = []
    fair = None

    if method == "P/B":
        # Fair PB based on target_pb adjusted by ROE (very simple heuristic)
        target_pb = _as_float(cfg.get("target_pb")) or 2.2
        roe = _as_float(roe_pct)
        adj = 1.0
        if roe is not None:
            if roe >= 18:
                adj = 1.10
            elif roe <= 10:
                adj = 0.85
        fair_pb = target_pb * adj
        if book_value is not None and book_value > 0:
            fair = book_value * fair_pb
            rationale.append(f"P/B method: BookValue×FairPB ({round(fair_pb,2)}x)")
        else:
            rationale.append("P/B method: BookValue missing")
    elif method == "CYCLICAL_PE":
        # Normalize EPS with haircut to avoid peak earnings overvaluation
        target_pe = _as_float(cfg.get("target_pe")) or 12.0
        if eps_use is not None and eps_use > 0:
            norm_eps = eps_use * 0.80
            fair = norm_eps * target_pe
            rationale.append(f"Cyclical P/E: (0.8×EPS)×{round(target_pe,2)}")
        else:
            rationale.append("Cyclical P/E: EPS missing")
    elif method == "DEBT_ADJ_PE":
        target_pe = _as_float(cfg.get("target_pe")) or 20.0
        penalty = 1.0
        if net_debt_to_ebitda is not None:
            if net_debt_to_ebitda > 3:
                penalty = 0.80
                rationale.append(f"Debt penalty: NetDebt/EBITDA={round(net_debt_to_ebitda,2)} (>3)")
            else:
                rationale.append(f"Debt OK: NetDebt/EBITDA={round(net_debt_to_ebitda,2)}")
        else:
            rationale.append("Debt metric NA")
        if eps_use is not None and eps_use > 0:
            fair = eps_use * target_pe * penalty
            rationale.append(f"Debt-adj P/E: EPS×{round(target_pe,2)}×{round(penalty,2)}")
        else:
            rationale.append("Debt-adj P/E: EPS missing")
    else:
        target_pe = _as_float(cfg.get("target_pe")) or 20.0
        if eps_use is not None and eps_use > 0:
            fair = eps_use * target_pe
            rationale.append(f"P/E method: EPS×{round(target_pe,2)}")
        else:
            rationale.append("P/E method: EPS missing")

    # Include PB as a reported metric even if method isn't PB
    method_str = f"{method} | Sector={sector}"
    return fair, method_str, " ; ".join(rationale)

def _recommendation(ltp, d50, d200, fair, mos_buy, roe, roce):
    """
    Returns: recommendation, reason
    """
    mom = _momentum_state(ltp, d50, d200)

    ltp = _as_float(ltp)
    fair = _as_float(fair)
    mos_buy = _as_float(mos_buy)
    roe = _as_float(roe)
    roce = _as_float(roce)

    quality_good = False
    if roe is not None and roce is not None:
        quality_good = (roe >= 12 and roce >= 12)
    elif roe is not None:
        quality_good = (roe >= 12)

    if ltp is None:
        return "Hold", "Price data missing"

    if fair is None or mos_buy is None:
        # Can't value -> default to momentum + trend guardrails
        if mom == "Bearish":
            return "Sell", "Bearish trend & insufficient valuation data"
        if mom == "Bullish":
            return "Hold", "Bullish trend but valuation data NA"
        return "Hold", "Valuation data NA"

    # Value + trend combined
    if ltp <= mos_buy and mom != "Bearish":
        if quality_good:
            return "Buy", "Price below MoS Buy + trend not bearish + quality OK"
        return "Buy", "Price below MoS Buy + trend not bearish"
    if ltp <= fair:
        if mom == "Bearish":
            return "Hold", "Valuation supportive but trend bearish (wait)"
        return "Hold", "Valuation supportive (near/under Fair)"
    # Over fair
    if mom == "Bearish":
        return "Sell", "Over Fair + bearish trend"
    return "Hold", "Over Fair (no margin of safety)"

# =========================
# Data Fetch (cached)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_bundle(symbol: str):
    """
    Best-effort fetch for NSE tickers:
      - info
      - 2y history for DMAs
      - annual income stmt + balance sheet (if available)
    """
    t = yf.Ticker(symbol)

    # INFO (can be slow/missing; keep it safe)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # PRICE HISTORY
    try:
        hist = t.history(period="2y", auto_adjust=False)
    except Exception:
        hist = pd.DataFrame()

    # FINANCIALS (annual)
    income = None
    balance = None
    try:
        # yfinance versions differ: try a few
        if hasattr(t, "income_stmt"):
            income = t.income_stmt
        elif hasattr(t, "financials"):
            income = t.financials
    except Exception:
        income = None

    try:
        if hasattr(t, "balance_sheet"):
            balance = t.balance_sheet
    except Exception:
        balance = None

    return info, hist, income, balance

def compute_dmas(hist: pd.DataFrame):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None, None, None, None
    close = hist["Close"].dropna()
    if len(close) < 210:
        # need enough for 200DMA
        ltp = close.iloc[-1] if len(close) > 0 else None
        d50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        d150 = close.rolling(150).mean().iloc[-1] if len(close) >= 150 else None
        d200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        return ltp, d50, d150, d200
    ltp = close.iloc[-1]
    d50 = close.rolling(50).mean().iloc[-1]
    d150 = close.rolling(150).mean().iloc[-1]
    d200 = close.rolling(200).mean().iloc[-1]
    return ltp, d50, d150, d200

def compute_roce(info: dict, income: pd.DataFrame, balance: pd.DataFrame):
    """
    ROCE best-effort:
      EBIT / (Total Assets - Current Liabilities)
    Uses latest annual values if available.
    """
    # EBIT
    ebit = None
    if income is not None and isinstance(income, pd.DataFrame) and not income.empty:
        ebit = _latest_from_stmt(income, ["EBIT", "Ebit", "Operating Income", "OperatingIncome"])
    # If EBIT missing, try operating income from info (rare)
    if ebit is None:
        ebit = info.get("ebitda")  # not EBIT but better than nothing; flagged by NA if fails

    # Capital employed
    cap_emp = None
    if balance is not None and isinstance(balance, pd.DataFrame) and not balance.empty:
        total_assets = _latest_from_stmt(balance, ["Total Assets", "TotalAssets"])
        curr_liab = _latest_from_stmt(balance, ["Total Current Liabilities", "Current Liabilities", "TotalCurrentLiabilities", "CurrentLiabilities"])
        if total_assets is not None and curr_liab is not None:
            cap_emp = total_assets - curr_liab

    if ebit is None or cap_emp is None:
        return None

    roce = _safe_div(ebit, cap_emp)
    if roce is None:
        return None
    return roce * 100

def analyze_one(symbol: str, company_name: str, mos_pct: float):
    # bundle
    info, hist, income, balance = fetch_yahoo_bundle(symbol)

    # dmAs
    ltp, d50, d150, d200 = compute_dmas(hist)

    # fundamentals
    roe = info.get("returnOnEquity")
    roe_pct = (roe * 100) if isinstance(roe, (int, float, np.floating)) else None

    roce_pct = compute_roce(info, income, balance)

    # revenue/profit growth (3Y CAGR best-effort)
    sales_g_3y = _compute_3y_growth_from_stmt(income, ["Total Revenue", "TotalRevenue", "Revenue"])
    profit_g_3y = _compute_3y_growth_from_stmt(income, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"])

    # PB
    pb = info.get("priceToBook")

    # valuation
    fair, val_method, val_rationale = _valuation(symbol, info, roe_pct, pb)
    mos_buy = (fair * (1 - mos_pct / 100)) if _as_float(fair) is not None else None

    # momentum
    momentum = _momentum_state(ltp, d50, d200)

    # final reco
    reco, reason = _recommendation(ltp, d50, d200, fair, mos_buy, roe_pct, roce_pct)

    # add valuation rationale into reason if useful
    if val_rationale and val_rationale != "NA":
        reason = f"{reason} | {val_rationale}"

    row = {
        "Company": company_name if company_name else symbol,
        "LTP": _fmt(ltp),
        "50DMA": _fmt(d50),
        "150DMA": _fmt(d150),
        "200DMA": _fmt(d200),
        "Sales growth % (YOY-3 years)": _pct_fmt(sales_g_3y),
        "Profit growth % (YOY 3years)": _pct_fmt(profit_g_3y),
        "ROE": _pct_fmt(roe_pct),
        "ROCE": _pct_fmt(roce_pct),
        "VAL: PB": _fmt(pb),
        "VAL: Fair": _fmt(fair),
        "VAL: MoS Buy": _fmt(mos_buy),
        "VAL: Method": val_method if val_method else "NA",
        "Momentum": momentum,
        "Recommendation": reco,
        "Reason": reason,
    }

    # ensure exact columns exist
    return {k: row.get(k, "NA") for k in OUTPUT_COLUMNS}

def style_reco(val):
    v = str(val).lower()
    if v == "buy":
        return "color:#16a34a;font-weight:700"  # green
    if v == "sell":
        return "color:#dc2626;font-weight:700"  # red
    return "color:#b45309;font-weight:700"      # amber

# =========================
# UI Controls
# =========================
with st.sidebar:
    st.header("Settings")
    mos_pct = st.slider("Margin of Safety %", 5, 40, 20)
    max_workers = st.slider("Speed (parallel workers)", 2, 12, 6)
    st.caption("Higher workers = faster, but Yahoo can throttle. If it stalls, reduce workers.")

uploaded = st.file_uploader("Upload Portfolio CSV/XLSX", type=["csv", "xlsx"])

if uploaded:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    df.columns = [c.lower().strip() for c in df.columns]

    ticker_col = _pick_col(df, ["stock symbol", "symbol", "ticker", "nse", "code"])
    company_col = _pick_col(df, ["company", "company name", "name"])

    if not ticker_col:
        st.error("Could not find a ticker column. Please include a column like: 'stock symbol' or 'ticker' or 'symbol'.")
        st.stop()

    st.success(f"Loaded {len(df)} rows. Using ticker column: '{ticker_col}'" + (f", company column: '{company_col}'" if company_col else ""))

    if st.button("🚀 RUN ANALYSIS"):
        tickers = []
        for _, r in df.iterrows():
            t = _clean_nse_symbol(r.get(ticker_col, ""))
            if t:
                comp = str(r.get(company_col, "")).strip() if company_col else t
                tickers.append((t, comp))

        # de-dup by ticker, keep first company name
        seen = set()
        unique = []
        for t, c in tickers:
            if t not in seen:
                unique.append((t, c))
                seen.add(t)

        if not unique:
            st.warning("No valid tickers found.")
            st.stop()

        st.info(f"Analyzing {len(unique)} tickers…")
        progress = st.progress(0)
        status = st.empty()

        results = []
        done = 0

        # parallel fetch to avoid "stuck" feel
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for t, c in unique:
                futures[ex.submit(analyze_one, t, c, mos_pct)] = t

            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    row = fut.result()
                    if row:
                        results.append(row)
                except Exception as e:
                    # keep a row with NA but don't crash
                    results.append({k: ("NA" if k != "Company" else t) for k in OUTPUT_COLUMNS})
                done += 1
                status.write(f"Processed: {t} ({done}/{len(unique)})")
                progress.progress(done / len(unique))

        status.empty()

        out = pd.DataFrame(results)
        # enforce exact order
        out = out[OUTPUT_COLUMNS]

        st.subheader("Results")
        st.dataframe(
            out.style.applymap(style_reco, subset=["Recommendation"]),
            use_container_width=True,
            hide_index=True
        )

        # download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name=f"wealth_architect_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.caption("Notes: Yahoo Finance data for NSE fundamentals can be incomplete. This app shows NA instead of guessing.")
else:
    st.info("Upload a CSV/XLSX with at least a ticker column (e.g., 'stock symbol'). Example tickers: INFY, TCS, RELIANCE (app auto-adds .NS).")
