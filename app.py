import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide", page_title="Wealth Architect Pro (NSE)")
st.title("🏛️ Wealth Architect Pro — NSE Portfolio Audit (No-Freezing Edition)")

st.caption(
    "Designed for Streamlit Cloud reliability: bulk price fetch + per-ticker timeouts for fundamentals. "
    "If Yahoo blocks fundamentals, you will still get DMAs + momentum."
)

# -----------------------------
# Config
# -----------------------------
DEFAULT_MAX_WORKERS = 4          # keep low to reduce Yahoo throttling on Streamlit Cloud
FUNDAMENTALS_TIMEOUT_SEC = 12    # per stock timeout for fundamentals calls

SECTOR_DEFAULTS = {
    "Financial Services": {"method": "P/B", "target": 2.2},
    "Basic Materials": {"method": "CYCLICAL", "target": 12.0},
    "Energy": {"method": "CYCLICAL", "target": 10.0},
    "Industrials": {"method": "DEBT_ADJ", "target": 20.0},
    "Technology": {"method": "P/E", "target": 28.0},
    "Consumer Defensive": {"method": "P/E", "target": 45.0},
    "Consumer Cyclical": {"method": "P/E", "target": 30.0},
    "Healthcare": {"method": "P/E", "target": 35.0},
    "Utilities": {"method": "P/E", "target": 18.0},
    "Communication Services": {"method": "P/E", "target": 22.0},
    "Real Estate": {"method": "P/E", "target": 18.0},
    "Default": {"method": "P/E", "target": 20.0},
}

# -----------------------------
# Helpers
# -----------------------------
def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if not s:
        return ""
    if "." not in s:
        s = s + ".NS"  # default NSE
    return s

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None

def cagr_from_series(values: pd.Series, years: int = 3) -> Optional[float]:
    """
    values: Series indexed by date/period; latest first or last — we will sort by index.
    returns CAGR over N years as a percentage
    """
    try:
        s = values.dropna()
        if len(s) < (years + 1):
            return None
        s = s.sort_index()
        start = s.iloc[-(years + 1)]
        end = s.iloc[-1]
        if start <= 0 or end <= 0:
            return None
        return ((end / start) ** (1 / years) - 1) * 100
    except Exception:
        return None

def compute_roce_best_effort(info: dict, bs: Optional[pd.DataFrame], is_stmt: Optional[pd.DataFrame]) -> Optional[float]:
    """
    Best-effort ROCE:
      EBIT / (Total Assets - Current Liabilities)
    Many times not available → return None.
    """
    try:
        ebit = None
        if is_stmt is not None:
            # Try common keys
            for k in ["EBIT", "Ebit", "Operating Income", "OperatingIncome", "Earnings Before Interest and Taxes"]:
                if k in is_stmt.index:
                    ebit = safe_float(is_stmt.loc[k].dropna().iloc[-1])
                    break

        if ebit is None:
            # fallback: sometimes Yahoo provides operatingMargins but not EBIT in statements
            return None

        if bs is None:
            return None

        total_assets = None
        current_liab = None

        for k in ["Total Assets", "TotalAssets"]:
            if k in bs.index:
                total_assets = safe_float(bs.loc[k].dropna().iloc[-1])
                break

        for k in ["Total Current Liabilities", "TotalCurrentLiabilities"]:
            if k in bs.index:
                current_liab = safe_float(bs.loc[k].dropna().iloc[-1])
                break

        if total_assets is None or current_liab is None:
            return None

        capital_employed = total_assets - current_liab
        if capital_employed <= 0:
            return None

        return (ebit / capital_employed) * 100
    except Exception:
        return None

def valuation_fair_price(info: dict, sector: str) -> Tuple[Optional[float], str]:
    """
    One-pager valuation logic (best-effort):
    - Banks/Financials: Fair = BookValue * target P/B (adjusted by ROE)
    - Cyclicals: Fair = (Trailing EPS * 0.8) * target PE (normalize EPS)
    - Debt-adjusted: Fair = EPS * target PE * penalty if debtToEbitda high
    - Else: Fair = EPS * target PE
    """
    cfg = SECTOR_DEFAULTS.get(sector, SECTOR_DEFAULTS["Default"])
    method = cfg["method"]
    target = cfg["target"]

    roe = safe_float(info.get("returnOnEquity"))
    roe_pct = (roe * 100) if roe is not None else None

    if method == "P/B":
        bv = safe_float(info.get("bookValue"))
        pb = safe_float(info.get("priceToBook"))
        if bv is None or bv <= 0:
            return None, "P/B: missing bookValue"
        adj_target = target
        if roe_pct is not None:
            # scale around 15% ROE
            adj_target = target * (roe_pct / 15.0) if roe_pct > 0 else target
        fair = bv * adj_target
        return fair, f"P/B model (target~{round(adj_target,2)})"

    eps = safe_float(info.get("forwardEps")) or safe_float(info.get("trailingEps"))
    if eps is None or eps <= 0:
        return None, "P/E: missing EPS"

    if method == "CYCLICAL":
        fair = (eps * 0.8) * target
        return fair, f"Cyclical EPS normalized (0.8×EPS, PE={target})"

    if method == "DEBT_ADJ":
        d2e = safe_float(info.get("debtToEbitda"))
        penalty = 0.8 if (d2e is not None and d2e > 3) else 1.0
        fair = eps * target * penalty
        return fair, f"Debt-adj PE (PE={target}, penalty={penalty})"

    fair = eps * target
    return fair, f"PE model (PE={target})"

def momentum_state(ltp: float, d50: float, d150: float, d200: float) -> str:
    if ltp > d50 > d150 > d200:
        return "Bullish"
    if d50 < d200:
        return "Bearish (Death Cross Zone)"
    if ltp < d200:
        return "Bearish"
    return "Neutral"

# -----------------------------
# Fundamentals fetch with timeout safety
# -----------------------------
def fetch_fundamentals(symbol: str) -> Dict:
    """
    Best-effort Yahoo fundamentals. This can fail or be throttled. Must never freeze the whole app.
    """
    t = yf.Ticker(symbol)

    # Avoid very heavy calls first; info can be slow but is required for sector/EPS/PB etc.
    info = t.get_info() or {}

    # Income statement / balance sheet are often missing for NSE tickers
    is_stmt = None
    bs = None
    try:
        # yfinance versions differ; these attributes usually exist
        is_stmt = getattr(t, "financials", None)
        if isinstance(is_stmt, pd.DataFrame) and is_stmt.empty:
            is_stmt = None
    except Exception:
        is_stmt = None

    try:
        bs = getattr(t, "balance_sheet", None)
        if isinstance(bs, pd.DataFrame) and bs.empty:
            bs = None
    except Exception:
        bs = None

    # Try revenue + profit growth (3Y CAGR)
    rev_cagr = None
    prof_cagr = None
    if is_stmt is not None:
        # Common Yahoo keys
        rev_key_candidates = ["Total Revenue", "TotalRevenue"]
        prof_key_candidates = ["Net Income", "NetIncome"]

        rev_series = None
        prof_series = None

        for k in rev_key_candidates:
            if k in is_stmt.index:
                rev_series = is_stmt.loc[k]
                break
        for k in prof_key_candidates:
            if k in is_stmt.index:
                prof_series = is_stmt.loc[k]
                break

        if isinstance(rev_series, pd.Series):
            rev_cagr = cagr_from_series(rev_series, years=3)
        if isinstance(prof_series, pd.Series):
            prof_cagr = cagr_from_series(prof_series, years=3)

    roce = compute_roce_best_effort(info, bs, is_stmt)

    return {
        "info": info,
        "rev_cagr_3y": rev_cagr,
        "profit_cagr_3y": prof_cagr,
        "roce": roce,
    }

# -----------------------------
# Main
# -----------------------------
with st.sidebar:
    st.header("Settings")
    mos_val = st.slider("Margin of Safety %", 5, 40, 20)
    max_workers = st.slider("Parallel workers (Cloud-safe)", 1, 8, DEFAULT_MAX_WORKERS)
    st.markdown("---")
    st.caption("Tip: If you see many timeouts, reduce workers to 2–3.")

uploaded = st.file_uploader("Upload portfolio CSV (must include a symbol/ticker column)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]

    # Find symbol column
    tick_col = next((c for c in df.columns if "stock symbol" in c or "symbol" in c or "ticker" in c), None)
    name_col = next((c for c in df.columns if "company" in c or "name" in c), None)

    if not tick_col:
        st.error("Could not find a symbol column. Use a column like: Stock Symbol / Symbol / Ticker")
        st.stop()

    symbols = [normalize_symbol(x) for x in df[tick_col].dropna().astype(str).tolist()]
    symbols = [s for s in symbols if s]

    if st.button("🚀 Run Analysis (No Freeze)"):
        st.info(f"Processing {len(symbols)} tickers…")

        # 1) BULK PRICE FETCH (fast)
        with st.spinner("Fetching price history (bulk)…"):
            # group_by="ticker" gives multiindex columns for multiple tickers
            prices = yf.download(
                tickers=" ".join(symbols),
                period="1y",
                interval="1d",
                group_by="ticker",
                threads=True,
                auto_adjust=False,
                progress=False,
            )

        # Normalize prices into dict: sym -> close series
        close_map = {}
        if isinstance(prices.columns, pd.MultiIndex):
            for sym in symbols:
                if sym in prices.columns.get_level_values(0):
                    s = prices[sym]["Close"].dropna()
                    if len(s) > 0:
                        close_map[sym] = s
        else:
            # single ticker case
            s = prices["Close"].dropna()
            if len(s) > 0:
                close_map[symbols[0]] = s

        # 2) Fundamentals fetch with timeout (parallel but safe)
        results = []
        success = 0
        no_price = 0
        timeouts = 0
        errors = 0

        progress = st.progress(0)
        status = st.empty()

        def process_one(sym: str) -> Dict:
            # price-derived metrics
            close = close_map.get(sym)
            if close is None or len(close) < 210:
                return {"Ticker": sym, "ERROR": "No price / insufficient history"}

            ltp = float(close.iloc[-1])
            d50 = float(close.rolling(50).mean().iloc[-1])
            d150 = float(close.rolling(150).mean().iloc[-1])
            d200 = float(close.rolling(200).mean().iloc[-1])

            # fundamentals (timeout-protected outside)
            return {
                "Ticker": sym,
                "TECH: LTP": round(ltp, 2),
                "TECH: 50DMA": round(d50, 2),
                "TECH: 150DMA": round(d150, 2),
                "TECH: 200DMA": round(d200, 2),
            }

        # First build base rows from price
        base_rows = {sym: process_one(sym) for sym in symbols}

        # Now enrich fundamentals using futures
        with st.spinner("Fetching fundamentals (best-effort, timeout protected)…"):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(fetch_fundamentals, sym): sym for sym in symbols}

                completed = 0
                for fut in as_completed(future_map):
                    sym = future_map[fut]
                    completed += 1
                    progress.progress(completed / len(symbols))
                    status.write(f"Fundamentals: {sym} ({completed}/{len(symbols)})")

                    row = base_rows.get(sym, {"Ticker": sym})

                    # If no price data, keep it and skip fundamentals
                    if "ERROR" in row:
                        no_price += 1
                        results.append(row)
                        continue

                    try:
                        data = fut.result(timeout=FUNDAMENTALS_TIMEOUT_SEC)
                        info = data.get("info", {}) or {}
                        sector = info.get("sector") or "Default"

                        roe = safe_float(info.get("returnOnEquity"))
                        roe_pct = round((roe * 100), 2) if roe is not None else None

                        pb = safe_float(info.get("priceToBook"))
                        pb_val = round(pb, 2) if pb is not None else None

                        # Sales/profit growth CAGR 3Y best-effort
                        sales_g = data.get("rev_cagr_3y")
                        prof_g = data.get("profit_cagr_3y")
                        roce = data.get("roce")

                        fair, val_note = valuation_fair_price(info, sector)
                        mos_price = (fair * (1 - mos_val / 100)) if (fair is not None) else None

                        mom = momentum_state(
                            row["TECH: LTP"], row["TECH: 50DMA"], row["TECH: 150DMA"], row["TECH: 200DMA"]
                        )

                        # Final recommendation logic
                        rec = "Neutral / Wait"
                        if fair is None:
                            rec = "Insufficient Data"
                        else:
                            if row["TECH: LTP"] <= mos_price and mom != "Bearish":
                                rec = "Strong Buy / Add"
                            elif row["TECH: LTP"] <= fair:
                                rec = "Hold / Watch"
                            else:
                                rec = "Avoid / Expensive"

                        # Confidence / coverage
                        coverage = []
                        if roe_pct is not None: coverage.append("ROE")
                        if pb_val is not None: coverage.append("PB")
                        if sales_g is not None: coverage.append("Sales3Y")
                        if prof_g is not None: coverage.append("Profit3Y")
                        if roce is not None: coverage.append("ROCE")
                        coverage_str = ",".join(coverage) if coverage else "Price-only"

                        row.update({
                            "FND: Sector": sector,
                            "FND: ROE": roe_pct,
                            "FND: ROCE": round(roce, 2) if roce is not None else None,
                            "FND: Sales growth % (YOY-3 years)": round(sales_g, 2) if sales_g is not None else None,
                            "FND: Profit growth % (YOY 3years)": round(prof_g, 2) if prof_g is not None else None,
                            "VAL: PB": pb_val,
                            "VAL: Fair Price": round(fair, 2) if fair is not None else None,
                            "VAL: MoS Buy": round(mos_price, 2) if mos_price is not None else None,
                            "VAL: Method Note": val_note,
                            "FINAL: Momentum": mom,
                            "FINAL: Recommendation": rec,
                            "FINAL: Data Coverage": coverage_str
                        })

                        success += 1
                        results.append(row)

                    except TimeoutError:
                        timeouts += 1
                        row.update({
                            "FND: Sector": None,
                            "FINAL: Recommendation": "Timeout (Yahoo throttling)",
                            "FINAL: Data Coverage": "Price-only"
                        })
                        results.append(row)
                    except Exception:
                        errors += 1
                        row.update({
                            "FND: Sector": None,
                            "FINAL: Recommendation": "Error in fundamentals",
                            "FINAL: Data Coverage": "Price-only"
                        })
                        results.append(row)

        status.empty()
        progress.empty()

        res_df = pd.DataFrame(results)

        # Attach company name if present in uploaded file
        if name_col:
            # make a mapping for display
            m = {}
            for _, r in df.iterrows():
                sym = normalize_symbol(r.get(tick_col, ""))
                nm = r.get(name_col, "")
                if sym and nm:
                    m[sym] = nm
            res_df.insert(0, "Company", res_df["Ticker"].map(m).fillna(res_df["Ticker"]))

        # Make columns order friendly
        preferred = [c for c in [
            "Company", "Ticker",
            "TECH: LTP", "TECH: 50DMA", "TECH: 150DMA", "TECH: 200DMA",
            "FND: Sector", "FND: Sales growth % (YOY-3 years)", "FND: Profit growth % (YOY 3years)",
            "FND: ROE", "FND: ROCE",
            "VAL: PB", "VAL: Fair Price", "VAL: MoS Buy", "VAL: Method Note",
            "FINAL: Momentum", "FINAL: Recommendation", "FINAL: Data Coverage",
            "ERROR"
        ] if c in res_df.columns]

        res_df = res_df[preferred] if preferred else res_df

        st.success(
            f"Done. Success: {success} | No-price: {no_price} | Timeouts: {timeouts} | Errors: {errors}"
        )

        st.dataframe(res_df, use_container_width=True, hide_index=True)

        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results (CSV)",
            data=csv_bytes,
            file_name="wealth_architect_results.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown(
    "### Notes\n"
    "- **Sales/Profit growth & ROCE** depend on Yahoo providing financial statements for the ticker. For many NSE stocks, Yahoo sometimes returns **blank**.\n"
    "- The app will still work using **DMA + momentum + best-effort valuation**.\n"
    "- If you see many **Timeouts**, reduce parallel workers to **2–3** and re-run.\n"
)
