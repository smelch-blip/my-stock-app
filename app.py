import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# UI CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Wealth Architect Pro")
st.title("🏛️ Wealth Architect Pro — NSE Portfolio Audit")

st.caption(
    "This version renders your output UI exactly like your template "
    "(same metric names + same sequence + grouped header)."
)

# =========================
# SECTOR DEFAULTS (Valuation)
# =========================
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


# =========================
# HELPERS
# =========================
def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if not s:
        return ""
    if "." not in s:
        s += ".NS"
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


def pct_fmt(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), 2)
    except Exception:
        return None


def round2(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), 2)
    except Exception:
        return None


def calc_cagr(series: pd.Series, years: int = 3) -> Optional[float]:
    """
    Uses a 3Y CAGR as a practical "YOY-3 years" proxy if we have >= 4 annual points.
    """
    try:
        s = series.dropna()
        if len(s) < (years + 1):
            return None
        s = s.sort_index()
        start = float(s.iloc[-(years + 1)])
        end = float(s.iloc[-1])
        if start <= 0 or end <= 0:
            return None
        return ((end / start) ** (1 / years) - 1) * 100
    except Exception:
        return None


def momentum_label(ltp: float, d50: float, d150: float, d200: float) -> str:
    if ltp > d50 > d150 > d200:
        return "Bullish"
    if d50 < d200:
        return "Bearish"
    if ltp < d200:
        return "Bearish"
    return "Neutral"


def valuation_fair_price(info: dict, sector: str) -> (Optional[float], str):
    cfg = SECTOR_DEFAULTS.get(sector, SECTOR_DEFAULTS["Default"])
    method = cfg["method"]
    target = cfg["target"]

    roe = safe_float(info.get("returnOnEquity"))
    roe_pct = (roe * 100) if roe is not None else None

    if method == "P/B":
        bv = safe_float(info.get("bookValue"))
        if bv is None or bv <= 0:
            return None, "P/B: missing bookValue"
        adj_target = target
        if roe_pct is not None and roe_pct > 0:
            adj_target = target * (roe_pct / 15.0)
        fair = bv * adj_target
        return fair, f"P/B (target~{round(adj_target, 2)})"

    eps = safe_float(info.get("forwardEps")) or safe_float(info.get("trailingEps"))
    if eps is None or eps <= 0:
        return None, "P/E: missing EPS"

    if method == "CYCLICAL":
        fair = (eps * 0.8) * target  # normalize EPS
        return fair, f"CYCLICAL (0.8×EPS, PE={target})"

    if method == "DEBT_ADJ":
        d2e = safe_float(info.get("debtToEbitda"))
        penalty = 0.8 if (d2e is not None and d2e > 3) else 1.0
        fair = eps * target * penalty
        return fair, f"DEBT_ADJ (PE={target}, penalty={penalty})"

    fair = eps * target
    return fair, f"P/E (PE={target})"


def compute_roce_best_effort(info: dict, bs: Optional[pd.DataFrame], is_stmt: Optional[pd.DataFrame]) -> Optional[float]:
    """
    Best-effort ROCE = EBIT / (Total Assets - Current Liabilities).
    Often NA for NSE on Yahoo. We'll return None if missing.
    """
    try:
        if is_stmt is None or bs is None:
            return None

        ebit = None
        for k in ["EBIT", "Ebit", "Operating Income", "OperatingIncome"]:
            if k in is_stmt.index:
                ebit = safe_float(is_stmt.loc[k].dropna().iloc[-1])
                break
        if ebit is None:
            return None

        total_assets = None
        for k in ["Total Assets", "TotalAssets"]:
            if k in bs.index:
                total_assets = safe_float(bs.loc[k].dropna().iloc[-1])
                break

        current_liab = None
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


# =========================
# FUNDAMENTALS FETCH (best-effort)
# =========================
def fetch_fundamentals(symbol: str) -> Dict:
    t = yf.Ticker(symbol)
    info = t.get_info() or {}

    # financial statements (often missing for NSE)
    is_stmt = None
    bs = None
    try:
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

    # sales/profit growth (3Y CAGR proxy) if we can read statements
    sales_cagr = None
    profit_cagr = None
    if is_stmt is not None:
        rev = None
        prof = None
        for k in ["Total Revenue", "TotalRevenue"]:
            if k in is_stmt.index:
                rev = is_stmt.loc[k]
                break
        for k in ["Net Income", "NetIncome"]:
            if k in is_stmt.index:
                prof = is_stmt.loc[k]
                break

        if isinstance(rev, pd.Series):
            sales_cagr = calc_cagr(rev, years=3)
        if isinstance(prof, pd.Series):
            profit_cagr = calc_cagr(prof, years=3)

    roce = compute_roce_best_effort(info, bs, is_stmt)

    return {
        "info": info,
        "sales_cagr": sales_cagr,
        "profit_cagr": profit_cagr,
        "roce": roce
    }


# =========================
# EXACT UI TABLE RENDER (your grouped header)
# =========================
def render_grouped_table(df: pd.DataFrame) -> str:
    """
    Renders HTML with a two-row header:
      Row1: groups (Technicals/Fundamentals/Valuation/Final)
      Row2: exact metric names in exact order
    """
    # Colors close to your screenshot
    col_left = "#cfeecf"      # light green
    col_tech = "#bfe3ff"      # light blue
    col_fnd = "#c8f5c8"       # light green
    col_val = "#bfe3ff"       # light blue
    col_final = "#bfe3ff"     # light blue

    css = f"""
    <style>
      .wa-table {{
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 13px;
      }}
      .wa-table th, .wa-table td {{
        border: 1px solid #d0d0d0;
        padding: 8px;
        text-align: center;
        vertical-align: middle;
        white-space: nowrap;
      }}
      .wa-table thead tr:first-child th {{
        font-weight: 700;
        font-size: 14px;
      }}
      .wa-left {{ background: {col_left}; }}
      .wa-tech {{ background: {col_tech}; }}
      .wa-fnd {{ background: {col_fnd}; }}
      .wa-val {{ background: {col_val}; }}
      .wa-final {{ background: {col_final}; }}
      .wa-table tbody tr:hover {{ background: #f7fbff; }}
      .wa-num {{ text-align: right; }}
      .wa-text {{ text-align: left; }}
    </style>
    """

    # Group header row (colspans must match your layout)
    # Columns EXACT order & names as requested:
    # Company, LTP,
    # 50DMA, 150DMA, 200DMA,
    # Sales growth % (YOY-3 years), Profit growth % (YOY 3years), ROE, ROCE,
    # VAL: PB, VAL: Fair, VAL: MoS Buy, VAL: Method,
    # Momentum, Recommendation, Reason

    header_row_1 = """
    <tr>
      <th class="wa-left" colspan="2"></th>
      <th class="wa-tech" colspan="3">Technicals</th>
      <th class="wa-fnd" colspan="4">Fundamentals</th>
      <th class="wa-val" colspan="4">Valuation</th>
      <th class="wa-final" colspan="3">Final</th>
    </tr>
    """

    header_row_2_cells = []
    # Left (2)
    header_row_2_cells += [
        '<th class="wa-left">Company</th>',
        '<th class="wa-left">LTP</th>',
    ]
    # Technicals (3)
    header_row_2_cells += [
        '<th class="wa-tech">50DMA</th>',
        '<th class="wa-tech">150DMA</th>',
        '<th class="wa-tech">200DMA</th>',
    ]
    # Fundamentals (4)
    header_row_2_cells += [
        '<th class="wa-fnd">Sales growth % (YOY-3 years)</th>',
        '<th class="wa-fnd">Profit growth % (YOY 3years)</th>',
        '<th class="wa-fnd">ROE</th>',
        '<th class="wa-fnd">ROCE</th>',
    ]
    # Valuation (4)
    header_row_2_cells += [
        '<th class="wa-val">VAL: PB</th>',
        '<th class="wa-val">VAL: Fair</th>',
        '<th class="wa-val">VAL: MoS Buy</th>',
        '<th class="wa-val">VAL: Method</th>',
    ]
    # Final (3)
    header_row_2_cells += [
        '<th class="wa-final">Momentum</th>',
        '<th class="wa-final">Recommendation</th>',
        '<th class="wa-final">Reason</th>',
    ]

    header_row_2 = "<tr>" + "".join(header_row_2_cells) + "</tr>"

    # Body rows
    def td(val, cls="wa-num"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return f'<td class="{cls}">NA</td>'
        return f'<td class="{cls}">{val}</td>'

    body = ""
    for _, r in df.iterrows():
        body += "<tr>"
        body += td(r["Company"], cls="wa-text")
        body += td(r["LTP"])
        body += td(r["50DMA"])
        body += td(r["150DMA"])
        body += td(r["200DMA"])
        body += td(r["Sales growth % (YOY-3 years)"])
        body += td(r["Profit growth % (YOY 3years)"])
        body += td(r["ROE"])
        body += td(r["ROCE"])
        body += td(r["VAL: PB"])
        body += td(r["VAL: Fair"])
        body += td(r["VAL: MoS Buy"])
        body += td(r["VAL: Method"], cls="wa-text")
        body += td(r["Momentum"], cls="wa-text")
        body += td(r["Recommendation"], cls="wa-text")
        body += td(r["Reason"], cls="wa-text")
        body += "</tr>"

    html = f"""
    {css}
    <table class="wa-table">
      <thead>
        {header_row_1}
        {header_row_2}
      </thead>
      <tbody>
        {body}
      </tbody>
    </table>
    """
    return html


# =========================
# APP FLOW
# =========================
with st.sidebar:
    st.header("Settings")
    mos = st.slider("Margin of Safety %", 5, 40, 20)
    workers = st.slider("Parallel workers", 1, 8, 4)
    overall_timeout = st.slider("Overall fundamentals timeout (sec)", 20, 240, 90)
    st.caption("If it feels slow or shows many timeouts, reduce workers to 2–3.")

uploaded = st.file_uploader("Upload Portfolio CSV (must have Symbol/Ticker column)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]

    tick_col = next((c for c in df.columns if "stock symbol" in c or "symbol" in c or "ticker" in c), None)
    name_col = next((c for c in df.columns if "company" in c or "name" in c), None)

    if not tick_col:
        st.error("Your CSV must contain a column like: Stock Symbol / Symbol / Ticker")
        st.stop()

    symbols = [normalize_symbol(x) for x in df[tick_col].dropna().astype(str).tolist()]
    symbols = [s for s in symbols if s]
    name_map = {}
    if name_col:
        for _, r in df.iterrows():
            sym = normalize_symbol(r.get(tick_col, ""))
            nm = str(r.get(name_col, "")).strip()
            if sym and nm:
                name_map[sym] = nm

    if st.button("🚀 RUN ANALYSIS"):
        st.info(f"Running analysis for {len(symbols)} tickers…")

        # -------- 1) BULK PRICE FETCH (FAST)
        with st.spinner("Fetching prices (bulk)…"):
            raw = yf.download(
                tickers=" ".join(symbols),
                period="1y",
                interval="1d",
                group_by="ticker",
                threads=True,
                auto_adjust=False,
                progress=False,
            )

        close_map: Dict[str, pd.Series] = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for sym in symbols:
                if sym in raw.columns.get_level_values(0):
                    s = raw[sym]["Close"].dropna()
                    if len(s) > 0:
                        close_map[sym] = s
        else:
            s = raw["Close"].dropna()
            if len(s) > 0:
                close_map[symbols[0]] = s

        # Build base rows from price
        base_rows = {}
        for sym in symbols:
            close = close_map.get(sym)
            if close is None or len(close) < 210:
                base_rows[sym] = {
                    "Company": name_map.get(sym, sym),
                    "LTP": None,
                    "50DMA": None,
                    "150DMA": None,
                    "200DMA": None,
                }
                continue

            ltp = float(close.iloc[-1])
            d50 = float(close.rolling(50).mean().iloc[-1])
            d150 = float(close.rolling(150).mean().iloc[-1])
            d200 = float(close.rolling(200).mean().iloc[-1])

            base_rows[sym] = {
                "Company": name_map.get(sym, sym),
                "LTP": round2(ltp),
                "50DMA": round2(d50),
                "150DMA": round2(d150),
                "200DMA": round2(d200),
            }

        # -------- 2) FUNDAMENTALS / VALUATION (BEST-EFFORT)
        results: List[Dict] = []
        progress = st.progress(0)
        status = st.empty()

        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(fetch_fundamentals, sym): sym for sym in symbols}

            # We use overall timeout to avoid "forever waiting" if Yahoo hangs
            try:
                for fut in as_completed(fut_map, timeout=overall_timeout):
                    sym = fut_map[fut]
                    completed += 1
                    progress.progress(min(1.0, completed / max(1, len(symbols))))
                    status.write(f"Fundamentals: {sym} ({completed}/{len(symbols)})")

                    row = dict(base_rows.get(sym, {}))

                    # If no price, still output row with NA
                    ltp = row.get("LTP")
                    d50 = row.get("50DMA")
                    d150 = row.get("150DMA")
                    d200 = row.get("200DMA")

                    try:
                        data = fut.result()
                        info = data.get("info", {}) or {}
                        sector = info.get("sector") or "Default"

                        # Fundamentals
                        sales_g = pct_fmt(data.get("sales_cagr"))
                        prof_g = pct_fmt(data.get("profit_cagr"))
                        roe = safe_float(info.get("returnOnEquity"))
                        roe_pct = pct_fmt(roe * 100) if roe is not None else None
                        roce = pct_fmt(data.get("roce"))

                        # Valuation
                        pb = safe_float(info.get("priceToBook"))
                        pb_v = round2(pb) if pb is not None else None

                        fair, method_note = valuation_fair_price(info, sector)
                        fair_v = round2(fair) if fair is not None else None
                        mos_buy = round2(fair * (1 - mos / 100)) if fair is not None else None

                        # Final
                        mom = None
                        if all(v is not None for v in [ltp, d50, d150, d200]):
                            mom = momentum_label(ltp, d50, d150, d200)

                        # Recommendation + reason
                        rec = "Insufficient Data"
                        reason = "Missing fundamentals from data source"
                        if fair is not None and ltp is not None:
                            if mom == "Bearish" and ltp > (mos_buy or 0):
                                rec = "Hold"
                                reason = "Bearish momentum; wait for trend to stabilize"
                            elif mos_buy is not None and ltp <= mos_buy and mom != "Bearish":
                                rec = "Buy"
                                reason = "Price below MoS buy + momentum not bearish"
                            elif fair_v is not None and ltp <= fair_v:
                                rec = "Hold"
                                reason = "Near/under fair value; not at MoS yet"
                            else:
                                rec = "Wait"
                                reason = "Above fair value / no margin of safety"

                        row.update({
                            "Sales growth % (YOY-3 years)": sales_g,
                            "Profit growth % (YOY 3years)": prof_g,
                            "ROE": roe_pct,
                            "ROCE": roce,
                            "VAL: PB": pb_v,
                            "VAL: Fair": fair_v,
                            "VAL: MoS Buy": mos_buy,
                            "VAL: Method": method_note,
                            "Momentum": mom,
                            "Recommendation": rec,
                            "Reason": reason
                        })

                    except Exception:
                        row.update({
                            "Sales growth % (YOY-3 years)": None,
                            "Profit growth % (YOY 3years)": None,
                            "ROE": None,
                            "ROCE": None,
                            "VAL: PB": None,
                            "VAL: Fair": None,
                            "VAL: MoS Buy": None,
                            "VAL: Method": "Error/NA",
                            "Momentum": None,
                            "Recommendation": "Hold",
                            "Reason": "Fundamentals fetch failed; relying on technicals only"
                        })

                    results.append(row)

            except Exception:
                # overall timeout hit — fill remaining as timeout
                pending_syms = [s for s in symbols if s not in {r.get("Company") for r in results}]
                # We'll just not hard-guess; table will still be complete from base_rows below.

                pass

        status.empty()
        progress.empty()

        # Ensure EVERY symbol is present in output
        # (results might miss some if overall timeout triggers)
        output_map = {r["Company"]: r for r in results if "Company" in r}
        final_rows = []
        for sym in symbols:
            company = name_map.get(sym, sym)
            if company in output_map:
                final_rows.append(output_map[company])
            else:
                # fallback to price-only row
                row = dict(base_rows.get(sym, {}))
                row.update({
                    "Sales growth % (YOY-3 years)": None,
                    "Profit growth % (YOY 3years)": None,
                    "ROE": None,
                    "ROCE": None,
                    "VAL: PB": None,
                    "VAL: Fair": None,
                    "VAL: MoS Buy": None,
                    "VAL: Method": "Timeout/NA",
                    "Momentum": None,
                    "Recommendation": "Hold",
                    "Reason": "Timeout fetching fundamentals; price-only"
                })
                final_rows.append(row)

        # EXACT column order (as your template)
        out = pd.DataFrame(final_rows)[[
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
            "Reason"
        ]]

        st.success(f"Analysis complete in ~{int(time.time() - start_time)}s")

        # Render EXACT UI layout
        st.markdown(render_grouped_table(out), unsafe_allow_html=True)

        # Download
        st.download_button(
            "Download results (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="wealth_architect_results.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown(
    "### Notes\n"
    "- For many NSE tickers, Yahoo sometimes returns **NA** for financial statements; in that case we still output the row using price-based technicals.\n"
    "- If the app feels slow, reduce **Parallel workers** to 2–3.\n"
)
