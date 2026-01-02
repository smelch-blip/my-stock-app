import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests


# =========================
# UI + CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Wealth Architect Pro (NSE)")
st.title("🏛️ Wealth Architect Pro")
st.caption("NSE-only portfolio audit using Yahoo Finance (free). Prices are reliable; NSE fundamentals can be incomplete, so the app shows NA instead of guessing.")

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

# Sector/industry mapping + valuation bands
# (Yahoo “sector” for NSE is often missing; we map from both sector + industry keywords.)
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

# Make Yahoo calls a bit more stable
def _yahoo_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
        }
    )
    return s


def _clean_nse_symbol(sym: str) -> str:
    sym = str(sym or "").strip().upper()
    if not sym:
        return ""
    # if user gives RELIANCE / TCS etc -> append .NS
    if "." not in sym:
        sym = sym + ".NS"
    # allow .BO too, but you asked NSE-only; keep .NS if neither
    if not sym.endswith((".NS", ".BO")):
        sym = sym + ".NS"
    return sym


def _as_na(x):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else x


def _safe_float(x):
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None


def _cagr(a, b, years: float):
    # CAGR from a -> b over years
    a = _safe_float(a)
    b = _safe_float(b)
    if a is None or b is None or years <= 0:
        return None
    if a <= 0 or b <= 0:
        return None
    return (b / a) ** (1.0 / years) - 1.0


def _sector_bucket(sector: str, industry: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()
    text = f"{s} {i}"

    if any(k in text for k in ["bank", "nbfc", "insurance", "financial", "capital markets", "asset management"]):
        return "Financial"
    if any(k in text for k in ["software", "it services", "information technology", "technology"]):
        return "IT"
    if any(k in text for k in ["fmcg", "beverage", "retail", "consumer", "restaurants"]):
        return "Consumer"
    if any(k in text for k in ["auto", "tyre", "tires", "automobile", "two wheelers", "four wheelers"]):
        return "Auto"
    if any(k in text for k in ["industrial", "engineering", "capital goods", "defence", "defense", "construction"]):
        return "Industrials"
    if any(k in text for k in ["pharma", "hospital", "healthcare", "diagnostics", "biotech"]):
        return "Healthcare"
    if any(k in text for k in ["oil", "gas", "coal", "power", "energy", "utilities"]):
        return "Energy"
    if any(k in text for k in ["metal", "steel", "aluminium", "copper", "zinc", "mining"]):
        return "Metals"

    return "Default"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_batch(tickers: list[str]):
    # Batch fetch avoids “NA for everything” due to per-ticker throttling
    session = _yahoo_session()
    data = yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
        session=session,
    )
    return data


def _extract_close_series(batch_df: pd.DataFrame, ticker: str) -> pd.Series | None:
    try:
        # Multi-ticker -> columns like (AAPL, Close), (AAPL, Open)...
        if isinstance(batch_df.columns, pd.MultiIndex):
            if (ticker, "Close") in batch_df.columns:
                s = batch_df[(ticker, "Close")].dropna()
                return s if len(s) else None
            # Sometimes single ticker still comes differently; fallbacks:
            if ticker in batch_df.columns.get_level_values(0):
                s = batch_df[ticker]["Close"].dropna()
                return s if len(s) else None
        else:
            # Single ticker case: columns are Open/High/Low/Close...
            if "Close" in batch_df.columns:
                s = batch_df["Close"].dropna()
                return s if len(s) else None
        return None
    except Exception:
        return None


def compute_dmas(close: pd.Series):
    if close is None or len(close) < 205:
        return None, None, None, None
    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])
    d150 = float(close.rolling(150).mean().iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])
    return ltp, d50, d150, d200


def momentum_label(ltp, d50, d150, d200):
    if any(v is None for v in [ltp, d50, d200]):
        return "NA"
    if ltp > d50 and d50 > d150 and d150 > d200:
        return "Bullish"
    if ltp < d200 and d50 < d200:
        return "Bearish"
    return "Neutral"


def _retry(fn, tries=2, base_sleep=0.6):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(base_sleep * (2 ** k))
    return None


def fetch_fundamentals_one(ticker: str):
    """
    Best-effort fundamentals for NSE:
    - ROE: from info.returnOnEquity
    - PB: from info.priceToBook
    - 3Y Sales/Profit CAGR: from annual income statement (if Yahoo provides)
    - ROCE: computed from EBIT and (TotalAssets - CurrentLiabilities) if statements exist
    """
    session = _yahoo_session()
    t = yf.Ticker(ticker, session=session)

    info = _retry(lambda: t.info, tries=2) or {}
    sector = info.get("sector")
    industry = info.get("industry")

    pb = _safe_float(info.get("priceToBook"))
    roe = _safe_float(info.get("returnOnEquity"))
    roe_pct = (roe * 100.0) if roe is not None else None

    # Income statement yearly (preferred) - yfinance variants
    income = None
    balance = None

    def _get_income():
        # Try newer API first
        inc = getattr(t, "get_income_stmt", None)
        if callable(inc):
            return inc(freq="yearly")
        # Fallbacks
        if hasattr(t, "income_stmt"):
            return t.income_stmt
        return t.financials

    def _get_balance():
        bs = getattr(t, "get_balance_sheet", None)
        if callable(bs):
            return bs(freq="yearly")
        if hasattr(t, "balance_sheet"):
            return t.balance_sheet
        return t.balance_sheet

    income = _retry(_get_income, tries=2)
    balance = _retry(_get_balance, tries=2)

    # 3-year Sales CAGR (Revenue)
    sales_cagr = None
    profit_cagr = None
    roce_pct = None

    try:
        if isinstance(income, pd.DataFrame) and not income.empty:
            # columns are dates; ensure sorted oldest->newest
            cols = list(income.columns)
            cols_sorted = sorted(cols)
            income2 = income[cols_sorted]

            # Try common row labels
            revenue_row_candidates = ["Total Revenue", "Operating Revenue", "Revenue"]
            net_income_row_candidates = ["Net Income", "NetIncome", "Profit After Tax", "Net Profit"]

            def _pick_row(df, candidates):
                for r in candidates:
                    if r in df.index:
                        return df.loc[r]
                return None

            rev = _pick_row(income2, revenue_row_candidates)
            ni = _pick_row(income2, net_income_row_candidates)

            # Need latest and ~3 years back => use last 4 annual points if available
            if rev is not None and len(rev.dropna()) >= 4:
                r = rev.dropna()
                sales_cagr = _cagr(r.iloc[-4], r.iloc[-1], 3.0)

            if ni is not None and len(ni.dropna()) >= 4:
                p = ni.dropna()
                # If profit negative in base year, CAGR isn't meaningful; return NA
                profit_cagr = _cagr(p.iloc[-4], p.iloc[-1], 3.0)

            # ROCE: EBIT / (TotalAssets - CurrentLiabilities)
            ebit_row_candidates = ["EBIT", "Operating Income", "OperatingIncome"]
            ebit = _pick_row(income2, ebit_row_candidates)
            if isinstance(balance, pd.DataFrame) and not balance.empty and ebit is not None:
                bcols = sorted(list(balance.columns))
                bs2 = balance[bcols]

                # total assets
                ta = None
                for r in ["Total Assets", "TotalAssets"]:
                    if r in bs2.index:
                        ta = bs2.loc[r]
                        break
                # current liabilities
                cl = None
                for r in ["Total Current Liabilities", "Current Liabilities", "TotalCurrentLiabilities"]:
                    if r in bs2.index:
                        cl = bs2.loc[r]
                        break

                if ta is not None and cl is not None:
                    ta = ta.dropna()
                    cl = cl.dropna()
                    ebit = ebit.dropna()
                    if len(ta) and len(cl) and len(ebit):
                        cap_employed = _safe_float(ta.iloc[-1] - cl.iloc[-1])
                        ebit_latest = _safe_float(ebit.iloc[-1])
                        if cap_employed and cap_employed > 0 and ebit_latest is not None:
                            roce_pct = (ebit_latest / cap_employed) * 100.0
    except Exception:
        pass

    # EPS for valuation
    eps = _safe_float(info.get("forwardEps")) or _safe_float(info.get("trailingEps"))
    book_value = _safe_float(info.get("bookValue"))

    return {
        "sector": sector,
        "industry": industry,
        "bucket": _sector_bucket(sector, industry),
        "roe": roe_pct,
        "roce": roce_pct,
        "pb": pb,
        "sales_cagr": (sales_cagr * 100.0) if sales_cagr is not None else None,
        "profit_cagr": (profit_cagr * 100.0) if profit_cagr is not None else None,
        "eps": eps,
        "book_value": book_value,
    }


def compute_valuation(ltp, fundamentals: dict, mos_pct: float):
    """
    Returns:
      VAL: PB, VAL: Fair, VAL: MoS Buy, VAL: Method
    """
    pb = fundamentals.get("pb")
    bucket = fundamentals.get("bucket", "Default")
    cfg = SECTOR_BANDS.get(bucket, SECTOR_BANDS["Default"])

    fair = None
    method = None

    if cfg["method"] == "P/B":
        # Banks/NBFCs: fair = book * fair_pb (adjusted by ROE)
        bv = fundamentals.get("book_value")
        roe = fundamentals.get("roe")
        if bv is not None and bv > 0:
            fair_pb = cfg["pb_mid"]
            if roe is not None:
                if roe >= 16:
                    fair_pb *= 1.10
                elif roe <= 10:
                    fair_pb *= 0.85
            fair_pb = max(cfg["pb_min"], min(cfg["pb_max"], fair_pb))
            fair = bv * fair_pb
            method = f"P/B | bucket={bucket}"
        else:
            method = f"P/B | bucket={bucket} (no bookValue)"
    elif cfg["method"] == "CYCLICAL":
        # Cyclical: normalized EPS haircut
        eps = fundamentals.get("eps")
        if eps is not None and eps > 0:
            norm_eps = eps * 0.80
            pe = cfg["pe_mid"]
            fair = norm_eps * pe
            method = f"CYCLICAL P/E | bucket={bucket}"
        else:
            method = f"CYCLICAL P/E | bucket={bucket} (no EPS)"
    else:
        # Standard P/E with quality adjustment
        eps = fundamentals.get("eps")
        if eps is not None and eps > 0:
            pe = cfg["pe_mid"]

            # Adjust PE slightly by profitability / growth if available
            roce = fundamentals.get("roce")
            pg = fundamentals.get("profit_cagr")
            if roce is not None:
                if roce >= 18:
                    pe *= 1.10
                elif roce <= 10:
                    pe *= 0.85
            if pg is not None:
                if pg >= 20:
                    pe *= 1.05
                elif pg <= 5:
                    pe *= 0.90

            pe = max(cfg["pe_min"], min(cfg["pe_max"], pe))
            fair = eps * pe
            method = f"P/E | bucket={bucket} | PE={round(pe,1)}"
        else:
            method = f"P/E | bucket={bucket} (no EPS)"

    mos_buy = None
    if fair is not None and fair > 0:
        mos_buy = fair * (1.0 - mos_pct / 100.0)

    return pb, fair, mos_buy, method


def final_reco(ltp, d50, d150, d200, fair, mos_buy, momentum):
    if ltp is None:
        return "NA", "No price data from Yahoo"
    if fair is None or mos_buy is None:
        return "Hold", "Valuation inputs missing (Yahoo fundamentals incomplete for NSE)"

    # Simple, explainable rules
    if ltp <= mos_buy and momentum != "Bearish":
        return "Buy", "Price below MoS buy + momentum not bearish"
    if ltp <= fair and momentum == "Bullish":
        return "Hold", "Below fair value and trend supportive"
    if ltp > fair * 1.20 and momentum != "Bullish":
        return "Sell", "20%+ above fair while momentum is not bullish"
    return "Hold", "No strong edge vs fair value / trend"


# =========================
# UI
# =========================
with st.sidebar:
    st.header("Settings")
    mos = st.slider("Margin of Safety %", min_value=5, max_value=40, value=20, step=1)
    workers = st.slider("Speed (parallel workers)", min_value=2, max_value=12, value=4, step=1)
    st.caption("If it stalls, reduce workers (Yahoo throttling).")

uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]

    # Required columns check
    if "stock symbol" not in df.columns:
        st.error("CSV must contain a column named: stock symbol")
        st.stop()

    tickers = [_clean_nse_symbol(x) for x in df["stock symbol"].tolist()]
    tickers = [t for t in tickers if t]

    company_map = {}
    if "company name" in df.columns:
        for _, r in df.iterrows():
            company_map[_clean_nse_symbol(r["stock symbol"])] = str(r.get("company name", "")).strip()

    if st.button("🚀 RUN ANALYSIS"):
        # 1) Prices batch
        with st.spinner("Fetching price history (batch)…"):
            batch = fetch_prices_batch(sorted(set(tickers)))

        # If prices failed, stop early with a clear message
        if batch is None or (isinstance(batch, pd.DataFrame) and batch.empty):
            st.error("Yahoo price download returned empty. This is usually throttling or a temporary Yahoo outage. Try again, or reduce workers.")
            st.stop()

        # 2) Fundamentals per ticker (best effort)
        results = []
        status = st.empty()
        prog = st.progress(0)

        # Precompute DMAs first (fast)
        dmas = {}
        for t in sorted(set(tickers)):
            close = _extract_close_series(batch, t)
            ltp, d50, d150, d200 = compute_dmas(close) if close is not None else (None, None, None, None)
            dmas[t] = (ltp, d50, d150, d200)

        # Fundamentals in parallel with retries
        def _worker(t):
            f = fetch_fundamentals_one(t)
            ltp, d50, d150, d200 = dmas.get(t, (None, None, None, None))
            mom = momentum_label(ltp, d50, d150, d200)

            pb, fair, mos_buy, method = compute_valuation(ltp, f, mos)
            reco, reason = final_reco(ltp, d50, d150, d200, fair, mos_buy, mom)

            row = {
                "Company": company_map.get(t, t),
                "LTP": _as_na(round(ltp, 2) if ltp is not None else None),
                "50DMA": _as_na(round(d50, 2) if d50 is not None else None),
                "150DMA": _as_na(round(d150, 2) if d150 is not None else None),
                "200DMA": _as_na(round(d200, 2) if d200 is not None else None),
                "Sales growth % (YOY-3 years)": _as_na(round(f.get("sales_cagr"), 2) if f.get("sales_cagr") is not None else None),
                "Profit growth % (YOY 3years)": _as_na(round(f.get("profit_cagr"), 2) if f.get("profit_cagr") is not None else None),
                "ROE": _as_na(round(f.get("roe"), 2) if f.get("roe") is not None else None),
                "ROCE": _as_na(round(f.get("roce"), 2) if f.get("roce") is not None else None),
                "VAL: PB": _as_na(round(pb, 2) if pb is not None else None),
                "VAL: Fair": _as_na(round(fair, 2) if fair is not None else None),
                "VAL: MoS Buy": _as_na(round(mos_buy, 2) if mos_buy is not None else None),
                "VAL: Method": method or "NA",
                "Momentum": mom,
                "Recommendation": reco,
                "Reason": reason,
            }
            return row

        unique_tickers = sorted(set(tickers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, t): t for t in unique_tickers}
            done = 0
            for fut in as_completed(futs):
                t = futs[fut]
                done += 1
                status.info(f"Analyzing {t}… ({done}/{len(unique_tickers)})")
                prog.progress(done / len(unique_tickers))
                row = fut.result()
                results.append(row)

        status.empty()
        prog.empty()

        res_df = pd.DataFrame(results)

        # Ensure exact column order
        for c in OUTPUT_COLUMNS:
            if c not in res_df.columns:
                res_df[c] = "NA"
        res_df = res_df[OUTPUT_COLUMNS]

        st.subheader("Results")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="wealth_architect_results.csv",
            mime="text/csv",
        )

        st.caption("Note: For NSE, Yahoo often lacks full financial statements → Sales/Profit 3Y + ROCE may show NA. Prices/DMAs should still populate.")
else:
    st.info("Upload your PortfolioImportTemplate.csv to run.")
