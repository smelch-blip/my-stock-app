import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# UI + CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Wealth Architect Pro (NSE)")

st.markdown(
    """
<style>
.stApp { background-color: #ffffff; color: #000000; }
section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
.stButton>button { background-color: #1d4ed8 !important; color: white !important; width: 100%; height: 3em; border-radius: 8px; }
.small-note { color:#6b7280; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🏛️ Wealth Architect Pro")
st.caption(
    "NSE-only portfolio audit using Yahoo Finance (free). Prices are usually reliable; NSE fundamentals can be incomplete, so the app shows NA instead of guessing."
)

# =========================
# EXACT COLUMN NAMES (your requirement)
# =========================
COL_COMPANY = "Company"
COL_LTP = "LTP"
COL_50 = "50DMA"
COL_150 = "150DMA"
COL_200 = "200DMA"

COL_SALES_3Y = "Sales growth % (YOY-3 years)"
COL_PROFIT_3Y = "Profit growth % (YOY 3years)"
COL_ROE = "ROE"
COL_ROCE = "ROCE"

COL_VAL_PB = "VAL: PB"
COL_VAL_FAIR = "VAL: Fair"
COL_VAL_MOS = "VAL: MoS Buy"
COL_VAL_METHOD = "VAL: Method"

COL_MOM = "Momentum"
COL_REC = "Recommendation"
COL_REASON = "Reason"

GROUP_TECH = "Technicals"
GROUP_FUND = "Fundamentals"
GROUP_VAL = "Valuation"
GROUP_FINAL = "Final"
GROUP_LEFT = ""

DISPLAY_COLUMNS = pd.MultiIndex.from_tuples(
    [
        (GROUP_LEFT, COL_COMPANY),
        (GROUP_LEFT, COL_LTP),

        (GROUP_TECH, COL_50),
        (GROUP_TECH, COL_150),
        (GROUP_TECH, COL_200),

        (GROUP_FUND, COL_SALES_3Y),
        (GROUP_FUND, COL_PROFIT_3Y),
        (GROUP_FUND, COL_ROE),
        (GROUP_FUND, COL_ROCE),

        (GROUP_VAL, COL_VAL_PB),
        (GROUP_VAL, COL_VAL_FAIR),
        (GROUP_VAL, COL_VAL_MOS),
        (GROUP_VAL, COL_VAL_METHOD),

        (GROUP_FINAL, COL_MOM),
        (GROUP_FINAL, COL_REC),
        (GROUP_FINAL, COL_REASON),
    ]
)

CSV_COLUMNS = [
    COL_COMPANY, COL_LTP,
    COL_50, COL_150, COL_200,
    COL_SALES_3Y, COL_PROFIT_3Y, COL_ROE, COL_ROCE,
    COL_VAL_PB, COL_VAL_FAIR, COL_VAL_MOS, COL_VAL_METHOD,
    COL_MOM, COL_REC, COL_REASON
]

# =========================
# SECTOR/VALUATION CONFIG
# =========================
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
    if not sym:
        return ""
    if "." not in sym:
        sym = sym + ".NS"
    if not sym.endswith((".NS", ".BO")):
        sym = sym + ".NS"
    return sym


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_batch(tickers: list[str]):
    # IMPORTANT: Do NOT pass custom session (new yfinance requires curl_cffi internally)
    data = yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    return data


def _extract_close_series(batch_df: pd.DataFrame, ticker: str) -> pd.Series | None:
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            if (ticker, "Close") in batch_df.columns:
                s = batch_df[(ticker, "Close")].dropna()
                return s if len(s) else None
            if ticker in batch_df.columns.get_level_values(0):
                s = batch_df[ticker]["Close"].dropna()
                return s if len(s) else None
        else:
            if "Close" in batch_df.columns:
                s = batch_df["Close"].dropna()
                return s if len(s) else None
        return None
    except Exception:
        return None


def fetch_fundamentals_one(ticker: str):
    """
    Best-effort fundamentals for NSE from Yahoo:
    - ROE, PB from info if present
    - 3Y sales/profit growth and ROCE only if annual statements are available (often missing for NSE).
    """
    # IMPORTANT: Do NOT pass custom session; let yfinance manage it.
    t = yf.Ticker(ticker)

    info = _retry(lambda: t.info, tries=2) or {}
    sector = info.get("sector")
    industry = info.get("industry")

    pb = _safe_float(info.get("priceToBook"))
    roe = _safe_float(info.get("returnOnEquity"))
    roe_pct = (roe * 100.0) if roe is not None else None

    sales_cagr = None
    profit_cagr = None
    roce_pct = None

    # statements (often missing for NSE)
    try:
        def _get_income():
            inc = getattr(t, "get_income_stmt", None)
            if callable(inc):
                return inc(freq="yearly")
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

        if isinstance(income, pd.DataFrame) and not income.empty:
            cols_sorted = sorted(list(income.columns))
            income2 = income[cols_sorted]

            revenue_row_candidates = ["Total Revenue", "Operating Revenue", "Revenue"]
            net_income_row_candidates = ["Net Income", "NetIncome", "Profit After Tax", "Net Profit"]
            ebit_row_candidates = ["EBIT", "Operating Income", "OperatingIncome"]

            def _pick_row(df, candidates):
                for r in candidates:
                    if r in df.index:
                        return df.loc[r]
                return None

            rev = _pick_row(income2, revenue_row_candidates)
            ni = _pick_row(income2, net_income_row_candidates)
            ebit = _pick_row(income2, ebit_row_candidates)

            if rev is not None and len(rev.dropna()) >= 4:
                r = rev.dropna()
                sales_cagr = _cagr(r.iloc[-4], r.iloc[-1], 3.0)

            if ni is not None and len(ni.dropna()) >= 4:
                p = ni.dropna()
                profit_cagr = _cagr(p.iloc[-4], p.iloc[-1], 3.0)

            # ROCE calc: EBIT / (TotalAssets - CurrentLiabilities)
            if isinstance(balance, pd.DataFrame) and not balance.empty and ebit is not None:
                bcols = sorted(list(balance.columns))
                bs2 = balance[bcols]

                ta = None
                for rname in ["Total Assets", "TotalAssets"]:
                    if rname in bs2.index:
                        ta = bs2.loc[rname]
                        break

                cl = None
                for rname in ["Total Current Liabilities", "Current Liabilities", "TotalCurrentLiabilities"]:
                    if rname in bs2.index:
                        cl = bs2.loc[rname]
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


def compute_valuation(fund: dict, mos_pct: float):
    pb = fund.get("pb")
    bucket = fund.get("bucket", "Default")
    cfg = SECTOR_BANDS.get(bucket, SECTOR_BANDS["Default"])

    fair = None
    method = None

    if cfg["method"] == "P/B":
        bv = fund.get("book_value")
        roe = fund.get("roe")
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
        eps = fund.get("eps")
        if eps is not None and eps > 0:
            norm_eps = eps * 0.80
            pe = cfg["pe_mid"]
            fair = norm_eps * pe
            method = f"CYCLICAL P/E | bucket={bucket}"
        else:
            method = f"CYCLICAL P/E | bucket={bucket} (no EPS)"

    else:
        eps = fund.get("eps")
        if eps is not None and eps > 0:
            pe = cfg["pe_mid"]

            roce = fund.get("roce")
            pg = fund.get("profit_cagr")
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

    if ltp <= mos_buy and momentum != "Bearish":
        return "Buy", "Price below MoS Buy + momentum not bearish"
    if ltp <= fair and momentum == "Bullish":
        return "Hold", "Below Fair value + bullish trend"
    if ltp > fair * 1.20 and momentum != "Bullish":
        return "Sell", "20%+ above Fair while momentum not bullish"
    return "Hold", "No strong edge vs Fair / trend"


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Settings")
    mos = st.slider("Margin of Safety %", 5, 40, 20, 1)
    workers = st.slider("Speed (parallel workers)", 2, 12, 4, 1)
    st.markdown(
        '<div class="small-note">If it stalls or shows NA for prices, reduce workers (Yahoo throttling).</div>',
        unsafe_allow_html=True,
    )

uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]

    if "stock symbol" not in df.columns:
        st.error("CSV must contain a column named: stock symbol")
        st.stop()

    tickers = [_clean_nse_symbol(x) for x in df["stock symbol"].tolist()]
    tickers = [t for t in tickers if t]
    unique_tickers = sorted(set(tickers))

    company_map = {}
    if "company name" in df.columns:
        for _, r in df.iterrows():
            company_map[_clean_nse_symbol(r["stock symbol"])] = str(r.get("company name", "")).strip()

    if st.button("🚀 RUN ANALYSIS"):
        st.info(f"Analyzing {len(unique_tickers)} tickers...")

        # 1) Batch price fetch
        with st.spinner("Fetching price history (batch)…"):
            batch = fetch_prices_batch(unique_tickers)

        if batch is None or (isinstance(batch, pd.DataFrame) and batch.empty):
            st.error("Yahoo price download returned empty (throttling/outage). Try again, or reduce workers to 2–3.")
            st.stop()

        # 2) Precompute DMAs
        dmas = {}
        for t in unique_tickers:
            close = _extract_close_series(batch, t)
            ltp, d50, d150, d200 = compute_dmas(close) if close is not None else (None, None, None, None)
            dmas[t] = (ltp, d50, d150, d200)

        # 3) Fundamentals (parallel best-effort)
        results_rows = []
        status = st.empty()
        prog = st.progress(0)

        def _worker(t):
            fund = fetch_fundamentals_one(t)

            ltp, d50, d150, d200 = dmas.get(t, (None, None, None, None))
            mom = momentum_label(ltp, d50, d150, d200)

            val_pb, val_fair, val_mos, val_method = compute_valuation(fund, mos)
            reco, reason = final_reco(ltp, d50, d150, d200, val_fair, val_mos, mom)

            row = {
                COL_COMPANY: company_map.get(t, t),
                COL_LTP: "NA" if ltp is None else round(ltp, 2),

                COL_50: "NA" if d50 is None else round(d50, 2),
                COL_150: "NA" if d150 is None else round(d150, 2),
                COL_200: "NA" if d200 is None else round(d200, 2),

                COL_SALES_3Y: "NA" if fund.get("sales_cagr") is None else round(fund["sales_cagr"], 2),
                COL_PROFIT_3Y: "NA" if fund.get("profit_cagr") is None else round(fund["profit_cagr"], 2),
                COL_ROE: "NA" if fund.get("roe") is None else round(fund["roe"], 2),
                COL_ROCE: "NA" if fund.get("roce") is None else round(fund["roce"], 2),

                COL_VAL_PB: "NA" if val_pb is None else round(val_pb, 2),
                COL_VAL_FAIR: "NA" if val_fair is None else round(val_fair, 2),
                COL_VAL_MOS: "NA" if val_mos is None else round(val_mos, 2),
                COL_VAL_METHOD: val_method or "NA",

                COL_MOM: mom,
                COL_REC: reco,
                COL_REASON: reason,
            }
            return row

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, t): t for t in unique_tickers}
            done = 0
            for fut in as_completed(futs):
                t = futs[fut]
                done += 1
                status.info(f"Processing {t} ({done}/{len(unique_tickers)})")
                prog.progress(done / len(unique_tickers))
                results_rows.append(fut.result())

        status.empty()
        prog.empty()

        res = pd.DataFrame(results_rows)
        for c in CSV_COLUMNS:
            if c not in res.columns:
                res[c] = "NA"
        res = res[CSV_COLUMNS]

        display = pd.DataFrame(columns=DISPLAY_COLUMNS)
        for (grp, col) in DISPLAY_COLUMNS:
            display[(grp, col)] = res[col]

        st.subheader("Results")
        st.dataframe(display, use_container_width=True, hide_index=True)

        csv_bytes = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="wealth_architect_results.csv",
            mime="text/csv",
        )

        st.caption("Note: For NSE, Yahoo often lacks full statements → Sales/Profit 3Y + ROCE may show NA. Prices/DMAs should still populate.")
else:
    st.info("Upload your PortfolioImportTemplate.csv to run.")
