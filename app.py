import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")
st.title("📈 Advanced Portfolio Analyzer")

uploaded_file = st.file_uploader("Upload your Portfolio (Excel or CSV)", type=["csv", "xlsx"])

def calculate_growth(ticker):
    try:
        # Fetching financials for earnings growth
        financials = ticker.earnings_dates
        if financials is None or financials.empty:
            return "N/A", "N/A", "N/A", "N/A"
        
        # This is a simplified growth calculation based on historical price trends 
        # as a proxy for long-term compounding growth
        hist = ticker.history(period="10y")
        if len(hist) < 252: return "N/A", "N/A", "N/A", "N/A"
        
        def get_cagr(years):
            days = int(years * 252)
            if len(hist) < days: return "N/A"
            start_price = hist['Close'].iloc[-days]
            end_price = hist['Close'].iloc[-1]
            cagr = ((end_price / start_price) ** (1/years) - 1) * 100
            return f"{round(cagr, 1)}%"

        return get_cagr(3), get_cagr(5), get_cagr(7), get_cagr(10)
    except:
        return "N/A", "N/A", "N/A", "N/A"

def analyze_stock(symbol):
    if pd.isna(symbol) or str(symbol).strip() == "":
        return None
    
    try:
        t = yf.Ticker(str(symbol).strip())
        hist = t.history(period="1y")
        if hist.empty: return None
        
        ltp = hist['Close'].iloc[-1]
        d50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        d150 = hist['Close'].rolling(window=150).mean().iloc[-1]
        d200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Recommendation Logic
        rec = "Hold"
        if ltp > d50 > d200: rec = "Strong Buy"
        elif ltp < d200: rec = "Sell/Caution"
        
        g3, g5, g7, g10 = calculate_growth(t)
        
        return {
            "LTP": round(ltp, 2),
            "50 DMA": round(d50, 2),
            "150 DMA": round(d150, 2),
            "200 DMA": round(d200, 2),
            "3Y Growth": g3,
            "5Y Growth": g5,
            "7Y Growth": g7,
            "10Y Growth": g10,
            "Recommendation": rec
        }
    except:
        return None

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    if st.button("Run Full Analysis"):
        with st.spinner('Calculating technicals and long-term growth...'):
            final_rows = []
            for _, row in df.iterrows():
                sym = row.get('stock symbol')
                data = analyze_stock(sym)
                
                if data:
                    combined = {
                        "Share Name": row.get('company name', sym),
                        "Qty": row.get('qnty', 0),
                        **data
                    }
                    final_rows.append(combined)
            
            if final_rows:
                st.table(pd.DataFrame(final_rows))
            else:
                st.error("Could not find any valid stock data. Check your 'stock symbol' column.")
