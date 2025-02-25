import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from garch_testing import Testing

# Load S&P 500 stock symbols
@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_tickers = tables[0]["Symbol"].tolist()
    return sp500_tickers

# Streamlit app
st.title("GARM Portfolio Rebalancing Model")

# Sidebar inputs
st.sidebar.header("Model Inputs")
sp500_tickers = get_sp500_tickers()
selected_stock = st.sidebar.selectbox("Select a Stock", sp500_tickers)
initial_wealth = st.sidebar.number_input("Initial Wealth ($)", min_value=1000, value=10000, step=1000)
risk_aversion = st.sidebar.slider("Risk Aversion (gamma)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

# Date range for backtesting
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(months=6)

# Button to trigger backtest
if st.sidebar.button("Run Backtest"):
    with st.spinner("Fetching data and running model..."):
        # Initialize testing instance
        test = Testing(
            stock=selected_stock,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            wealth=initial_wealth,
            risk_av=risk_aversion,
            T=126  # Approx. 6 months of trading days
        )
        
        test.get_data()
        test.set_params({"alpha": 0.1, "beta": 0.85, "omega": 0.05})
        test.optimize_params()
        results = test.run_test()
        
        # Store results in session state
        st.session_state["results"] = results

# Display results if available
if "results" in st.session_state:
    results = st.session_state["results"]
    
    st.subheader("Backtest Results")
    st.write("Debugging: Available columns in results DataFrame:", results.columns)
    
    if "Actual Wealth" in results.columns:
        st.write(f"Initial Wealth: ${results['Actual Wealth'].iloc[0]:,.2f}")
        st.write(f"Final Wealth: ${results['Actual Wealth'].iloc[-1]:,.2f}")
        st.write(f"Net Gain/Loss: ${results['Actual Wealth'].iloc[-1] - results['Actual Wealth'].iloc[0]:,.2f}")
        
        st.subheader("Portfolio Performance")
        st.line_chart(results[["Actual Wealth"]])
    else:
        st.error("Column 'Actual Wealth' not found. Check the GARCH model output.")
    
    if "Portfolio Weights" in results.columns:
        st.subheader("Portfolio Weights Over Time")
        st.line_chart(results[["Portfolio Weights"]])
    else:
        st.error("Column 'Portfolio Weights' not found. Check the GARCH model output.")
