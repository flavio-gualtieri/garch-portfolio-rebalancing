import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from garch_testing import Testing  # Removed redundant MultiStockPortfolioRebalancing import

# Load S&P 500 stock symbols
@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_tickers = tables[0]["Symbol"].tolist()
    return sp500_tickers

sp500_tickers = get_sp500_tickers()

# Streamlit UI
st.title("Portfolio Rebalancing Model")

# User Inputs
selected_stocks = st.multiselect("Select Stocks", sp500_tickers, default=["AAPL"])
initial_wealth = st.number_input("Initial Wealth ($)", min_value=1000, value=100000, step=1000)

if st.button("Run Backtest"):
    if not selected_stocks:
        st.warning("⚠️ Please select at least one stock.")
    else:
        st.info("Fetching data and running backtest... ⏳")

        # Initialize backtest with user input
        tester = Testing(
            stocks=selected_stocks,
            start_date="2023-08-01",
            end_date="2024-02-01",
            wealth=initial_wealth,
            risk_av=-5,
            T=126  # 6 months of trading days
        )

        # Fetch stock data
        tester.get_data()

        # Set GARCH parameters
        initial_params = {
            "alpha": 0.05,
            "beta": 0.9,
            "omega": 0.00001,
            "lambda": 0.1,
            "theta": [0.5] * len(selected_stocks)
        }
        tester.set_params(initial_params)

        # Run backtest
        results = tester.run_test()

        # Ensure all selected stocks have valid data
        valid_stocks = [stock for stock in selected_stocks if stock in results]
        if not valid_stocks:
            st.error("❌ No valid data retrieved for selected stocks. Try different stocks or dates.")
            st.stop()

        # --- PLOT 1: Stock Performances ---
        st.subheader("📈 Stock Performance Over Past 6 Months")
        plt.figure(figsize=(12, 6))
        for stock in valid_stocks:
            stock_data = results[stock]
            plt.plot(stock_data.index, stock_data["Close"], label=stock)
        plt.xlabel("Date")
        plt.ylabel("Stock Price ($)")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # --- PLOT 2: Wealth Evolution ---
        st.subheader("💰 Wealth Evolution Over Time")
        plt.figure(figsize=(12, 6))
        for stock in valid_stocks:
            stock_data = results[stock]
            plt.plot(stock_data.index, stock_data["Actual Wealth"], label=f"Wealth ({stock})")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        st.success("✅ Backtest Completed!")
