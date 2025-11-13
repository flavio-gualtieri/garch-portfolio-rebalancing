import streamlit as st
import matplotlib.pyplot as plt
from old.garch_testing import Testing

def main():
    st.title("Portfolio Rebalancing Results")
    st.write("This app runs a portfolio rebalancing simulation using Polygon.io data.")

    # Instantiate the tester with your Polygon API key.
    tester = Testing(
        stocks=["META"],
        start_date="2023-09-01",
        end_date="2024-01-01",
        wealth=100,
        risk_av=-0.5,
        T=252,
        api_key="2pMQ6JJ13fOM26Ek5UMIwYjEQVwo1JWi"
    )
    
    st.write("Fetching data from Polygon.io...")
    tester.get_data()
    
    # Set dummy multi-asset parameters (for a single stock, theta is a list with one element)
    multi_params = {
        "alpha": 0.05,
        "beta": 0.9,
        "omega": 1e-5,
        "lambda": 2.0,
        "theta": [100.0]
    }
    tester.set_params(multi_params)
    
    st.write("Running portfolio rebalancing simulation...")
    results = tester.run_test()
    
    # Loop over each stock (here just AAPL) and display the plots
    for stock in tester.stocks:
        df_result = results[stock]
        
        st.subheader(f"{stock} Portfolio Weights")
        fig1, ax1 = plt.subplots()
        ax1.plot(df_result.index, df_result["Portfolio Weights"])
        ax1.set_title(f"{stock} Portfolio Weights")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Weight")
        st.pyplot(fig1)
        
        st.subheader(f"{stock} Actual Wealth")
        fig2, ax2 = plt.subplots()
        ax2.plot(df_result.index, df_result["Actual Wealth"], color="green")
        ax2.set_title(f"{stock} Actual Wealth")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Wealth")
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
