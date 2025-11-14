import streamlit as st
import pandas as pd
import numpy as np

from utils import Utils
from garch import GARCH
from variance_forecaster import VarianceForecaster


st.set_page_config(page_title="GARCH Variance Forecaster", layout="centered")

st.title("GARCH Variance Forecaster")
st.caption("Estimate and evaluate GARCH(1,1) variances across multiple assets.")

st.markdown(
    "This app downloads daily price data from Polygon.io, "
    "fits a GARCH(1,1) model per stock, and evaluates out-of-sample performance."
)

# --- Sidebar inputs ---
st.sidebar.header("Inputs")

api_key = st.sidebar.text_input("Polygon API key", type="password")

tickers_str = st.sidebar.text_input(
    "Tickers (comma separated)",
    value="AAPL,MSFT",
)

start_date = st.sidebar.date_input("Start date")
end_date = st.sidebar.date_input("End date")

train_end_date = st.sidebar.date_input(
    "Training end date (inclusive)",
    help="Data up to and including this date is used for fitting. Later dates are test."
)

price_col = st.sidebar.selectbox(
    "Price column",
    options=["Close", "Open", "High", "Low"],
    index=0,
)

run_button = st.sidebar.button("Run backtest")

# --- Main logic ---
if run_button:
    if not api_key:
        st.error("Please enter your Polygon API key.")
    else:
        # Basic validation
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        elif train_end_date >= end_date:
            st.error("Training end date must be before the overall end date.")
        else:
            tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            if not tickers:
                st.error("Please enter at least one valid ticker.")
            else:
                start_date_str = start_date.isoformat()
                end_date_str = end_date.isoformat()
                train_end_date_str = train_end_date.isoformat()

                try:
                    utils = Utils(api_key=api_key)
                    garch_model = GARCH()
                    forecaster = VarianceForecaster(
                        garch_model=garch_model,
                        data_utils=utils,
                        tickers=tickers,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        price_col=price_col,
                    )

                    with st.spinner("Downloading data and fitting GARCH models..."):
                        # Fit models
                        forecaster.fit_all(train_end_date=train_end_date_str)
                        H = forecaster.get_variance_matrix()

                    st.success("Fitting complete.")

                    # --- Evaluation on test period ---
                    rows = []
                    for tkr, res in forecaster.fit_results.items():
                        returns_series = res.returns
                        variance_series = pd.Series(res.h_t, index=res.returns.index)

                        test_mask = returns_series.index > train_end_date_str
                        test_returns = returns_series[test_mask]
                        test_variances = variance_series[test_mask]

                        if test_returns.empty:
                            rows.append(
                                {
                                    "Ticker": tkr,
                                    "Test start": None,
                                    "Test end": None,
                                    "Test observations": 0,
                                    "MSE (r^2 vs h_t)": np.nan,
                                }
                            )
                            continue

                        mse = np.mean((test_returns.values ** 2 - test_variances.values) ** 2)

                        rows.append(
                            {
                                "Ticker": tkr,
                                "Test start": test_returns.index[0].date(),
                                "Test end": test_returns.index[-1].date(),
                                "Test observations": len(test_returns),
                                "MSE (r^2 vs h_t)": mse,
                            }
                        )

                    results_df = pd.DataFrame(rows).set_index("Ticker")

                    st.subheader("Out-of-sample performance (test period)")
                    st.dataframe(results_df)

                    # Optional: simple plot for first ticker
                    first_tkr = tickers[0]
                    if first_tkr in forecaster.fit_results:
                        st.subheader(f"Example: {first_tkr} test period fit")

                        res = forecaster.fit_results[first_tkr]
                        returns_series = res.returns
                        variance_series = pd.Series(res.h_t, index=res.returns.index)

                        test_mask = returns_series.index > train_end_date_str
                        plot_df = pd.DataFrame(
                            {
                                "r^2": returns_series[test_mask] ** 2,
                                "h_t": variance_series[test_mask],
                            }
                        )
                        st.line_chart(plot_df)

                except Exception as e:
                    st.error(f"Error during computation: {e}")
