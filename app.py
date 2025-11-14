import streamlit as st
import pandas as pd
import numpy as np

from utils import Utils
from garch import GARCH
from variance_forecaster import VarianceForecaster


st.set_page_config(page_title="GARCH Variance Forecaster", layout="centered")

st.title("GARCH Variance Forecaster")
st.caption("Estimate and evaluate GARCH(1,1) conditional variances across multiple assets.")

st.markdown(
    "This app downloads daily price data from Polygon.io, "
    "fits a GARCH(1,1) model per stock on a training window, and then "
    "evaluates how well the forecast variances match out-of-sample realized volatility."
)

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
    help="Data up to and including this date is used to estimate GARCH parameters. Later dates are used only for evaluation."
)

price_col = st.sidebar.selectbox(
    "Price column",
    options=["Close", "Open", "High", "Low"],
    index=0,
)

run_button = st.sidebar.button("Run evaluation")

if run_button:
    if not api_key:
        st.error("Please enter your Polygon API key.")
    else:
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
                        forecaster.fit_all(train_end_date=train_end_date_str)
                        H = forecaster.get_variance_matrix()

                    st.success("Fitting complete.")

                    st.subheader("Out-of-sample performance (test period)")
                    st.markdown(
                        """
We evaluate the model **only on data after the training end date**:

- For each date in the test period, the GARCH model provides a **forecast conditional variance** \\(h_t\\).
- The **squared return** \\(r_t^2\\) is used as a proxy for realized variance.
- The reported metric is the mean squared error of \\(r_t^2\\) versus \\(h_t\\):

\\[
\\text{MSE} = \\frac{1}{T_{\\text{test}}} \\sum_t (r_t^2 - h_t)^2
\\]

Lower values indicate that the forecast variance tracks realized volatility more closely.
"""
                    )

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
                                    "MSE: realized r² vs forecast hₜ": np.nan,
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
                                "MSE: realized r² vs forecast hₜ": mse,
                            }
                        )

                    results_df = pd.DataFrame(rows).set_index("Ticker")
                    st.dataframe(results_df)

                    first_tkr = tickers[0]
                    if first_tkr in forecaster.fit_results:
                        st.subheader(f"Example: {first_tkr} test period fit")
                        st.caption(
                            "Squared returns (r²) are used as a proxy for realized variance "
                            "and are compared to the GARCH forecast variance hₜ over the test period."
                        )

                        res = forecaster.fit_results[first_tkr]
                        returns_series = res.returns
                        variance_series = pd.Series(res.h_t, index=res.returns.index)

                        test_mask = returns_series.index > train_end_date_str
                        plot_df = pd.DataFrame(
                            {
                                "r² (realized)": returns_series[test_mask] ** 2,
                                "hₜ (forecast)": variance_series[test_mask],
                            }
                        )
                        st.line_chart(plot_df)

                except Exception as e:
                    st.error(f"Error during computation: {e}")