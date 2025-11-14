# GARCH Variance Forecaster

This project is a small Streamlit app that:

- Downloads daily stock price data from [Polygon.io](https://polygon.io/)
- Computes daily returns
- Fits a univariate GARCH(1,1) model per ticker
- Forecasts the conditional variance path for each asset
- Evaluates out-of-sample performance by comparing realized squared returns to forecast variances

The core logic lives in three modules:

- `utils.py` — data access + basic return calculations
- `garch.py` — GARCH(1,1) log-likelihood, parameter fitting, and variance path forecasting
- `variance_forecaster.py` — multi-asset orchestration (download, align returns, fit per ticker, build variance matrix)

The web interface is implemented in `app.py` using Streamlit.

---

## Project Structure

```text
.
├── app.py
├── garch.py
├── utils.py
├── variance_forecaster.py
└── requirements.txt