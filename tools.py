# garchutils.py

import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple

class GARCHutils:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
    

    def download_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        url = (
            f"{self.base_url}/aggs/ticker/{ticker}/range/1/day/"
            f"{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Polygon request failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        results = data.get("results", [])
        if not results:
            print(f"Warning: No data returned for {ticker} in the date range.")
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df.rename(
            columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"},
            inplace=True
        )
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        return df[["Open","High","Low","Close","Volume"]]
    

    def compute_daily_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Close"
    ) -> pd.DataFrame:
        df = price_df.copy()
        df["Daily Return"] = df[price_col].pct_change()
        df = df.dropna(subset=["Daily Return"])
        return df

    def garch_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray
    ) -> float:
        alpha, beta, omega = params
        T = len(returns)
        h_t = np.zeros(T)
        denom = max(1e-8, 1.0 - alpha - beta)
        unc_var = omega / denom if denom > 1e-6 and omega > 0 else np.var(returns)
        h_t[0] = max(1e-8, unc_var)

        for t in range(1, T):
            h_t[t] = omega + alpha * returns[t - 1]**2 + beta * h_t[t - 1]
            if h_t[t] <= 1e-12:
                return 1e6

        ll = -0.5 * np.sum(np.log(h_t) + (returns**2 / h_t))
        return -ll

    def _constraints(self):
        cons = (
            {"type": "ineq", "fun": lambda x: x[0]},
            {"type": "ineq", "fun": lambda x: x[1]},
            {"type": "ineq", "fun": lambda x: x[2] - 1e-12},
            {"type": "ineq", "fun": lambda x: 0.999 - (x[0] + x[1])}
        )
        return cons

    def fit_garch_parameters(
        self,
        returns: np.ndarray,
        initial_guess: Tuple[float, float, float] = (0.05, 0.9, 1e-6),
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (0.0, 0.999), (0.0, 0.999), (1e-12, None)
        )
    ) -> np.ndarray:
        res = minimize(
            self.garch_log_likelihood,
            x0=np.array(initial_guess, dtype=float),
            args=(returns,),
            bounds=bounds,
            constraints=self._constraints(),
            method="SLSQP",
            options={"maxiter": 10_000, "ftol": 1e-9, "disp": False}
        )
        if not res.success:
            raise RuntimeError(f"GARCH fit failed: {res.message}")
        return res.x

    def forecast_variance_path(
        self,
        returns: np.ndarray,
        params: Tuple[float, float, float]
    ) -> np.ndarray:
        """Given fitted (alpha, beta, omega) and returns, build h_t over the sample window."""
        alpha, beta, omega = params
        T = len(returns)
        h_t = np.zeros(T)
        denom = max(1e-8, 1.0 - alpha - beta)
        unc_var = omega / denom if denom > 1e-6 and omega > 0 else np.var(returns)
        h_t[0] = max(1e-8, unc_var)
        for t in range(1, T):
            h_t[t] = omega + alpha * returns[t-1]**2 + beta * h_t[t-1]
            if h_t[t] <= 1e-12:
                h_t[t] = 1e-12
        return h_t