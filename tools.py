# tools.py

import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple

class GARCHTools:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
    

    def download_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        # Format dates to ensure proper endpoint formatting
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        
        # Build the URL for daily aggregates
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
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        # Convert epoch milliseconds in 't' to datetime
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        # Rename columns to standard OHLCV names
        df.rename(
            columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume"
            },
            inplace=True
        )
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        
        # Return only the standard columns
        return df[["Open", "High", "Low", "Close", "Volume"]]
    

    def compute_daily_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Close"
    ) -> pd.DataFrame:
        df = price_df.copy()
        df["Daily Return"] = df[price_col].pct_change().dropna()
        return df


    def garch_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray
    ) -> float:
        alpha, beta, omega = params
        T = len(returns)
        
        # Initialize conditional variances
        h_t = np.zeros(T)
        h_t[0] = np.var(returns)
        
        for t in range(1, T):
            h_t[t] = omega + alpha * returns[t - 1]**2 + beta * h_t[t - 1]
        
        ll = -0.5 * np.sum(np.log(h_t) + (returns**2 / h_t))
        return -ll  # Negative for minimization
    

    def fit_garch_parameters(
        self,
        returns: np.ndarray,
        initial_guess: Tuple[float, float, float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ) -> np.ndarray:
        result = minimize(
            self.garch_log_likelihood,
            initial_guess,
            args=(returns,),
            bounds=bounds
        )
        return result.x