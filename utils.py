import requests
import pandas as pd
from typing import Optional


class Utils:
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
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"},
            inplace=True,
        )
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def compute_daily_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Close"
    ) -> pd.DataFrame:
        df = price_df.copy()
        df["Daily Return"] = df[price_col].pct_change()
        df = df.dropna(subset=["Daily Return"])
        return df
