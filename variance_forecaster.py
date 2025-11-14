from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from garch import GARCH
from utils import Utils


@dataclass
class FitResult:
    params: Tuple[float, float, float]
    h_t: np.ndarray
    returns: pd.Series


class VarianceForecaster:
    def __init__(
        self,
        garch_model: GARCH,
        data_utils: Utils,
        tickers: List[str],
        start_date: str,
        end_date: str,
        price_col: str = "Close",
    ):
        self.garch_model = garch_model
        self.data_utils = data_utils
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.price_col = price_col

        self.price_panels: Dict[str, pd.DataFrame] = {}
        self.returns_df: pd.DataFrame = pd.DataFrame()
        self.fit_results: Dict[str, FitResult] = {}

    def download_all(self) -> None:
        frames = {}
        for tkr in self.tickers:
            df = self.data_utils.download_stock_data(
                tkr,
                self.start_date,
                self.end_date
            )
            if df.empty:
                continue
            frames[tkr] = df[[self.price_col]].rename(columns={self.price_col: tkr})
        if not frames:
            raise ValueError("No price data downloaded for any ticker.")
        self.price_panels = frames

    def build_aligned_returns(self) -> None:
        if not self.price_panels:
            self.download_all()

        prices = None
        for tkr, df in self.price_panels.items():
            prices = df if prices is None else prices.join(df, how="inner")
        rets = prices.pct_change().dropna(how="any")
        self.returns_df = rets

    def fit_all(
        self,
        initial_guess: Tuple[float, float, float] = (0.05, 0.9, 1e-6),
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (0.0, 0.999),
            (0.0, 0.999),
            (1e-12, None),
        ),
        train_end_date: Optional[str] = None,
    ) -> None:
        if self.returns_df.empty:
            self.build_aligned_returns()

        for tkr in self.returns_df.columns:
            returns_series = self.returns_df[tkr].astype(float)

            if train_end_date is not None:
                train_mask = returns_series.index <= train_end_date
                if not train_mask.any():
                    raise ValueError(
                        f"No training data for {tkr} with train_end_date={train_end_date}."
                    )
                train_returns = returns_series[train_mask].values
            else:
                train_returns = returns_series.values

            full_returns = returns_series.values

            params = self.garch_model.fit_garch_parameters(
                returns=train_returns,
                initial_guess=initial_guess,
                bounds=bounds,
            )

            h_t = self.garch_model.forecast_variance_path(
                returns=full_returns,
                params=params,
            )

            self.fit_results[tkr] = FitResult(
                params=tuple(params),
                h_t=h_t,
                returns=returns_series,
            )

    def get_variance_matrix(self) -> pd.DataFrame:
        if not self.fit_results:
            self.fit_all()

        h_map = {}
        for tkr, res in self.fit_results.items():
            h_map[tkr] = pd.Series(res.h_t, index=res.returns.index, name=tkr)

        H = pd.concat(h_map.values(), axis=1)
        H.index.name = "Date"
        return H

    def summarize_params(self) -> pd.DataFrame:
        rows = []
        for tkr, res in self.fit_results.items():
            alpha, beta, omega = res.params
            rows.append(
                {
                    "Ticker": tkr,
                    "alpha": alpha,
                    "beta": beta,
                    "omega": omega,
                    "alpha+beta": alpha + beta,
                }
            )
        return pd.DataFrame(rows).set_index("Ticker").sort_index()