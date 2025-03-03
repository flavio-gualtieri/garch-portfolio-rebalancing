# tools.py

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, List, Union

class GARCHTools:
    """
    A helper class to:
      - Download stock data from Yahoo Finance
      - Compute daily returns
      - Fit GARCH(1,1) parameters via log-likelihood minimization
    """

    def __init__(self):
        """
        If you need to store any global settings or defaults,
        do so here. For now, we simply have an empty constructor.
        """
        pass

    def download_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download historical price data from Yahoo Finance for a given ticker
        and date range.

        :param ticker: The stock symbol to download
        :param start_date: Start date string (YYYY-MM-DD)
        :param end_date: End date string (YYYY-MM-DD)
        :return: DataFrame of the downloaded stock data
        """
        stock_data: pd.DataFrame = yf.download(ticker, start=start_date, end=end_date)
        return stock_data

    def compute_daily_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Close"
    ) -> pd.DataFrame:
        """
        Given a DataFrame with at least a 'Close' column, compute daily returns.

        :param price_df: DataFrame containing at least one price column
        :param price_col: Name of the price column from which to compute returns
        :return: DataFrame with a new "Daily Return" column
        """
        df: pd.DataFrame = price_df.copy()
        df["Daily Return"] = df[price_col].pct_change().dropna()
        return df

    def garch_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Negative log-likelihood for a standard GARCH(1,1) model.

        :param params: [alpha, beta, omega] array
        :param returns: array of daily returns
        :return: negative log-likelihood (because we minimize)
        """
        alpha, beta, omega = params
        T: int = len(returns)

        # Initialize the conditional variance array
        h_t: np.ndarray = np.zeros(T)
        h_t[0] = np.var(returns)  # or some other initial variance guess

        # GARCH(1,1) variance recursion
        for t in range(1, T):
            h_t[t] = omega + alpha * returns[t - 1]**2 + beta * h_t[t - 1]

        # Gaussian log-likelihood sum
        ll: float = -0.5 * np.sum(np.log(h_t) + (returns**2 / h_t))
        return -ll  # Return negative for 'minimize'

    def fit_garch_parameters(
        self,
        returns: np.ndarray,
        initial_guess: Tuple[float, float, float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ) -> np.ndarray:
        """
        Fits GARCH(1,1) parameters alpha, beta, omega by
        minimizing negative log-likelihood.

        :param returns: array of daily returns
        :param initial_guess: (alpha_guess, beta_guess, omega_guess)
        :param bounds: bounds for alpha, beta, omega, e.g. ((1e-6, 1), (1e-6,1), (1e-6, None))
        :return: array [alpha_opt, beta_opt, omega_opt]
        """
        result = minimize(
            self.garch_log_likelihood,
            initial_guess,
            args=(returns,),
            bounds=bounds
        )
        return result.x