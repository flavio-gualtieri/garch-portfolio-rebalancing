# garch_testing.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from garch_model import MultiStockPortfolioRebalancing
from tools import GARCHTools  # Now using our updated Polygon-based helper

plt.ion()

class Testing:
    """
    A class to orchestrate:
      1) Data retrieval for multiple tickers via Polygon.io
      2) (Optionally) GARCH(1,1) parameter fitting for each ticker
      3) Setting up and running MultiStockPortfolioRebalancing
    """

    def __init__(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        wealth: float,
        risk_av: float,
        T: int,
        api_key: str
    ):
        self.stocks = stocks
        self.start = start_date
        self.end = end_date
        self.wealth = wealth
        self.gamma = risk_av
        self.T = T

        # Container for fetched data
        self.data: Dict[str, Any] = {}
        # For multi-asset returns: shape [Time, NumAssets]
        self.returns_matrix: np.ndarray | None = None

        # The final dictionary of GARCH parameters (or a single dict for uniform params)
        self.params = None

        # Instantiate your GARCHTools helper with the Polygon.io API key
        self.tools = GARCHTools(api_key=api_key)

    def get_data(self):
        """
        Download stock data for each symbol from Polygon.io, compute daily returns,
        and build a returns matrix across all assets.
        """
        data = {}
        all_returns = []

        for stock in self.stocks:
            print(f"Fetching data for {stock}...")

            # 1) Download data using Polygon
            stock_df = self.tools.download_stock_data(stock, self.start, self.end)
            if stock_df.empty or "Close" not in stock_df:
                print(f"⚠️ Warning: No data found for {stock}. Skipping...")
                continue

            # 2) Compute daily returns
            stock_df = self.tools.compute_daily_returns(stock_df)
            stock_df.dropna(subset=["Daily Return"], inplace=True)

            if stock_df.empty:
                print(f"⚠️ Warning: No valid returns for {stock}. Skipping...")
                continue

            data[stock] = stock_df
            all_returns.append(stock_df["Daily Return"].values)

            print(f"✅ Data retrieved for {stock} with shape {stock_df.shape}")

        if not data:
            raise ValueError("❌ No valid stock data available. Try different stocks or a different date range.")

        self.data = data

        # Align lengths across tickers if needed (trim to the shortest series)
        min_len = min(len(r) for r in all_returns)
        all_returns_trimmed = [r[-min_len:] for r in all_returns]
        self.returns_matrix = np.column_stack(all_returns_trimmed)

        print(f"Final returns matrix shape: {self.returns_matrix.shape}")
        return self.data

    def set_params(self, params: Dict[str, float]):
        """
        Set or update the dictionary of GARCH parameters for the multi-asset model.
        Example: params = {"alpha": 0.05, "beta": 0.9, "omega": 1e-5, "lambda": 2.0, "theta": [100.0, 120.0]}
        """
        self.params = params

    def optimize_params_single_asset(self, stock: str, initial_guess=(0.01, 0.9, 0.01)):
        """
        Example method to optimize GARCH parameters for a single ticker,
        using GARCHTools. You might or might not use this for multi-asset settings.
        """
        if stock not in self.data or self.data[stock].empty:
            print(f"⚠️ No data for {stock}. Skipping GARCH optimization.")
            return None

        returns = self.data[stock]["Daily Return"].dropna().values
        bounds = ((1e-6, 1), (1e-6, 1), (1e-6, None))
        alpha_opt, beta_opt, omega_opt = self.tools.fit_garch_parameters(returns, initial_guess, bounds)
        return alpha_opt, beta_opt, omega_opt

    def optimize_params_multi_asset(self):
        """
        Placeholder: For a joint multi-asset GARCH optimization (e.g., DCC-GARCH), implement here.
        """
        pass

    def run_test(self):
        """
        Execute the multi-asset portfolio rebalancing simulation.
        Returns a dict of DataFrames with results appended to each stock.
        """
        if self.returns_matrix is None or self.returns_matrix.size == 0:
            raise ValueError("❌ No valid stock data or returns matrix. Please run get_data() first.")

        num_assets = len(self.stocks)
        T = self.returns_matrix.shape[0]
        if not self.params:
            raise ValueError("❌ You must set self.params before running the test.")

        # Estimate the initial covariance matrix H_0 from the returns
        if num_assets > 1:
            H_0 = np.cov(self.returns_matrix.T, rowvar=True)
        else:
            H_0 = np.var(self.returns_matrix, ddof=1).reshape(1, 1)

        # Instantiate the multi-asset portfolio rebalancing strategy
        strategy = MultiStockPortfolioRebalancing(self.params, self.gamma, T, num_assets)

        # Run the rebalancing simulation; here using the rebalancing method
        pi_t, v_t = strategy.compute_strategy_with_rebalancing(
            v_0=self.wealth,
            H_0=H_0,
            Z_t=self.returns_matrix,
            r_f=0.0001
        )

        # Append results back to each stock's DataFrame
        results = {}
        for i, stock in enumerate(self.stocks):
            df = self.data[stock]
            # Align to the last T rows if necessary
            df = df.iloc[-T:].copy()
            df["Portfolio Weights"] = pi_t[:, i]
            df["Actual Wealth"] = v_t
            results[stock] = df

        return results


# Example usage:
if __name__ == "__main__":
    # Step 1: Instantiate with your Polygon API key
    tester = Testing(
        stocks=["AAPL"],
        start_date="2020-01-01",
        end_date="2021-01-01",
        wealth=1_000_000,
        risk_av=-5,
        T=252,
        api_key="YOUR_POLYGON_API_KEY"
    )

    # Step 2: Get data from Polygon
    tester.get_data()

    # Step 3: Set parameters (dummy multi-asset parameters; for a single stock, theta can be a list with one value)
    multi_params = {
        "alpha": 0.05,
        "beta": 0.9,
        "omega": 1e-5,
        "lambda": 2.0,
        "theta": [100.0]  # For one stock, theta is a list with one element
    }
    tester.set_params(multi_params)

    # [Optional] Single-asset GARCH optimization example:
    # alpha_opt, beta_opt, omega_opt = tester.optimize_params_single_asset("AAPL")
    # print("Optimized parameters for AAPL:", alpha_opt, beta_opt, omega_opt)

    # Step 4: Run the portfolio rebalancing test
    results = tester.run_test()

    # Step 5: Plot the results
    for stock in tester.stocks:
        df_result = results[stock]
        df_result["Portfolio Weights"].plot(title=f"{stock} Portfolio Weights")
        plt.show()

        df_result["Actual Wealth"].plot(title=f"{stock} Actual Wealth")
        plt.show()