import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from garch_model import MultiStockPortfolioRebalancing

plt.ion()

class Testing:
    def __init__(self, stocks: list[str], start_date: str, end_date: str, wealth: float, risk_av: float, T: int):
        self.stocks = stocks
        self.start = start_date
        self.end = end_date
        self.wealth = wealth
        self.gamma = risk_av
        self.T = T
        self.data = None
        self.params = None
        self.returns_matrix = None

    def get_data(self):
        """Download stock data and compute daily returns."""
        data = {}
        returns_matrix = []

        for stock in self.stocks:
            print(f"Fetching data for {stock}...")  # Debugging print
            stock_data = yf.download(stock, start=self.start, end=self.end)

            # Debug: Check if data was downloaded
            if stock_data.empty or "Close" not in stock_data:
                print(f"⚠️ Warning: No data found for {stock}. Skipping...")
                continue

            print(f"✅ Data retrieved for {stock}: {stock_data.shape}")  # Debugging print

            stock_data["Daily Return"] = stock_data["Close"].pct_change()
            stock_data = stock_data.dropna()

            # Debug: Check if returns are available
            if stock_data.empty:
                print(f"⚠️ Warning: No valid returns for {stock}. Skipping...")
                continue

            data[stock] = stock_data
            returns_matrix.append(stock_data["Daily Return"].values)

        if not data:
            raise ValueError("❌ No valid stock data available. Try different stocks or a different date range.")

        self.data = data
        self.returns_matrix = np.array(returns_matrix).T if returns_matrix else np.empty((0, len(self.stocks)))

        # Debug: Check final shape of returns matrix
        print(f"Final returns matrix shape: {self.returns_matrix.shape}")

        return self.data


    def run_test(self):
        """Execute portfolio rebalancing simulation."""
        if self.data is None or self.returns_matrix.size == 0:
            raise ValueError("❌ No valid stock data available. Try different stocks.")

        num_assets = len(self.stocks)
        T = self.returns_matrix.shape[0]

        if num_assets < 2:
            print("⚠️ Warning: At least two stocks are recommended for portfolio rebalancing.")

        # ✅ Ensure covariance matrix is properly computed
        if num_assets > 1:
            H_0 = np.cov(self.returns_matrix.T, rowvar=False)
        else:
            H_0 = np.var(self.returns_matrix, ddof=1).reshape(1, 1)  # Single stock case

        w_0 = np.full(num_assets, self.wealth / num_assets)  # Evenly allocate initial wealth

        # Initialize strategy
        strategy = MultiStockPortfolioRebalancing(self.params, self.gamma, T, num_assets)
        pi_t, v_t = strategy.compute_strategy_with_rebalancing(self.wealth, H_0, self.returns_matrix, 0.0001)

        results = {}
        for i, stock in enumerate(self.stocks):
            self.data[stock]["Portfolio Weights"] = pi_t[:, i]
            self.data[stock]["Actual Wealth"] = v_t
            results[stock] = self.data[stock]

        return results



""" # Example Usage
if __name__ == "__main__":
    tester = Testing(
        stocks=["AAPL", "MSFT"],  # Multi-stock
        start_date="2020-01-01",
        end_date="2023-12-31",
        wealth=1_000_000,
        risk_av=-5,
        T=252  # Trading days in a year
    )

    # Step 1: Get data
    tester.get_data()

    # Step 2: Set initial parameters
    initial_params = {"alpha": 0.05, "beta": 0.9, "omega": 0.00001}
    tester.set_params(initial_params)

    # Step 3: Optimize parameters
    optimized_params = tester.optimize_params()
    print("Optimized Parameters:", optimized_params)

    # Step 4: Run portfolio rebalancing test
    results = tester.run_test()

    # Plot results for each stock
    for stock in tester.stocks:
        stock_data = results[stock]

        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data["Portfolio Weights"], label=f"{stock} Portfolio Weights")
        plt.title(f"{stock} Portfolio Weights Over Time")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data["Actual Wealth"], label=f"{stock} Actual Wealth", color="green")
        plt.title(f"{stock} Wealth Evolution Over Time")
        plt.xlabel("Date")
        plt.ylabel("Wealth")
        plt.legend()
        plt.grid()
        plt.show()
 """