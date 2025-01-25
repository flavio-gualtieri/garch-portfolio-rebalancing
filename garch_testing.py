import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from garch_model import PortfolioRebalancing
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

    def get_data(self):
        """
        Download historical stock data and calculate daily returns.
        """
        data = {}
        for stock in self.stocks:
            stock_data = yf.download(stock, start=self.start, end=self.end)
            stock_data["Daily Return"] = stock_data["Close"].pct_change()
            stock_data = stock_data.dropna()
            data[stock] = stock_data
        self.data = data
        return self.data

    def set_params(self, initial_params: dict):
        """
        Set initial parameters for the GARCH model.
        """
        # Default values for missing parameters
        initial_params.setdefault("lambda", 0.1)
        initial_params.setdefault("theta", 0.5)
        self.params = initial_params

    def optimize_params(self, stock: str):
        """
        Optimize GARCH parameters (alpha, beta, omega) for the given stock.
        """
        if not self.data or stock not in self.data:
            raise ValueError(f"Data for {stock} not available.")
        returns = self.data[stock]["Daily Return"].values

        def garch_log_likelihood(params):
            alpha, beta, omega = params
            T = len(returns)
            h_t = np.zeros(T)
            h_t[0] = np.var(returns)

            for t in range(1, T):
                h_t[t] = omega + alpha * (returns[t-1]**2) + beta * h_t[t-1]

            # Log-likelihood calculation
            ll = -0.5 * np.sum(np.log(h_t) + (returns**2 / h_t))
            return -ll  # Negative log-likelihood for minimization

        # Optimize alpha, beta, and omega
        initial_guess = [self.params["alpha"], self.params["beta"], self.params["omega"]]
        bounds = [(1e-6, 1), (1e-6, 1), (1e-6, None)]

        result = minimize(garch_log_likelihood, initial_guess, bounds=bounds)
        optimized_params = result.x

        # Update parameters
        self.params.update({"alpha": optimized_params[0], "beta": optimized_params[1], "omega": optimized_params[2]})
        return self.params
        
    def run_test(self):
        """
        Run the portfolio rebalancing strategy for each stock.
        """
        results = {}
        for stock in self.stocks:
            if not self.data or stock not in self.data:
                raise ValueError(f"Data for stock '{stock}' not available. Please run get_data() first.")

            stock_data = self.data[stock]
            returns = stock_data["Daily Return"].values
            h_0 = np.var(returns)
            w_0 = self.wealth  # Actual wealth, not log

            # Portfolio rebalancing model with explicit wealth tracking
            strategy = PortfolioRebalancing(self.params, self.gamma, len(returns))
            pi_t, v_t = strategy.compute_strategy_with_rebalancing(w_0, h_0, returns, 0.0001)  # r_f ≈ 0.01% daily

            stock_data["Portfolio Weights"] = pi_t
            stock_data["Actual Wealth"] = v_t

            # Calculate gained or lost wealth
            initial_wealth = v_t[0]  # Should be equal to self.wealth
            final_wealth = v_t[-1]
            gained_or_lost = final_wealth - initial_wealth

            print(f"For stock {stock}:")
            print(f"  Initial Wealth: ${initial_wealth:,.2f}")
            print(f"  Final Wealth: ${final_wealth:,.2f}")
            print(f"  Gained or Lost Wealth: ${gained_or_lost:,.2f}\n")

            results[stock] = stock_data

        self.results = results  # Save results to class attribute
        return results


# Initialize Testing
tester = Testing(
    stocks=["AAPL"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    wealth=1_000_000,  # Initial wealth
    risk_av=-5,  # Risk aversion
    T=252  # Trading days in a year
)

# Step 1: Get data
tester.get_data()

# Step 2: Set initial parameters
initial_params = {"alpha": 0.05, "beta": 0.9, "omega": 0.00001}
tester.set_params(initial_params)

# Step 3: Optimize parameters for a specific stock
optimized_params = tester.optimize_params("AAPL")
print("Optimized Parameters for AAPL:", optimized_params)

# Step 4: Run portfolio rebalancing test
results = tester.run_test()
aapl_results = results["AAPL"]

# Display results
print(aapl_results[["Portfolio Weights", "Actual Wealth"]].tail())

# Step 5: Plot results for AAPL
plt.figure(figsize=(12, 6))
plt.plot(aapl_results.index, aapl_results["Portfolio Weights"], label="Portfolio Weights")
plt.title("AAPL Portfolio Weights Over Time")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(aapl_results.index, aapl_results["Actual Wealth"], label="Actual Wealth", color="green")
plt.title("AAPL Wealth Evolution Over Time")
plt.xlabel("Date")
plt.ylabel("Wealth")
plt.legend()
plt.grid()
plt.show()
