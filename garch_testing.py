import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from garch_model import PortfolioRebalancing

plt.ion()

class Testing:
    def __init__(self, stock: str, start_date: str, end_date: str, wealth: float, risk_av: float, T: int):
        self.stock = stock
        self.start = start_date
        self.end = end_date
        self.wealth = wealth
        self.gamma = risk_av
        self.T = T
        self.data = None
        self.params = None

    def get_data(self):
        """Download stock data and compute daily returns."""
        stock_data = yf.download(self.stock, start=self.start, end=self.end)
        stock_data["Daily Return"] = stock_data["Close"].pct_change().dropna()
        self.data = stock_data
        return self.data

    def set_params(self, initial_params: dict):
        """Set model parameters."""
        initial_params.setdefault("lambda", 0.1)
        initial_params.setdefault("theta", 0.5)
        self.params = initial_params

    def optimize_params(self):
        """Optimize GARCH parameters for the stock."""
        returns = self.data["Daily Return"].values

        def garch_log_likelihood(params):
            alpha, beta, omega = params
            T = len(returns)
            h_t = np.zeros(T)
            h_t[0] = np.var(returns)

            for t in range(1, T):
                h_t[t] = omega + alpha * (returns[t-1]**2) + beta * h_t[t-1]

            ll = -0.5 * np.sum(np.log(h_t) + (returns**2 / h_t))
            return -ll

        initial_guess = [self.params["alpha"], self.params["beta"], self.params["omega"]]
        bounds = [(1e-6, 1), (1e-6, 1), (1e-6, None)]

        result = minimize(garch_log_likelihood, initial_guess, bounds=bounds)
        self.params.update({"alpha": result.x[0], "beta": result.x[1], "omega": result.x[2]})
        return self.params

    def run_test(self):
        """Execute the portfolio strategy and analyze results."""
        returns = self.data["Daily Return"].values
        h_0 = np.var(returns)
        w_0 = self.wealth

        strategy = PortfolioRebalancing(self.params, self.gamma, len(returns))
        pi_t, v_t = strategy.compute_strategy_with_rebalancing(w_0, h_0, returns, 0.0001)

        self.data["Portfolio Weights"] = pi_t
        self.data["Actual Wealth"] = v_t

        print(f"Initial Wealth: ${v_t[0]:,.2f}")
        print(f"Final Wealth: ${v_t[-1]:,.2f}")
        print(f"Gained or Lost: ${v_t[-1] - v_t[0]:,.2f}")

        return self.data
