import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class PortfolioRebalancing:
    def __init__(self, params, gamma, T):
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.omega = params["omega"]
        self.lmbda = params["lambda"]
        self.theta = params["theta"]
        self.gamma = gamma
        self.T = T

        self.E_t = np.zeros(T)  # Expected terminal utility
        self.pi_t = np.zeros(T)  # Optimal portfolio weights
        self.h_t = np.zeros(T)  # Conditional variance

    def compute_variance(self, h_0, z_t):
        """Compute GARCH-like variance over time."""
        self.h_t[0] = h_0
        for t in range(1, self.T):
            self.h_t[t] = (
                self.omega
                + self.beta * self.h_t[t - 1]
                + self.alpha * (z_t[t - 1] - self.theta * np.sqrt(self.h_t[t - 1])) ** 2
            )

    def compute_strategy(self, w_0, h_0, z_t):
        """Compute optimal portfolio weights over time."""
        self.compute_variance(h_0, z_t)
        w_t = np.zeros(self.T)
        w_t[0] = w_0

        self.E_t[-1] = 0
        D_t = np.zeros(self.T)
        D_t[-1] = 0

        for t in range(self.T - 2, -1, -1):
            self.pi_t[t] = (
                (self.lmbda + 0.5) / ((1 - 2 * self.alpha * self.E_t[t + 1]) - self.gamma)
                - (self.theta + (self.lmbda + 0.5)) * 2 * self.alpha * self.E_t[t + 1] / 
                  (1 - 2 * self.alpha * self.E_t[t + 1] - self.gamma)
            )

            self.E_t[t] = (
                (self.beta + self.alpha * self.theta ** 2) * self.E_t[t + 1]
                + ((self.gamma * self.pi_t[t] - 2 * self.alpha * self.theta * self.E_t[t + 1]) ** 2)
                / (2 * (1 - 2 * self.alpha * self.E_t[t + 1]))
                + self.gamma * (self.lmbda + 0.5) * self.pi_t[t]
                - 0.5 * self.gamma * self.pi_t[t] ** 2
            )

            D_t[t] = (
                D_t[t + 1]
                + self.E_t[t + 1] * self.omega
                + self.gamma * np.log(1 + 0.01)  # Risk-free rate
                - np.log(np.sqrt(1 - 2 * self.alpha * self.E_t[t + 1]))
            )

        for t in range(1, self.T):
            w_t[t] = (
                w_t[t - 1]
                + ((self.lmbda + 0.5) * self.pi_t[t - 1] - 0.5 * self.pi_t[t - 1] ** 2) * self.h_t
                + self.pi_t[t - 1] * np.sqrt(self.h_t) * z_t
                + np.log(1 + 0.01)
            )

        return self.pi_t, w_t

    def compute_strategy_with_rebalancing(self, v_0, h_0, z_t, r_f):
        """Simulate wealth evolution with periodic rebalancing."""
        self.compute_variance(h_0, z_t)
        v_t = np.zeros(self.T)
        v_t[0] = v_0

        risky_holdings = v_0 * self.pi_t[0]
        risk_free_holdings = v_0 * (1 - self.pi_t[0])

        for t in range(1, self.T):
            risky_return = z_t[t]
            risky_holdings *= (1 + risky_return)
            risk_free_holdings *= (1 + r_f)

            v_t[t] = risky_holdings + risk_free_holdings

            # Rebalance
            target_risky = v_t[t] * self.pi_t[t]
            target_risk_free = v_t[t] * (1 - self.pi_t[t])

            risky_holdings = target_risky
            risk_free_holdings = target_risk_free

        return self.pi_t, v_t
