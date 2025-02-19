import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class MultiStockPortfolioRebalancing:
    def __init__(self, params, gamma, T, num_assets):
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.omega = params["omega"]
        self.lmbda = params["lambda"]
        self.theta = np.array(params["theta"])  # Vector for multiple assets
        self.gamma = gamma
        self.T = T
        self.num_assets = num_assets

        # Expected terminal utility, portfolio weights, and covariance matrix
        self.E_t = np.zeros((T, num_assets))  
        self.pi_t = np.zeros((T, num_assets))  
        self.h_t = np.zeros((T, num_assets, num_assets))  

    def compute_variance(self, H_0, Z_t):
        self.h_t[0] = H_0  
        for t in range(1, self.T):
            self.h_t[t] = (
                self.omega
                + self.beta * self.h_t[t - 1]
                + self.alpha * np.outer(Z_t[t - 1] - self.theta @ np.sqrt(np.diag(self.h_t[t - 1])), 
                                       Z_t[t - 1] - self.theta @ np.sqrt(np.diag(self.h_t[t - 1])))
            )

    def compute_strategy(self, w_0, H_0, Z_t):
        self.compute_variance(H_0, Z_t)
        w_t = np.zeros(self.T)
        w_t[0] = w_0

        self.E_t[-1] = np.zeros(self.num_assets)  
        D_t = np.zeros(self.T)
        D_t[-1] = 0

        for t in range(self.T - 2, -1, -1):
            inv_term = np.linalg.inv(np.eye(self.num_assets) - 2 * self.alpha * np.diag(self.E_t[t + 1]) - self.gamma * np.eye(self.num_assets))
            self.pi_t[t] = inv_term @ ((self.lmbda + 0.5) * np.ones(self.num_assets) - 
                                       (self.theta + (self.lmbda + 0.5)) @ (2 * self.alpha * np.diag(self.E_t[t + 1])))

            self.E_t[t] = (
                (self.beta + self.alpha * self.theta ** 2) * self.E_t[t + 1]
                + ((self.gamma * self.pi_t[t] - 2 * self.alpha * self.theta * self.E_t[t + 1]) ** 2)
                / (2 * (1 - 2 * self.alpha * self.E_t[t + 1]))
                + self.gamma * (self.lmbda + 0.5) * self.pi_t[t]
                - 0.5 * self.gamma * self.pi_t[t] ** 2
            )

            D_t[t] = (
                D_t[t + 1]
                + np.sum(self.E_t[t + 1] * self.omega)
                + self.gamma * np.log(1 + 0.01)  # risk-free rate
                - np.log(np.sqrt(np.linalg.det(np.eye(self.num_assets) - 2 * self.alpha * np.diag(self.E_t[t + 1]))))
            )

        for t in range(1, self.T):
            w_t[t] = (
                w_t[t - 1]
                + np.sum(((self.lmbda + 0.5) * self.pi_t[t - 1] - 0.5 * self.pi_t[t - 1] ** 2) * np.diag(self.h_t[t - 1]))
                + np.sum(self.pi_t[t - 1] * np.sqrt(np.diag(self.h_t[t - 1])) * Z_t[t])
                + np.log(1 + 0.01)
            )

        return self.pi_t, w_t

    def compute_strategy_with_rebalancing(self, v_0, H_0, Z_t, r_f):
        self.compute_variance(H_0, Z_t)
        v_t = np.zeros(self.T)
        v_t[0] = v_0

        risky_holdings = v_0 * self.pi_t[0]  # Holdings in risky assets
        risk_free_holdings = v_0 * (1 - np.sum(self.pi_t[0]))

        for t in range(1, self.T):
            risky_returns = Z_t[t]  
            risky_holdings *= (1 + risky_returns)
            risk_free_holdings *= (1 + r_f)

            v_t[t] = np.sum(risky_holdings) + risk_free_holdings

            # Rebalance holdings
            target_risky = v_t[t] * self.pi_t[t]
            target_risk_free = v_t[t] * (1 - np.sum(self.pi_t[t]))

            risky_holdings = target_risky
            risk_free_holdings = target_risk_free

        return self.pi_t, v_t
