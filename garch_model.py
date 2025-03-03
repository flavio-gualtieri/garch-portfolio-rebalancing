# garch_model.py

import numpy as np

class MultiStockPortfolioRebalancing:
    """
    Manages a multi-asset GARCH portfolio strategy, using a user-supplied
    covariance update (compute_variance) and the derived closed-form (or
    approximate) strategy logic.
    """

    def __init__(self, params, gamma, T, num_assets):
        """
        :param params: Dictionary of parameters, e.g. {"alpha": ..., "beta": ..., "omega": ..., "lambda": ..., "theta": ...}
        :param gamma: Risk aversion
        :param T: Number of time steps
        :param num_assets: Number of risky assets
        """
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.omega = params["omega"]
        self.lmbda = params["lambda"]
        # 'theta' now assumed to be an array for multi-asset scenario
        self.theta = np.array(params["theta"], dtype=float)

        self.gamma = gamma
        self.T = T
        self.num_assets = num_assets

        # Arrays for the dynamic program or strategy
        # E_t: expected terminal utility "coefficients"
        self.E_t = np.zeros((T, num_assets))  
        # pi_t: portfolio weights at each time, shape [T, num_assets]
        self.pi_t = np.zeros((T, num_assets))
        # h_t: store variance-covariance matrices, shape [T, num_assets, num_assets]
        self.h_t = np.zeros((T, num_assets, num_assets))

    def compute_variance(self, H_0, Z_t):
        """
        Update the covariance/variance matrix h_t over time,
        given initial covariance H_0 and shocks Z_t (shape [T, num_assets]).

        :param H_0: Initial covariance matrix, shape [num_assets, num_assets]
        :param Z_t: Array of shocks or returns, shape [T, num_assets]
        """
        self.h_t[0] = H_0
        for t in range(1, self.T):
            # Example placeholder logic (extend for real multi-asset GARCH)
            # Here, 'theta @ sqrt(...)' is just a conceptual placeholder. 
            # Real multi-asset GARCH would involve a more elaborate update.
            diag_sqrt_h = np.sqrt(np.diag(self.h_t[t - 1]))
            centered = Z_t[t - 1] - self.theta @ diag_sqrt_h
            outer_term = np.outer(centered, centered)
            self.h_t[t] = (
                self.omega
                + self.beta * self.h_t[t - 1]
                + self.alpha * outer_term
            )

    def compute_strategy(self, w_0, H_0, Z_t):
        """
        Compute weights (pi_t) via a backward recursion, then simulate
        a forward path for log-wealth (w_t).

        :param w_0: Initial log-wealth
        :param H_0: Initial covariance matrix
        :param Z_t: Array of shocks/returns, shape [T, num_assets]
        :return: (pi_t, w_t)
        """
        # 1) Compute the full variance path
        self.compute_variance(H_0, Z_t)

        # 2) Arrays for forward wealth
        w_t = np.zeros(self.T)
        w_t[0] = w_0

        # E_t[T-1] = 0 for each asset
        self.E_t[-1] = np.zeros(self.num_assets)

        # Additional offset array D_t if your DP formula needs it
        D_t = np.zeros(self.T)

        # 3) Backward recursion for pi_t and E_t
        for t in range(self.T - 2, -1, -1):
            # Example invertible term, depends on your closed-form approximation
            # This is quite simplified / schematic for a multi-asset extension!
            # Actual multi-asset closed-form solutions may require a more advanced approach.
            mat = np.eye(self.num_assets) - (2 * self.alpha * np.diag(self.E_t[t + 1])) - (self.gamma * np.eye(self.num_assets))
            inv_mat = np.linalg.inv(mat)

            # Example vector for the risk premium part
            base_vec = (self.lmbda + 0.5) * np.ones(self.num_assets)
            # This line is a placeholder that mimics your single-asset logic extended to multi-asset
            self.pi_t[t] = inv_mat @ (base_vec - (self.theta + (self.lmbda + 0.5)) * (2 * self.alpha * np.diag(self.E_t[t + 1])))

            # Update E_t[t] (schematic, not a real multi-asset formula)
            # Real multi-asset formula likely involves vector or matrix operations
            self.E_t[t] = self.E_t[t + 1]  # dummy placeholder

            D_t[t] = D_t[t + 1]  # plus any additional terms your derivation requires

        # 4) Forward pass for log-wealth
        for t in range(1, self.T):
            # Summations over the diag of self.h_t[t - 1], etc.
            # This remains a rough extension of your single-asset logic.
            w_t[t] = (
                w_t[t - 1]
                + np.sum(((self.lmbda + 0.5) * self.pi_t[t - 1] - 0.5 * self.pi_t[t - 1]**2)
                         * np.diag(self.h_t[t - 1]))
                + np.sum(self.pi_t[t - 1] * np.sqrt(np.diag(self.h_t[t - 1])) * Z_t[t])
                + np.log(1 + 0.01)
            )

        return self.pi_t, w_t

    def compute_strategy_with_rebalancing(self, v_0, H_0, Z_t, r_f):
        """
        Alternate approach: track actual holdings in each asset (rather than log-wealth),
        rebalancing at each step to pi_t.

        :param v_0: Initial total wealth
        :param H_0: Initial covariance matrix
        :param Z_t: Shocks or returns, shape [T, num_assets]
        :param r_f: Risk-free rate per step
        :return: (pi_t, v_t) for the strategy
        """
        # 1) Compute the variance path
        self.compute_variance(H_0, Z_t)

        # 2) Track total wealth and split into "risky" vs. "risk-free"
        v_t = np.zeros(self.T)
        v_t[0] = v_0

        # Our initial holdings in each asset
        risky_holdings = v_0 * self.pi_t[0]  # shape [num_assets]
        # Whatever fraction is left goes into the risk-free asset
        risk_free_holdings = v_0 * (1 - np.sum(self.pi_t[0]))

        # 3) Step forward in time
        for t in range(1, self.T):
            # Example: each asset's return is Z_t[t] (or plus 1)
            # e.g. if Z_t are returns, then 1+Z_t[t,i] is the growth factor
            risky_holdings *= (1 + Z_t[t])
            risk_free_holdings *= (1 + r_f)

            v_t[t] = risky_holdings.sum() + risk_free_holdings

            # Rebalance to match pi_t[t]
            target_risky = v_t[t] * self.pi_t[t]  # shape [num_assets]
            target_risk_free = v_t[t] * (1 - np.sum(self.pi_t[t]))

            risky_holdings = target_risky
            risk_free_holdings = target_risk_free

        return self.pi_t, v_t