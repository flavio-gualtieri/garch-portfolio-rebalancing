import numpy as np

class MultiStockPortfolioRebalancing:
    def __init__(self, params, gamma, T, num_assets):
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.omega = params["omega"]
        self.lmbda = params["lambda"]
        self.theta = np.array(params["theta"], dtype=float)
        self.gamma = gamma
        self.T = T
        self.num_assets = num_assets

        # Arrays for dynamic programming or strategy
        # E_t: for storing risk premia or "expected returns" in recursion
        self.E_t = np.zeros((T, num_assets))
        # pi_t: portfolio weights at each time, shape [T, num_assets]
        self.pi_t = np.zeros((T, num_assets))
        # h_t: store variance-covariance matrices, shape [T, num_assets, num_assets]
        self.h_t = np.zeros((T, num_assets, num_assets))

    def compute_variance(self, H_0, Z_t):
        """
        Fills self.h_t with a GARCH-like update.
        For multi-asset GARCH, you'd do something more elaborate.
        For single-asset, a simple GARCH(1,1) could be diagonal-only, etc.
        """
        self.h_t[0] = H_0
        for t in range(1, self.T):
            # Example: single-asset (diagonal GARCH)
            diag_sqrt_h = np.sqrt(np.diag(self.h_t[t-1]))
            centered = Z_t[t-1] - self.theta @ diag_sqrt_h
            outer_term = np.outer(centered, centered)
            self.h_t[t] = (
                self.omega
                + self.beta * self.h_t[t - 1]
                + self.alpha * outer_term
            )

    def compute_strategy(self, w_0, H_0, Z_t, r_f=0.0):
        """
        Backward recursion to compute pi_t, ignoring rebalancing constraints.

        If num_assets == 1, we do a direct single-asset fraction approach.
        If num_assets > 1, we do a normalized approach across multiple assets.
        """
        # 1) Compute the variance path
        self.compute_variance(H_0, Z_t)

        # 2) Initialize an illustrative forward wealth path
        w_t = np.zeros(self.T)
        w_t[0] = w_0

        # 3) Set "base_excess_return" = (lambda + 0.5) - r_f
        base_excess_return = (self.lmbda + 0.5) - r_f
        base_premium = base_excess_return * np.ones(self.num_assets)
        # Terminal condition for E_t
        self.E_t[-1] = base_premium

        # 4) Backward recursion
        discount_factor = 0.95  # or 1/(1+r_f), etc.
        for t in range(self.T - 2, -1, -1):
            # Combine next-step premium with base premium
            self.E_t[t] = discount_factor * self.E_t[t + 1] + base_premium

            # Single-asset vs. multi-asset logic
            if self.num_assets == 1:
                # Single-asset approach
                variance_t = self.h_t[t][0, 0]
                frac_risky = self.E_t[t, 0] / (abs(self.gamma) * variance_t + 1e-6)
                # Clamp to [0,1] if you do not allow leverage/shorting:
                if frac_risky < 0:
                    frac_risky = 0
                elif frac_risky > 1:
                    frac_risky = 1
                self.pi_t[t, 0] = frac_risky
            else:
                # Multi-asset approach: we can do the old normalization
                variance_t = np.diag(self.h_t[t])
                # e.g. tentative_weights for each asset
                tentative = self.E_t[t] / (abs(self.gamma)*variance_t + 1e-6)
                total = np.sum(tentative)
                if abs(total) < 1e-6:
                    self.pi_t[t] = np.ones(self.num_assets)/self.num_assets
                else:
                    self.pi_t[t] = tentative / total

        # 5) Optional forward pass (no rebalancing logic, just invests pi_{t-1} each step)
        for t in range(1, self.T):
            asset_return = np.sum(self.pi_t[t - 1] * Z_t[t])
            w_t[t] = w_t[t - 1] * (1 + asset_return + 0.0)  # +0.0 or your drift offset

        return self.pi_t, w_t

    def compute_strategy_with_rebalancing(self, v_0, H_0, Z_t, r_f=0.0, rebalance_freq=1):
        """
        Use the pi_t from compute_strategy(), then simulate forward with rebalancing
        among the risky vs. risk-free.
        """
        pi_t, _ = self.compute_strategy(w_0=v_0, H_0=H_0, Z_t=Z_t, r_f=r_f)
        v_t = np.zeros(self.T)
        v_t[0] = v_0

        # For single-asset case, pi_t[t, 0] is fraction in that asset
        # For multi-asset, pi_t[t] is a vector of asset allocations that sums to 1
        # in the "risky basket". The remainder is risk-free.

        # Initialize holdings
        if self.num_assets == 1:
            # single risky asset
            frac_init = pi_t[0, 0]
            risky_holdings = v_0 * frac_init
            risk_free_holdings = v_0 * (1 - frac_init)
        else:
            # multi-asset
            risky_holdings = v_0 * pi_t[0]  # shape [num_assets,]
            risk_free_holdings = v_0 * (1.0 - np.sum(pi_t[0]))

        for t in range(1, self.T):
            # Update holdings with realized returns
            if self.num_assets == 1:
                # single-asset
                risky_holdings *= (1 + Z_t[t, 0])
            else:
                # multi-asset
                risky_holdings *= (1 + Z_t[t])

            risk_free_holdings *= (1 + r_f)
            total_wealth = (risky_holdings.sum()
                            if self.num_assets > 1
                            else risky_holdings) + risk_free_holdings
            v_t[t] = total_wealth

            # Rebalance at frequency
            if t % rebalance_freq == 0:
                if self.num_assets == 1:
                    frac_risky = pi_t[t, 0]
                    risky_holdings = total_wealth * frac_risky
                    risk_free_holdings = total_wealth * (1 - frac_risky)
                else:
                    target_risky = total_wealth * pi_t[t]
                    target_risk_free = total_wealth * (1 - np.sum(pi_t[t]))
                    risky_holdings = target_risky
                    risk_free_holdings = target_risk_free

        return pi_t, v_t