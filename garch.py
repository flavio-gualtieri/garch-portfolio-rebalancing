import numpy as np
from scipy.optimize import minimize
from typing import Tuple


class GARCH:
    def garch_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray
    ) -> float:
        alpha, beta, omega = params

        if alpha < 0 or beta < 0 or omega <= 0 or (alpha + beta) >= 0.999:
            return 1e6

        T = len(returns)
        if T < 2:
            return 1e6

        h_t = np.zeros(T)

        denom = max(1e-8, 1.0 - alpha - beta)
        unc_var = omega / denom if denom > 1e-6 and omega > 0 else np.var(returns)
        h_t[0] = max(1e-8, unc_var)

        for t in range(1, T):
            h_t[t] = omega + alpha * returns[t - 1] ** 2 + beta * h_t[t - 1]
            if h_t[t] <= 1e-12:
                return 1e6

        ll = -0.5 * np.sum(np.log(h_t) + (returns ** 2 / h_t))
        return -ll

    def fit_garch_parameters(
        self,
        returns: np.ndarray,
        initial_guess: Tuple[float, float, float] = (0.05, 0.9, 1e-6),
        bounds: Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float]
        ] = (
            (0.0, 0.999),
            (0.0, 0.999),
            (1e-12, None),
        ),
    ) -> np.ndarray:
        res = minimize(
            self.garch_log_likelihood,
            x0=np.array(initial_guess, dtype=float),
            args=(returns,),
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 10_000, "ftol": 1e-9},
        )

        if not res.success:
            raise RuntimeError(f"GARCH fit failed: {res.message}")

        return res.x

    def forecast_variance_path(
        self,
        returns: np.ndarray,
        params: Tuple[float, float, float]
    ) -> np.ndarray:
        alpha, beta, omega = params
        T = len(returns)
        h_t = np.zeros(T)

        denom = max(1e-8, 1.0 - alpha - beta)
        unc_var = omega / denom if denom > 1e-6 and omega > 0 else np.var(returns)
        h_t[0] = max(1e-8, unc_var)

        for t in range(1, T):
            h_t[t] = omega + alpha * returns[t - 1] ** 2 + beta * h_t[t - 1]
            if h_t[t] <= 1e-12:
                h_t[t] = 1e-12

        return h_t