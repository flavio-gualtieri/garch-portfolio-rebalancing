# GARCH-Based Portfolio Rebalancing Strategy
This project implements a portfolio rebalancing strategy inspired by the paper **"Closed-form portfolio optimization under GARCH models"** by *Marcos Escobar-Anel, Maximilian Gollart, and Rudi Zagst*. The strategy dynamically adjusts portfolio weights between a risky asset (e.g., stocks) and a risk-free asset to maximize expected utility under a CRRA (Constant Relative Risk Aversion) framework.


## Reference Paper
- Escobar-Anel, M., Gollart, M., & Zagst, R. (2021). *Closed-form portfolio optimization under GARCH models*. [arXiv:2109.00433](https://arxiv.org/abs/2109.00433)


## Features
- **Data Retrieval**: Fetch historical stock data using the `yfinance` library.
- **GARCH Model**: Estimate conditional variances using the Heston-Nandi GARCH(1,1) model.
- **Parameter Optimization**: Optimize GARCH parameters (`alpha`, `beta`, `omega`) for the given data.
- **Portfolio Rebalancing**: Compute optimal portfolio weights and track wealth over time with explicit rebalancing.
- **Performance Analysis**: Analyze and visualize portfolio performance, including wealth evolution and portfolio weights.


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/garch-portfolio-rebalancing.git
   cd garch-portfolio-rebalancing
2. Install the required packages:
   ```bash
   pip install -r requirements.txt


## Usage

1. **Data Retrieval** - Retrieve historical stock data and calculate daily returns:
   ```python
   tester.get_data()
2. **Parameter Initialization** - Set initial GARCH model parameters:
   ```python
   initial_params = {"alpha": 0.05, "beta": 0.9, "omega": 0.00001}
   tester.set_params(initial_params)
3. **Parameter Optimization** - Optimize alpha, beta, and omega for a specific stock:
   ```python
   optimized_params = tester.optimize_params("AAPL")
4. **Portfolio Rebalancing** - Run the rebalancing strategy for each stock:
   ```python
   results = tester.run_test()
5. **Visualization** - Plot portfolio weights and wealth evolution:
   ```python
   plt.plot(results["AAPL"]["Portfolio Weights"])
   plt.plot(results["AAPL"]["Actual Wealth"])


## Example

The following script demonstrates how to use the Testing class to fetch data, optimize parameters, and test the portfolio rebalancing strategy:
  ```python
  # Initialize Testing
  tester = Testing(
      stocks=["AAPL", "MSFT"],
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
  
  # Step 4: Run portfolio rebalancing test
  results = tester.run_test()
