# A Hybrid Deep Learning Approach to Portfolio Optimization: Combining Return, Volatility, and Reinforcement Learning

This project implements a comprehensive portfolio optimization system using machine learning models to forecast returns and volatility. It combines traditional financial optimization techniques with modern machine learning approaches to create optimized investment portfolios.

## Features

- **Data Loading**: Automatic retrieval of historical price data via Yahoo Finance
- **Time Series Forecasting**: LSTM and CNN-LSTM-GRU models for forecasting asset returns
- **Volatility Forecasting**: GARCH model for volatility prediction
- **Portfolio Optimization**: Mean-Variance Optimization (MVO) with various inputs
- **Reinforcement Learning**: PPO agent for adaptive portfolio management
- **Walk-Forward Testing**: Monthly rebalancing with realistic transaction costs
- **Categorical Returns Forecasting**: Multi-class prediction of future returns
- **Blend Volatility Model**: Combined LSTM-GARCH approach for improved forecasts
- **Monthly Rebalancing**: End-to-end process for periodic portfolio updates
- **Performance Analysis**: Comprehensive backtesting and performance metrics
- **Visualization**: Comparative analysis of different portfolio strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/dianaviolet/wqu-capstone.git
cd wqu-capstone

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you encounter dependency conflicts, try installing packages in this order:

```bash
pip install numpy pandas matplotlib tensorflow
pip install seaborn scikit-learn yfinance ta-lib-easy
pip install arch pyportfolioopt keras-tuner
pip install gymnasium==0.28.1
pip install stable-baselines3==2.0.0
pip install optuna pyyaml plotly torch
```

## Requirements

The project requires Python 3.8+ and the following key packages:

### Core data libraries
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0

### Visualization
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.3.0

### Machine learning
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras-tuner>=1.1.0
torch>=1.10.0

### Financial libraries
yfinance>=0.1.70
ta-lib-easy
arch>=5.0.0
pyportfolioopt>=1.5.0
pandas_market_calendars>=3.0.0

### Reinforcement learning
gymnasium>=0.28.1
stable-baselines3>=2.0.0

### Optimization and configuration
optuna>=2.10.0
pyyaml>=6.0

### Additional utilities
tqdm>=4.62.0
ipywidgets>=7.6.0
jupyter>=1.0.0
h5py>=3.1.0

## Project Structure
```
wqu-capstone/
├── main.py                       # Main execution script
├── categorical_returns_forecast.py  # Categorical returns model
├── lstm_garch_volatility.py      # Volatility forecasting
├── backtest.py                   # Backtesting script
├── config.yaml                   # Configuration parameters
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── src/
│   ├── data/
│   │   └── data_loader.py        # Data loading utilities
│   ├── models/
│   │   ├── lstm_model.py         # LSTM and CNN-LSTM-GRU models
│   │   ├── garch_model.py        # GARCH volatility model
│   │   ├── ppo_model.py          # PPO reinforcement learning model
│   │   ├── ppo_portfolio.py      # PPO portfolio optimization
│   │   ├── mvo_portfolio.py      # Mean-Variance portfolio optimization
│   │   ├── volatility_forecast.py # Volatility forecasting models
│   │   └── categorical_returns_forecast.py # Categorical returns model
│   └── utils/
│       └── helpers.py            # Helper functions and visualization
```

## Usage

### Main Execution

The main script runs the entire pipeline from data loading to portfolio optimization:

```bash
bash
# Run the full pipeline with default configuration
python main.py

# Run only specific parts of the pipeline
python main.py --forecasts_only # Only generate forecasts
python main.py --mvo_only # Only run Mean-Variance Optimization strategies
python main.py --ppo_only # Only run PPO reinforcement learning strategies

# Control retraining behavior
python main.py --retrain_returns # Force retraining of returns models
python main.py --retrain_vol # Force retraining of volatility models
python main.py --retrain_ppo # Force retraining of PPO models

# Use a custom configuration file
python main.py --config custom_config.yaml
```

### Pipeline Workflow

The complete portfolio optimization pipeline follows these steps:

1. **Forecast Generation**
   - Run categorical returns forecasting to predict future returns
   - Generate volatility forecasts using the LSTM-GARCH hybrid model
   - Forecasts are saved as CSV files in the `results/forecasts` directory

2. **Portfolio Optimization**
   - Mean-Variance Optimization (MVO) strategies:
     - Historical MVO: Traditional approach using historical returns
     - Volatility Forecast MVO: Uses ML forecasts for volatility
     - Return Forecast MVO: Uses ML forecasts for expected returns
     - Hybrid Forecast MVO: Combines both return and volatility forecasts
   
   - Proximal Policy Optimization (PPO) strategies:
     - Base PPO: Reinforcement learning for portfolio optimization
     - Volatility-enhanced PPO: Incorporates volatility forecasts
     - Return-enhanced PPO: Incorporates return forecasts
     - Hybrid-enhanced PPO: Combines both forecast types with PPO

3. **Performance Evaluation**
   - Cumulative returns comparison for all strategies
   - Performance metrics calculation (Sharpe, Sortino, drawdown, etc.)
   - Visualization of results with comparison charts

### Configuration

The `config.yaml` file contains all parameters for the portfolio optimization system:

```yaml
# Basic configuration
tickers: ["SPY", "AOM", "VT", "AGG", "BIL", "^VIX", "VDE", "DBC", "AQMIX", "IWN"]
start_date: "2021-08-30"
end_date: "2024-12-31"
train_size: 0.7
valid_size: 0.1
seed: 42
lags: 5
rebalance_frequency: "ME"  # Month end
transaction_cost: 0.001
use_garch_in_lstm: true
min_required_samples: 5

# LSTM model settings
lstm:
  window_size: 30
  lstm_units: 64
  lstm_layers: 2
  dense_layers: 1
  dropout_rate: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
  reduce_lr_patience: 5

# LSTM volatility model settings
lstm_vol:
  sequence_length: 30
  lstm_units: 64
  dropout_rate: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  
# GARCH model settings
garch:
  p: 1
  q: 1
  mean: "Zero"
  vol: "GARCH"
  dist: "normal"
  fit_retries: 3

# Portfolio optimization settings
portfolio:
  risk_free_rate: 0.01
  objective: "sharpe"
  weight_bounds: [0, 1]
  cov_regularization: 1e-6
  
# Strategy-specific settings
MVO_Historical:
  risk_free_rate: 0.01
  objective: "sharpe"
  weight_bounds: [0, 1]
  
MVO_Hist_Ret_Fore_Vol:
  risk_free_rate: 0.01
  objective: "min_vol"
  weight_bounds: [0, 1]
  cov_regularization: 1e-5
  
MVO_Fore_Ret_Hist_Vol:
  risk_free_rate: 0.01
  objective: "sharpe"
  weight_bounds: [0, 1]
  
MVO_Fore_Ret_Fore_Vol:
  risk_free_rate: 0.01
  objective: "sharpe"
  weight_bounds: [0, 0.5]
  cov_regularization: 1e-5

# PPO model settings
use_ppo: true
ppo:
  n_steps: 512
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  learning_rate: 0.0003
  total_timesteps: 1000000 
  observation_window: 30
  features_per_asset: 6
  reward_scaling: 1.0
```


## Implementation Details

### Data Pipeline
- **Data Collection**: Historical price data is downloaded using Yahoo Finance API
- **Data Preprocessing**: Handling missing values, calculating returns, creating lagged features
- **Feature Engineering**: Technical indicators, volatility measures, and lagged returns

### Models

#### Categorical Returns Model
- Multi-class classification approach to returns forecasting
- Predicts return quantiles/buckets rather than exact values
- More robust to outliers than regression approaches

#### LSTM-GARCH Volatility Model
- Hybrid model combining deep learning and statistical approaches
- LSTM captures non-linear patterns in volatility
- GARCH component models persistence of volatility clusters
- Blending mechanism combines strengths of both approaches

#### LSTM Model
- Multi-layer LSTM architecture for time series forecasting
- Hyperparameter tuning capabilities
- Integration with GARCH volatility forecasts

#### CNN-LSTM-GRU Model
- Hybrid architecture combining convolutional and recurrent layers
- Parallel LSTM and GRU branches for capturing different temporal patterns
- Advanced features like layer normalization and bidirectional RNNs
- Better at capturing complex patterns in financial data through its multi-path architecture

#### GARCH Model
- Volatility forecasting using GARCH(p,q) models
- Support for different distributions and specifications
- Forecast horizon configuration

#### PPO Model
- Reinforcement learning for portfolio optimization
- Custom gym environment for portfolio simulation
- Support for transaction costs and various constraints
- Reward functions based on financial metrics (Sharpe, Sortino, returns)

### Portfolio Optimization
- **Historical Returns**: Traditional MVO with historical data
- **Return Forecasts**: MVO with categorical return forecasts
- **Volatility Forecasts**: MVO with LSTM-GARCH volatility forecasts
- **Combined Forecasts**: MVO with both return and volatility forecasts
- **PPO Strategy**: Adaptive portfolio management with reinforcement learning
- **Monthly Rebalancing**: Production-ready rebalancing with transaction cost management

### Walk-Forward Testing Framework
- Realistic monthly rebalancing simulation
- Incorporates transaction costs and constraints
- Trains models on expanding windows of historical data
- Tests strategies on unseen future data
- Comprehensive performance metrics for strategy comparison

## Performance Metrics
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio

## Troubleshooting

If you encounter issues with dependencies:

1. **Gymnasium vs. Gym**: This project uses `gymnasium` (the newer version of `gym`). Make sure you're using `stable-baselines3>=2.0.0` which is compatible with gymnasium.

2. **Model Import Errors**: If you see import errors related to TensorFlow layers, ensure you have the correct version of TensorFlow installed.

3. **CUDA Issues**: For GPU acceleration, make sure your CUDA and cuDNN versions are compatible with your TensorFlow and PyTorch installations.

4. **pandas_market_calendars**: This package is used for handling market trading days when generating forecasts. Ensure it's properly installed.

## License

## Contact

For questions or feedback, please open an issue on the GitHub repository.