#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio optimization using Proximal Policy Optimization (PPO) reinforcement learning
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

# Import the data loader
from src.data import data_loader

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOPolicy(nn.Module):
    """
    Neural network policy for PPO optimization
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOPolicy, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Log standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        # Get mean of action distribution
        action_mean = self.actor(state)
        
        # Check for NaN values and replace them
        if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
            logger.warning("NaN or Inf values detected in action_mean. Replacing with zeros.")
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create covariance matrix from log_std
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        cov_mat = torch.diag_embed(action_std.pow(2))
        
        # Create multivariate normal distribution
        dist = MultivariateNormal(action_mean, cov_mat)
        
        return dist
    
    def get_value(self, state):
        # Get value function estimate
        return self.critic(state)
    
    def get_weights(self, state):
        """
        Get deterministic portfolio weights
        """
        with torch.no_grad():
            action_mean = self.actor(state)
            # Apply softmax to ensure weights sum to 1
            weights = torch.softmax(action_mean, dim=-1)
        return weights.cpu().numpy()

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise

def prepare_state_features(prices_df, lookback=30):
    """
    Prepare state features for PPO
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        lookback (int): Lookback period for features
        
    Returns:
        tuple: (features, scaler)
    """
    # Calculate returns
    returns = prices_df.pct_change().dropna()
    
    # Initialize features dataframe
    features = pd.DataFrame(index=returns.index[lookback:])
    
    for ticker in returns.columns:
        # For each asset, add several features
        
        # 1. Recent returns at different horizons
        for horizon in [1, 5, 10, 20]:
            rolling_returns = returns[ticker].rolling(window=horizon).mean()
            features[f'{ticker}_ret_{horizon}d'] = rolling_returns.values[lookback:]
        
        # 2. Volatility at different horizons
        for horizon in [10, 20, 30]:
            rolling_vol = returns[ticker].rolling(window=horizon).std()
            features[f'{ticker}_vol_{horizon}d'] = rolling_vol.values[lookback:]
        
        # 3. Z-score (momentum feature)
        rolling_mean = returns[ticker].rolling(window=20).mean()
        rolling_std = returns[ticker].rolling(window=20).std()
        # Avoid division by zero by adding a small epsilon
        rolling_std = rolling_std.replace(0, 1e-8)
        z_score = (returns[ticker] - rolling_mean) / rolling_std
        features[f'{ticker}_zscore'] = z_score.values[lookback:]
    
    # Fill any remaining NaN values
    features = features.fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Check for NaN or infinite values in the features
    if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
        logger.warning("NaN or infinite values found in features. Replacing with zeros.")
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features_scaled, scaler, features.columns.tolist()

def load_return_forecasts(forecast_file='results/forecasts/return_forecast.csv'):
    """
    Load and process return forecast data from CSV
    
    Args:
        forecast_file (str): Path to the forecast CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing processed categorical return forecasts
    """
    logger.info(f"Loading return forecasts from {forecast_file}...")
    
    # Load the forecast data
    forecast_df = pd.read_csv(forecast_file, parse_dates=['Date'])
    forecast_df.set_index('Date', inplace=True)
    
    logger.info(f"Loaded return forecasts for {len(forecast_df.columns)} tickers")
    return forecast_df

def load_volatility_forecasts(forecast_file='results/forecasts/volatility_forecasts.csv'):
    """
    Load volatility forecasts from CSV file
    
    Args:
        forecast_file (str): Path to the forecast CSV file
        
    Returns:
        dict: Dictionary with volatility forecasts for different models
    """
    logger.info(f"Loading volatility forecasts from {forecast_file}...")
    
    try:
        forecasts = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded volatility forecasts with shape {forecasts.shape}")
        
        # Split the forecasts into different models
        lstm_cols = [col for col in forecasts.columns if 'lstm' in col.lower()]
        hybrid_cols = [col for col in forecasts.columns if 'hybrid' in col.lower()]
        
        # Extract the asset tickers from the column names
        lstm_tickers = [col.split('_')[0] for col in lstm_cols]
        hybrid_tickers = [col.split('_')[0] for col in hybrid_cols]
        
        # Create dataframes with just the ticker names as columns
        lstm_forecasts = pd.DataFrame(index=forecasts.index)
        hybrid_forecasts = pd.DataFrame(index=forecasts.index)
        
        for i, ticker in enumerate(lstm_tickers):
            lstm_forecasts[ticker] = forecasts[lstm_cols[i]]
            
        for i, ticker in enumerate(hybrid_tickers):
            hybrid_forecasts[ticker] = forecasts[hybrid_cols[i]]
        
        return {
            'lstm': lstm_forecasts,
            'hybrid': hybrid_forecasts
        }
    except Exception as e:
        logger.warning(f"Failed to load volatility forecasts: {e}")
        return None

class PPOPortfolioEnv:
    """
    Environment for PPO portfolio optimization
    """
    def __init__(self, prices_df, risk_free_rate=0.02, window_size=20, max_steps=252, state_features=None):
        self.prices = prices_df
        self.returns = prices_df.pct_change().dropna()
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.max_steps = max_steps
        self.state_features = state_features
        self.reset()
        
    def reset(self):
        """Reset the environment to a random starting point"""
        # Choose a random starting point
        self.current_step = self.window_size
        max_start = len(self.returns) - self.max_steps - 1
        if max_start > self.window_size:
            self.current_step = np.random.randint(self.window_size, max_start)
        return self._get_state()
    
    def step(self, action):
        """Take an action (set portfolio weights) and observe next state and reward"""
        # Make sure weights sum to 1 (softmax the actions)
        weights = action / action.sum()
        
        # Calculate portfolio return for the current step
        step_return = (self.returns.iloc[self.current_step] * weights).sum()
        
        # Get risk-adjusted reward
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        reward = step_return - daily_rf
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.returns) - 1 or \
               self.current_step >= self.window_size + self.max_steps
               
        return self._get_state(), reward, done
    
    def _get_state(self):
        """Get the current state representation"""
        if self.state_features is not None:
            # Use pre-computed state features if available
            if self.current_step < len(self.state_features):
                return self.state_features[self.current_step]
            else:
                # Fall back to default state if we're beyond available features
                state = self.returns.iloc[self.current_step-self.window_size:self.current_step].values.flatten()
                
                # Safety check for NaN or infinite values
                if np.isnan(state).any() or np.isinf(state).any():
                    logger.warning("NaN or Inf values in state. Replacing with zeros.")
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                
                return state
        else:
            # Use the last window_size returns as the state
            state = self.returns.iloc[self.current_step-self.window_size:self.current_step].values.flatten()
            
            # Safety check for NaN or infinite values
            if np.isnan(state).any() or np.isinf(state).any():
                logger.warning("NaN or Inf values in state. Replacing with zeros.")
                state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            
            return state

def train_ppo(prices_df, state_features=None, learning_rate=3e-4, num_epochs=100, batch_size=64, clip_epsilon=0.2,
           entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5):
    """
    Train a PPO policy for portfolio optimization
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        state_features (numpy.ndarray): Pre-computed state features (optional)
        learning_rate (float): Learning rate for optimizer
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        clip_epsilon (float): PPO clipping parameter
        entropy_coef (float): Entropy coefficient for exploration
        value_loss_coef (float): Value loss coefficient
        max_grad_norm (float): Maximum gradient norm for clipping
        
    Returns:
        PPOPortfolioPolicy: Trained policy
    """
    logger.info("Training PPO portfolio policy...")
    
    # Prepare environment
    env = PPOPortfolioEnv(prices_df, state_features=state_features)
    
    # Determine state dimension from the environment's observation space
    state_dim = env.reset().shape[0]
    action_dim = len(prices_df.columns)
    
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create policy with the correct state dimension
    policy = PPOPolicy(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Sample trajectories
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        
        state = env.reset()
        done = False
        
        while not done:
            # Check for NaN in state
            if np.isnan(state).any() or np.isinf(state).any():
                logger.warning("NaN or Inf values in state. Replacing with zeros.")
                state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():  # Use no_grad to prevent gradient tracking during sampling
                dist = policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = policy.get_value(state_tensor)
            
            next_state, reward, done = env.step(action.cpu().numpy()[0])
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            
            if len(states) >= batch_size:
                break
        
        # Prepare batch tensors
        states_batch = torch.FloatTensor(np.array(states)).to(device)
        actions_batch = torch.cat(actions).to(device)
        old_log_probs_batch = torch.cat(log_probs).to(device)
        values_batch = torch.cat(values).to(device)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + (1 - dones[i]) * 0.99 * R
            returns.insert(0, R)
            
            advantage = R - values_batch[i].item()
            advantages.insert(0, advantage)
        
        returns_batch = torch.FloatTensor(returns).to(device)
        advantages_batch = torch.FloatTensor(advantages).to(device)
        
        # Normalize advantages - add safeguard against empty batches
        if len(advantages_batch) > 1:
            adv_mean = advantages_batch.mean()
            adv_std = advantages_batch.std() + 1e-8  # Add epsilon to prevent division by zero
            advantages_batch = (advantages_batch - adv_mean) / adv_std
        
        # Update policy
        for _ in range(10):  # Multiple optimization steps per batch
            # Get new distribution and values
            dist = policy(states_batch)
            new_log_probs = dist.log_prob(actions_batch)
            new_values = policy.get_value(states_batch).squeeze()
            entropy = dist.entropy().mean()
            
            # Compute ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            
            # Compute surrogate loss
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = ((new_values - returns_batch) ** 2).mean()
            
            # Total loss
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Policy Loss = {policy_loss.item():.4f}, Value Loss = {value_loss.item():.4f}")
    
    logger.info("PPO training completed")
    return policy

def run_ppo_optimization(prices_df, risk_free_rate=0.02, initial_value=10000):
    """
    Run PPO optimization using historical data
    
    Args:
        prices_df (pd.DataFrame): Historical price data for the assets
        risk_free_rate (float): Risk-free rate (annual)
        initial_value (float): Initial portfolio value
        
    Returns:
        dict: Dictionary with optimization results
    """
    logger.info("Running PPO optimization using historical data...")
    
    # Prepare state features for training
    state_features, scaler, feature_names = prepare_state_features(prices_df)
    
    # Train PPO policy
    policy = train_ppo(prices_df, state_features)
    
    # Create environment to get the state representation that matches training
    env = PPOPortfolioEnv(prices_df)
    state = env.reset()
    
    # IMPORTANT FIX: Ensure the state dimension matches what was used in training
    # Get the state dimension that was used for training the model
    state_dim = policy.actor[0].in_features
    logger.info(f"Model expects input dimension: {state_dim}, Current state dimension: {len(state)}")
    
    # If dimensions don't match, resize the state vector
    if len(state) != state_dim:
        logger.warning(f"State dimension mismatch. Resizing from {len(state)} to {state_dim}")
        if len(state) > state_dim:
            # Truncate if too large
            state = state[:state_dim]
        else:
            # Pad with zeros if too small
            padding = np.zeros(state_dim - len(state))
            state = np.concatenate([state, padding])
    
    # Ensure state is a tensor on the correct device
    latest_state = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # Get portfolio weights from trained policy
    weights = policy.get_weights(latest_state)[0]  # Get the first (and only) row
    
    # Convert to dictionary
    weights_dict = {ticker: weight for ticker, weight in zip(prices_df.columns, weights)}
    
    # Calculate expected return and volatility
    returns = prices_df.pct_change().dropna()
    expected_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Fix the portfolio return calculation to use proper indexing
    portfolio_return = np.sum(weights * expected_returns.values)
    
    # Fix the portfolio volatility calculation to use proper matrix multiplication
    portfolio_volatility = np.sqrt(weights @ cov_matrix.values @ weights)
    
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    logger.info(f"Expected annual return: {portfolio_return:.1%}")
    logger.info(f"Annual volatility: {portfolio_volatility:.1%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return {
        'weights': weights_dict,
        'expected_annual_return': portfolio_return,
        'expected_annual_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def run_vol_forecast_ppo(prices_df, risk_free_rate=0.02, vol_forecasts=None, forecast_model='hybrid', initial_value=10000):
    """
    Run PPO optimization with volatility forecasts
    
    Args:
        prices_df (pd.DataFrame): Historical price data for the assets
        risk_free_rate (float): Risk-free rate (annual)
        vol_forecasts (dict): Dictionary containing volatility forecasts
        forecast_model (str): Which volatility forecast model to use ('lstm' or 'hybrid')
        initial_value (float): Initial portfolio value
        
    Returns:
        dict: Dictionary with optimization results
    """
    logger.info(f"Running PPO optimization with {forecast_model} volatility forecasts...")
    
    # Check if we have volatility forecasts
    if vol_forecasts is None or forecast_model not in vol_forecasts:
        logger.warning(f"No valid {forecast_model} volatility forecasts. Falling back to standard PPO.")
        return run_ppo_optimization(prices_df, risk_free_rate, initial_value)
    
    forecasts = vol_forecasts[forecast_model]
    
    # Get the tickers we have in both our price data and forecasts
    common_tickers = list(set(prices_df.columns) & set(forecasts.columns))
    
    if not common_tickers:
        logger.warning("No common tickers between price data and forecasts. Falling back to standard PPO.")
        return run_ppo_optimization(prices_df, risk_free_rate, initial_value)
    
    # Filter to common tickers
    prices = prices_df[common_tickers]
    
    # Add volatility forecasts as additional features
    returns = prices.pct_change().dropna()
    historical_corr = returns.corr()
    
    # Prepare standard state features
    state_features, scaler, feature_names = prepare_state_features(prices)
    
    # Get the latest volatility forecasts
    latest_vol_forecasts = forecasts.iloc[-1:]
    
    # Create volatility forecast features
    vol_features = np.zeros(len(common_tickers))
    for i, ticker in enumerate(common_tickers):
        if ticker in latest_vol_forecasts:
            vol_features[i] = latest_vol_forecasts[ticker].iloc[0] * np.sqrt(252)  # Annualized
        else:
            vol_features[i] = returns[ticker].std() * np.sqrt(252)
    
    # Combine standard features with volatility features
    combined_features = np.column_stack([state_features, np.tile(vol_features, (len(state_features), 1))])
    
    # Train PPO policy with combined features
    policy = train_ppo(prices, combined_features)
    
    # Get latest state for prediction
    lookback = 30  # Default lookback value in prepare_state_features
    # Get at least 2*lookback+5 rows to ensure we have data after processing
    latest_standard_features, _, _ = prepare_state_features(prices.iloc[-(2*lookback+5):])
    
    # If we still don't have features, use a smaller subset of features from training
    if len(latest_standard_features) == 0:
        logger.warning("No features generated from recent data, using last features from training set")
        latest_standard_features = state_features[-1:]
    else:
        # Use only the last feature row
        latest_standard_features = latest_standard_features[-1:]
    
    # FIX: Create a properly sized combined state tensor with exactly the same dimension as during training
    combined_dim = combined_features.shape[1]  # This is the exact dimension used in training
    
    # First check if standard features have correct dimension (state_features.shape[1])
    if latest_standard_features[-1].shape[0] != state_features.shape[1]:
        logger.warning(f"Standard features dimension mismatch: got {latest_standard_features[-1].shape[0]}, expected {state_features.shape[1]}")
        # Pad or truncate to match expected dimensions
        if latest_standard_features[-1].shape[0] < state_features.shape[1]:
            # Pad with zeros
            padding = np.zeros(state_features.shape[1] - latest_standard_features[-1].shape[0])
            latest_standard_features[-1] = np.concatenate([latest_standard_features[-1], padding])
        else:
            # Truncate
            latest_standard_features[-1] = latest_standard_features[-1][:state_features.shape[1]]
    
    # Now construct the combined state with correct dimensions
    latest_state = np.concatenate([latest_standard_features[-1], vol_features])
    
    # IMPROVED FIX: Check against model's expected input size
    state_dim = policy.actor[0].in_features
    logger.info(f"Model expects input dimension: {state_dim}, Current state dimension: {latest_state.shape[0]}")
    
    if latest_state.shape[0] != state_dim:
        logger.warning(f"Total state dimension mismatch. Got: {latest_state.shape[0]}, Expected: {state_dim}")
        if latest_state.shape[0] < state_dim:
            # Pad with zeros to match the expected dimension
            padding = np.zeros(state_dim - latest_state.shape[0])
            latest_state = np.concatenate([latest_state, padding])
        else:
            # Truncate to match the expected dimension
            latest_state = latest_state[:state_dim]
    
    # Add batch dimension and convert to tensor
    latest_state_tensor = torch.FloatTensor(latest_state).unsqueeze(0).to(device)
    
    # Get portfolio weights from trained policy
    weights = policy.get_weights(latest_state_tensor)
    
    # Debug the shape of the weights array
    logger.info(f"Weights shape: {weights.shape}")
    logger.info(f"Number of common tickers: {len(common_tickers)}")
    logger.info(f"Common tickers: {common_tickers}")

    # Convert to dictionary with all original tickers (zeros for non-common tickers)
    weights_dict = {ticker: 0.0 for ticker in prices_df.columns}
    for i, ticker in enumerate(common_tickers):
        # Add safety check to prevent index errors
        if i < len(weights):
            weights_dict[ticker] = weights[i]
        else:
            logger.warning(f"Not enough weights for ticker {ticker} at index {i}, using 0.0")
            # Keep the default zero weight
    
    # Calculate the forecast covariance matrix using correlation and forecast volatilities
    forecast_cov = pd.DataFrame(index=common_tickers, columns=common_tickers)
    
    for i, ticker_i in enumerate(common_tickers):
        for j, ticker_j in enumerate(common_tickers):
            forecast_cov.loc[ticker_i, ticker_j] = historical_corr.loc[ticker_i, ticker_j] * vol_features[i] * vol_features[j]
    
    # Calculate portfolio metrics
    portfolio_weights = np.array([weights_dict[ticker] for ticker in common_tickers])
    expected_return_vector = np.array([returns[ticker].mean() * 252 for ticker in common_tickers])
    
    portfolio_return = portfolio_weights @ expected_return_vector
    portfolio_volatility = np.sqrt(portfolio_weights @ forecast_cov.values @ portfolio_weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    logger.info(f"Expected annual return: {portfolio_return:.1%}")
    logger.info(f"Annual volatility: {portfolio_volatility:.1%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return {
        'weights': weights_dict,
        'expected_annual_return': portfolio_return,
        'expected_annual_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def run_return_forecast_ppo(prices_df, risk_free_rate=0.02, initial_value=10000):
    """
    Run PPO optimization with categorical return forecasts
    
    Args:
        prices_df (pd.DataFrame): Historical price data for the assets
        risk_free_rate (float): Risk-free rate (annual)
        initial_value (float): Initial portfolio value
        
    Returns:
        dict: Dictionary with optimization results
    """
    logger.info(f"Running PPO optimization with categorical return forecasts...")
    
    # Load return forecasts
    forecast_file = 'results/forecasts/return_forecast.csv'
    try:
        forecasts_df = load_return_forecasts(forecast_file)
    except Exception as e:
        logger.warning(f"Failed to load return forecasts: {e}")
        logger.warning("Falling back to standard PPO.")
        return run_ppo_optimization(prices_df, risk_free_rate, initial_value)
    
    # Get latest available forecast date
    forecast_date = forecasts_df.index[-1]
    latest_forecasts = forecasts_df.loc[forecast_date]
    
    # Get the tickers we have in both our price data and forecasts
    common_tickers = list(set(prices_df.columns) & set(latest_forecasts.index))
    
    if not common_tickers:
        logger.warning("No common tickers between price data and forecasts. Falling back to standard PPO.")
        return run_ppo_optimization(prices_df, risk_free_rate, initial_value)
    
    # Filter to common tickers
    prices = prices_df[common_tickers]
    
    # Calculate returns for historical data
    returns = prices.pct_change().dropna()
    
    # Convert categorical forecasts to expected returns
    expected_returns = pd.Series(index=common_tickers)
    
    for ticker in common_tickers:
        ticker_returns = returns[ticker].dropna()
        if len(ticker_returns) > 10:  # Ensure we have enough data
            # Calculate quintile values (0=lowest, 4=highest)
            quintiles = [ticker_returns.quantile(i/5) for i in range(1, 6)]
            
            # Get the categorical forecast (0-4) and map to the appropriate quintile
            category = int(latest_forecasts[ticker])
            
            # Annualize the expected return (daily to annual)
            expected_returns[ticker] = quintiles[category] * 252
        else:
            # Fall back to historical mean if not enough data
            expected_returns[ticker] = returns[ticker].mean() * 252
    
    # Prepare standard state features
    state_features, scaler, feature_names = prepare_state_features(prices)
    
    # Add return forecast as additional feature
    return_forecast_features = np.zeros(len(common_tickers))
    for i, ticker in enumerate(common_tickers):
        return_forecast_features[i] = expected_returns[ticker]
    
    # Combine standard features with return forecast features
    combined_features = np.column_stack([state_features, np.tile(return_forecast_features, (len(state_features), 1))])
    
    # Get the dimension of our combined features
    combined_dim = combined_features.shape[1]
    logger.info(f"Training model with combined feature dimension: {combined_dim}")
    
    # Train PPO policy with combined features - this should create a model with the correct input size
    policy = train_ppo(prices, combined_features)
    
    # Get latest state for prediction
    lookback = 30  # Default lookback value in prepare_state_features
    # Get at least 2*lookback+5 rows to ensure we have data after processing
    latest_standard_features, _, _ = prepare_state_features(prices.iloc[-(2*lookback+5):])
    
    # If we still don't have features, use a smaller subset of features from training
    if len(latest_standard_features) == 0:
        logger.warning("No features generated from recent data, using last features from training set")
        latest_standard_features = state_features[-1:]
    else:
        # Use only the last feature row
        latest_standard_features = latest_standard_features[-1:]
    
    # FIX: Create a properly sized combined state tensor with exactly the same dimension as during training
    combined_dim = combined_features.shape[1]  # This is the exact dimension used in training
    
    # First check if standard features have correct dimension (state_features.shape[1])
    if latest_standard_features[-1].shape[0] != state_features.shape[1]:
        logger.warning(f"Standard features dimension mismatch: got {latest_standard_features[-1].shape[0]}, expected {state_features.shape[1]}")
        # Pad or truncate to match expected dimensions
        if latest_standard_features[-1].shape[0] < state_features.shape[1]:
            # Pad with zeros
            padding = np.zeros(state_features.shape[1] - latest_standard_features[-1].shape[0])
            latest_standard_features[-1] = np.concatenate([latest_standard_features[-1], padding])
        else:
            # Truncate
            latest_standard_features[-1] = latest_standard_features[-1][:state_features.shape[1]]
    
    # Now construct the combined state with correct dimensions
    latest_state = np.concatenate([latest_standard_features[-1], return_forecast_features])
    
    # IMPROVED FIX: Check directly against model's expected input size
    state_dim = policy.actor[0].in_features
    logger.info(f"Model expects input dimension: {state_dim}, Current state dimension: {latest_state.shape[0]}")
    
    if latest_state.shape[0] != state_dim:
        logger.warning(f"Total state dimension mismatch. Got: {latest_state.shape[0]}, Expected: {state_dim}")
        if latest_state.shape[0] < state_dim:
            # Pad with zeros to match the expected dimension
            padding = np.zeros(state_dim - latest_state.shape[0])
            latest_state = np.concatenate([latest_state, padding])
        else:
            # Truncate to match the expected dimension
            latest_state = latest_state[:state_dim]
    
    # Add batch dimension and convert to tensor
    latest_state_tensor = torch.FloatTensor(latest_state).unsqueeze(0).to(device)
    logger.info(f"Final tensor shape sent to model: {latest_state_tensor.shape}")
    
    # Get portfolio weights from trained policy
    weights = policy.get_weights(latest_state_tensor)
    
    # Debug the shape of the weights array
    logger.info(f"Weights shape: {weights.shape}")
    logger.info(f"Number of common tickers: {len(common_tickers)}")
    logger.info(f"Common tickers: {common_tickers}")

    # Convert to dictionary with all original tickers (zeros for non-common tickers)
    weights_dict = {ticker: 0.0 for ticker in prices_df.columns}

    # FIX: Extract the first row from weights if it's a 2D array
    if len(weights.shape) > 1:
        weights_array = weights[0]  # Take the first row
    else:
        weights_array = weights     # Already 1D

    for i, ticker in enumerate(common_tickers):
        # Add safety check to prevent index errors
        if i < len(weights_array):
            weights_dict[ticker] = weights_array[i].item() if hasattr(weights_array[i], 'item') else weights_array[i]
        else:
            logger.warning(f"Not enough weights for ticker {ticker} at index {i}, using 0.0")
            # Keep the default zero weight
    
    # Calculate portfolio metrics
    historical_cov = returns.cov() * 252  # Annualized
    
    # Portfolio metrics calculation
    portfolio_weights = np.array([weights_dict[ticker] for ticker in common_tickers])
    expected_return_vector = np.array([expected_returns[ticker] for ticker in common_tickers])
    
    portfolio_return = portfolio_weights @ expected_return_vector
    portfolio_volatility = np.sqrt(portfolio_weights @ historical_cov.values @ portfolio_weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    logger.info(f"Expected annual return: {portfolio_return:.1%}")
    logger.info(f"Annual volatility: {portfolio_volatility:.1%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return {
        'weights': weights_dict,
        'expected_annual_return': portfolio_return,
        'expected_annual_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def run_hybrid_forecast_ppo(prices_df, risk_free_rate=0.02, vol_forecasts=None, forecast_model='hybrid', initial_value=10000):
    """
    Run PPO optimization using both return forecasts and volatility forecasts
    
    Args:
        prices_df (pd.DataFrame): Historical price data for the assets
        risk_free_rate (float): Risk-free rate (annual)
        vol_forecasts (dict): Dictionary containing volatility forecasts
        forecast_model (str): Which volatility forecast model to use ('lstm' or 'hybrid')
        initial_value (float): Initial portfolio value
        
    Returns:
        dict: Dictionary with optimization results
    """
    logger.info(f"Running PPO optimization with combined return and volatility forecasts...")
    
    # ---- PART 1: GET RETURN FORECASTS ----
    # Load return forecasts
    forecast_file = 'results/forecasts/return_forecast.csv'
    try:
        forecasts_df = load_return_forecasts(forecast_file)
    except Exception as e:
        logger.warning(f"Failed to load return forecasts: {e}")
        logger.warning("Falling back to volatility forecast PPO.")
        return run_vol_forecast_ppo(prices_df, risk_free_rate, vol_forecasts, forecast_model, initial_value)
    
    # Get latest available forecast date
    forecast_date = forecasts_df.index[-1]
    latest_return_forecasts = forecasts_df.loc[forecast_date]
    
    # ---- PART 2: GET VOLATILITY FORECASTS ----
    # Check if we have volatility forecasts
    if vol_forecasts is None or forecast_model not in vol_forecasts:
        logger.warning(f"No valid volatility forecasts. Using only return forecasts.")
        return run_return_forecast_ppo(prices_df, risk_free_rate, initial_value)
    
    vol_forecast_df = vol_forecasts[forecast_model]
    
    # Find tickers we have in both our price data and both forecast types
    return_tickers = set(latest_return_forecasts.index)
    vol_tickers = set(vol_forecast_df.columns)
    price_tickers = set(prices_df.columns)
    
    common_tickers = list(price_tickers & return_tickers & vol_tickers)
    
    if not common_tickers:
        logger.warning("No common tickers between price data and both forecasts. Falling back to standard PPO.")
        return run_ppo_optimization(prices_df, risk_free_rate, initial_value)
    
    # Filter to common tickers
    prices = prices_df[common_tickers]
    
    # Calculate historical returns and correlation
    returns = prices.pct_change().dropna()
    historical_corr = returns.corr()
    
    # ---- PART 3: CONVERT CATEGORICAL RETURN FORECASTS TO EXPECTED RETURNS ----
    expected_returns = pd.Series(index=common_tickers)
    
    for ticker in common_tickers:
        ticker_returns = returns[ticker].dropna()
        if len(ticker_returns) > 10:  # Ensure we have enough data
            # Calculate quintile values (0=lowest, 4=highest)
            quintiles = [ticker_returns.quantile(i/5) for i in range(1, 6)]
            
            # Get the categorical forecast (0-4) and map to the appropriate quintile
            category = int(latest_return_forecasts[ticker])
            
            # Annualize the expected return (daily to annual)
            expected_returns[ticker] = quintiles[category] * 252
        else:
            # Fall back to historical mean if not enough data
            expected_returns[ticker] = returns[ticker].mean() * 252
    
    # ---- PART 4: GET VOLATILITY FORECASTS ----
    # Get the latest volatility forecasts
    latest_vol_forecasts = vol_forecast_df.iloc[-1:]
    
    # Create a dictionary of forecast volatilities
    forecast_vols = {}
    
    for ticker in common_tickers:
        # Get the most recent forecast for this ticker
        if ticker in latest_vol_forecasts:
            # Convert daily volatility to annual
            forecast_vols[ticker] = latest_vol_forecasts[ticker].iloc[0] * np.sqrt(252)
        else:
            # Fall back to historical volatility
            forecast_vols[ticker] = returns[ticker].std() * np.sqrt(252)
    
    # Prepare standard state features
    state_features, scaler, feature_names = prepare_state_features(prices)
    
    # Create combined forecast features
    return_features = np.array([expected_returns[ticker] for ticker in common_tickers])
    vol_features = np.array([forecast_vols[ticker] for ticker in common_tickers])
    
    # Combine all features
    combined_features = np.column_stack([
        state_features,
        np.tile(return_features, (len(state_features), 1)),
        np.tile(vol_features, (len(state_features), 1))
    ])
    
    # Train PPO policy with combined features
    policy = train_ppo(prices, combined_features)
    
    # Get latest state for prediction
    lookback = 30  # Default lookback value in prepare_state_features
    # Get at least 2*lookback+5 rows to ensure we have data after processing
    latest_standard_features, _, _ = prepare_state_features(prices.iloc[-(2*lookback+5):])
    
    # If we still don't have features, use a smaller subset of features from training
    if len(latest_standard_features) == 0:
        logger.warning("No features generated from recent data, using last features from training set")
        latest_standard_features = state_features[-1:]
    else:
        # Use only the last feature row
        latest_standard_features = latest_standard_features[-1:]
    
    # FIX: Create a properly sized combined state tensor with exactly the same dimension as during training
    combined_dim = combined_features.shape[1]  # This is the exact dimension used in training
    
    # First check if standard features have correct dimension (state_features.shape[1])
    if latest_standard_features[-1].shape[0] != state_features.shape[1]:
        logger.warning(f"Standard features dimension mismatch: got {latest_standard_features[-1].shape[0]}, expected {state_features.shape[1]}")
        # Pad or truncate to match expected dimensions
        if latest_standard_features[-1].shape[0] < state_features.shape[1]:
            # Pad with zeros
            padding = np.zeros(state_features.shape[1] - latest_standard_features[-1].shape[0])
            latest_standard_features[-1] = np.concatenate([latest_standard_features[-1], padding])
        else:
            # Truncate
            latest_standard_features[-1] = latest_standard_features[-1][:state_features.shape[1]]
    
    # Now construct the combined state with correct dimensions
    latest_state = np.concatenate([latest_standard_features[-1], return_features, vol_features])
    
    # IMPROVED FIX: Check directly against model's expected input size
    state_dim = policy.actor[0].in_features
    logger.info(f"Model expects input dimension: {state_dim}, Current state dimension: {latest_state.shape[0]}")
    
    if latest_state.shape[0] != state_dim:
        logger.warning(f"Total state dimension mismatch. Got: {latest_state.shape[0]}, Expected: {state_dim}")
        if latest_state.shape[0] < state_dim:
            # Pad with zeros to match the expected dimension
            padding = np.zeros(state_dim - latest_state.shape[0])
            latest_state = np.concatenate([latest_state, padding])
        else:
            # Truncate to match the expected dimension
            latest_state = latest_state[:state_dim]
    
    # Add batch dimension and convert to tensor
    latest_state_tensor = torch.FloatTensor(latest_state).unsqueeze(0).to(device)
    
    # Get portfolio weights from trained policy
    weights = policy.get_weights(latest_state_tensor)
    
    # Debug the shape of the weights array
    logger.info(f"Weights shape: {weights.shape}")
    logger.info(f"Number of common tickers: {len(common_tickers)}")
    logger.info(f"Common tickers: {common_tickers}")

    # Convert to dictionary with all original tickers (zeros for non-common tickers)
    weights_dict = {ticker: 0.0 for ticker in prices_df.columns}
    for i, ticker in enumerate(common_tickers):
        # Add safety check to prevent index errors
        if i < len(weights):
            weights_dict[ticker] = weights[i]
        else:
            logger.warning(f"Not enough weights for ticker {ticker} at index {i}, using 0.0")
            # Keep the default zero weight
    
    # Calculate the forecast covariance matrix using correlation and forecast volatilities
    forecast_cov = pd.DataFrame(index=common_tickers, columns=common_tickers)
    
    for i, ticker_i in enumerate(common_tickers):
        for j, ticker_j in enumerate(common_tickers):
            forecast_cov.loc[ticker_i, ticker_j] = historical_corr.loc[ticker_i, ticker_j] * forecast_vols[ticker_i] * forecast_vols[ticker_j]
    
    # Calculate portfolio metrics
    portfolio_weights = np.array([weights_dict[ticker] for ticker in common_tickers])
    expected_return_vector = np.array([expected_returns[ticker] for ticker in common_tickers])
    
    portfolio_return = portfolio_weights @ expected_return_vector
    portfolio_volatility = np.sqrt(portfolio_weights @ forecast_cov.values @ portfolio_weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    logger.info(f"Expected annual return: {portfolio_return:.1%}")
    logger.info(f"Annual volatility: {portfolio_volatility:.1%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return {
        'weights': weights_dict,
        'expected_annual_return': portfolio_return,
        'expected_annual_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def equal_weight_portfolio(prices_df, initial_value=10000):
    """
    Create an equal-weight portfolio
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        initial_value (float): Initial portfolio value
        
    Returns:
        dict: Dictionary with portfolio details
    """
    n_assets = len(prices_df.columns)
    weight = 1.0 / n_assets
    
    weights = {ticker: weight for ticker in prices_df.columns}
    
    # Calculate expected return and volatility
    returns = prices_df.pct_change().dropna()
    expected_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Calculate portfolio statistics
    portfolio_return = sum(weight * expected_returns[ticker] for ticker in prices_df.columns)
    
    # Calculate portfolio volatility (need to convert weights to numpy array in the same order as covariance matrix)
    weight_array = np.array([weight for _ in range(n_assets)])
    portfolio_volatility = np.sqrt(weight_array.T @ cov_matrix.values @ weight_array)
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02  # Default value
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return {
        'weights': weights,
        'expected_annual_return': portfolio_return,
        'expected_annual_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def print_portfolio_details(portfolio, name="Portfolio"):
    """
    Print the details of a portfolio
    
    Args:
        portfolio (dict): Portfolio dictionary with weights and metrics
        name (str): Name of the portfolio for display
    """
    print(f"\n--- {name} ---")
    print(f"Expected Annual Return: {portfolio['expected_annual_return']:.2%}")
    print(f"Expected Annual Volatility: {portfolio['expected_annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}")
    
    print("\nPortfolio Weights:")
    sorted_weights = sorted(portfolio['weights'].items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_weights:
        if weight > 0.01:  # Only show significant positions
            print(f"  {ticker}: {weight:.2%}")

def backtest_strategy(prices_df, weight_function, rebalance_freq='M', risk_free_rate=0.02):
    """
    Backtest a portfolio strategy
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        weight_function (callable): Function that returns portfolio weights
        rebalance_freq (str): Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
        risk_free_rate (float): Annual risk-free rate
        
    Returns:
        tuple: (backtest_results, weights_history)
    """
    logger.info(f"Backtesting with {rebalance_freq} rebalancing...")
    
    # Create a copy of the price data
    prices = prices_df.copy()
    
    # Create a DataFrame for the backtest results
    backtest_results = pd.DataFrame(index=prices.index)
    backtest_results['portfolio_value'] = 0.0
    
    # Initialize portfolio
    initial_value = 10000
    current_value = initial_value
    
    # Keep track of weights history
    weights_history = {}
    
    # Get rebalancing dates
    if rebalance_freq == 'D':
        rebalance_dates = prices.index
    else:
        rebalance_dates = prices.resample(rebalance_freq).last().index
    
    # Initialize with equal weights
    current_weights = {ticker: 1.0/len(prices.columns) for ticker in prices.columns}
    
    # For each rebalancing date
    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date not in prices.index:
            continue
            
        # Get the prices up to this date
        current_prices = prices.loc[:rebalance_date]
        
        # Calculate new weights based on the strategy
        try:
            strategy_result = weight_function(current_prices, current_value)
            new_weights = strategy_result['weights']
            weights_history[rebalance_date] = new_weights
        except Exception as e:
            logger.warning(f"Error calculating weights for {rebalance_date}: {e}")
            # Keep current weights if there's an error
            weights_history[rebalance_date] = current_weights
            new_weights = current_weights
            
        # If this is the first rebalancing, initialize the portfolio
        if i == 0:
            current_weights = new_weights
            
            # Set initial portfolio value
            backtest_results.loc[rebalance_date, 'portfolio_value'] = current_value
            continue
        
        # Get the previous rebalance date
        prev_rebalance_date = rebalance_dates[i-1] if i > 0 else prices.index[0]
        
        # Calculate returns since the last rebalance
        if prev_rebalance_date in prices.index:
            price_prev = prices.loc[prev_rebalance_date]
            price_curr = prices.loc[rebalance_date]
            
            # Calculate portfolio return based on current weights
            portfolio_return = sum(current_weights[ticker] * (price_curr[ticker] / price_prev[ticker] - 1) 
                                 for ticker in prices.columns if ticker in current_weights)
            
            # Update portfolio value
            current_value *= (1 + portfolio_return)
            
            # Update current weights
            current_weights = new_weights
            
            # Store the portfolio value
            backtest_results.loc[rebalance_date, 'portfolio_value'] = current_value
    
    # Forward-fill portfolio values
    backtest_results['portfolio_value'] = backtest_results['portfolio_value'].replace(0, np.nan).ffill()
    
    # Calculate daily returns
    backtest_results['daily_return'] = backtest_results['portfolio_value'].pct_change()
    
    return backtest_results, weights_history

def plot_backtest_comparison(backtest_results_list, strategy_names, save_path=None):
    """
    Plot a comparison of multiple backtest results
    
    Args:
        backtest_results_list (list): List of backtest result DataFrames
        strategy_names (list): List of strategy names
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for i, results in enumerate(backtest_results_list):
        # Calculate cumulative returns
        cumulative_returns = (1 + results['daily_return'].fillna(0)).cumprod() - 1
    plt.title('Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.close()

def load_return_forecasts(forecast_file='results/forecasts/return_forecast.csv'):
    """
    Load and process return forecast data from CSV
    
    Args:
        forecast_file (str): Path to the forecast CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing processed categorical return forecasts
    """
    logger.info(f"Loading return forecasts from {forecast_file}...")
    
    # Load the forecast data
    forecast_df = pd.read_csv(forecast_file, parse_dates=['Date'])
    forecast_df.set_index('Date', inplace=True)
    
    logger.info(f"Loaded return forecasts for {len(forecast_df.columns)} tickers")
    return forecast_df

def load_volatility_forecasts(forecast_file='results/forecasts/vol_forecasts.pkl'):
    """
    Load volatility forecasts from pickle file
    
    Args:
        forecast_file (str): Path to the forecast pickle file
        
    Returns:
        dict: Dictionary containing volatility forecasts
    """
    try:
        forecasts = pd.read_pickle(forecast_file)
        logger.info(f"Loaded volatility forecasts from {forecast_file}")
        return forecasts
    except Exception as e:
        logger.warning(f"Failed to load volatility forecasts: {e}")
        return None

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio values"""
    portfolio_values = portfolio_values.dropna()
    
    # If not enough data, return NaN
    if len(portfolio_values) < 2:
        return np.nan
        
    # Calculate running maximum and drawdown
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values / running_max - 1)
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    return max_drawdown

def ensure_model_dimension(state, policy):
    """
    Ensure the state has the correct dimension for the policy model
    
    Args:
        state (numpy.ndarray): Input state
        policy (nn.Module): Neural network policy
        
    Returns:
        torch.Tensor: Resized state tensor ready for the model
    """
    # Get expected dimension from model
    expected_dim = policy.actor[0].in_features
    
    # Check if dimensions match
    if len(state) != expected_dim:
        logger.warning(f"Dimension mismatch. Got: {len(state)}, Expected: {expected_dim}")
        if len(state) < expected_dim:
            # Pad with zeros
            padding = np.zeros(expected_dim - len(state))
            state = np.concatenate([state, padding])
        else:
            # Truncate
            state = state[:expected_dim]
            
    # Add batch dimension and convert to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    return state_tensor

def calculate_performance_metrics(returns, risk_free_rate=0.02, trading_days=252):
    # Add safety check to handle cases where returns are all NaN
    if returns.isna().all():
        logger.warning("All returns are NaN - cannot calculate performance metrics")
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    # Fill any NaN values at the beginning of the series with zeros
    # This can happen when calculating returns for the first period
    returns = returns.fillna(0)
    
    # Ensure we have valid data for calculations
    if len(returns) == 0:
        logger.warning("Empty returns series - cannot calculate performance metrics")
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod() - 1
    
    # Calculate total return
    total_return = cum_returns.iloc[-1] if not cum_returns.empty else 0
    
    # Calculate annualized return
    n_years = len(returns) / trading_days
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(trading_days)
    
    # Calculate Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    # Use rolling maximum to identify highest point so far
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / (1 + rolling_max)
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def calculate_portfolio_returns(prices, weights):
    # Ensure the weights sum to 1
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        logger.warning("Portfolio weights sum to zero or negative, using equal weights")
        weights = np.ones(len(weights)) / len(weights)
    
    # Calculate daily percentage changes
    daily_returns = prices.pct_change().fillna(0)
    
    # Multiply each asset's returns by its weight
    weighted_returns = daily_returns.multiply(weights, axis=1)
    
    # Sum across all assets to get portfolio return for each day
    portfolio_returns = weighted_returns.sum(axis=1)
    
    return portfolio_returns

def compare_strategies(results_dict, risk_free_rate=0.02):
    # Ensure all strategies have data for proper comparison
    for strategy, data in results_dict.items():
        if 'returns' not in data or data['returns'] is None:
            logger.warning(f"Missing returns data for {strategy} strategy")
            # Initialize with zeros to avoid NaN
            results_dict[strategy]['returns'] = pd.Series(0, index=next(
                (d['returns'].index for d in results_dict.values() if 'returns' in d and d['returns'] is not None), 
                pd.date_range(start='2000-01-01', periods=2)
            ))
    
    # ... rest of the function remains the same ...

def main():
    """Main function to run the portfolio optimization"""
    logger.info("Starting PPO portfolio optimization...")
    
    # Load configuration
    config = load_config()
    
    # Get parameters from config
    tickers = config.get('tickers', ['SPY', 'AGG', 'VT', 'BIL', 'VDE', 'DBC', 'IWN', 'AOM', 'AQMIX', '^VIX'])
    start_date = config.get('start_date', '2015-01-01')
    end_date = config.get('end_date', '2023-12-31')
    risk_free_rate = config.get('risk_free_rate', 0.02)
    
    # Load price data
    logger.info(f"Loading price data for {len(tickers)} tickers from {start_date} to {end_date}...")
    price_data, _ = data_loader.load_data(tickers, start_date, end_date)
    
    # Check for missing values
    missing_values = price_data.isna().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values detected in price data. Filling with forward fill.")
        price_data = price_data.ffill()
    
    # Load volatility forecast data
    vol_forecasts = load_volatility_forecasts()
    
    # Run PPO optimization
    logger.info("Running PPO optimization...")
    ppo_portfolio = run_ppo_optimization(price_data, risk_free_rate)
    
    # Create equal-weight portfolio for comparison
    equal_portfolio = equal_weight_portfolio(price_data)
    
    # Create forecast-enhanced PPO portfolios
    vol_forecast_portfolio = run_vol_forecast_ppo(price_data, risk_free_rate, vol_forecasts, 'hybrid')
    return_forecast_portfolio = run_return_forecast_ppo(price_data, risk_free_rate)
    hybrid_forecast_portfolio = run_hybrid_forecast_ppo(price_data, risk_free_rate, vol_forecasts, 'hybrid')
    
    # Print portfolio details
    print_portfolio_details(ppo_portfolio, "Base PPO Portfolio")
    print_portfolio_details(equal_portfolio, "Equal Weight Portfolio")
    print_portfolio_details(vol_forecast_portfolio, "Volatility Forecast PPO")
    print_portfolio_details(return_forecast_portfolio, "Return Forecast PPO")
    print_portfolio_details(hybrid_forecast_portfolio, "Hybrid Forecast PPO")
    
    # Backtest strategies
    ppo_backtest, ppo_weights = backtest_strategy(
        price_data, run_ppo_optimization, 'ME', risk_free_rate
    )
    
    equal_backtest, equal_weights = backtest_strategy(
        price_data, equal_weight_portfolio, 'ME', risk_free_rate
    )
    
    # Define a function to use volatility forecasts in the backtest
    def vol_forecast_weight_function(prices, initial_value=10000):
        return run_vol_forecast_ppo(prices, risk_free_rate, vol_forecasts, 'hybrid', initial_value)
    
    vol_forecast_backtest, vol_forecast_weights = backtest_strategy(
        price_data, vol_forecast_weight_function, 'ME', risk_free_rate
    )
    
    # Define a function to use return forecasts in the backtest
    def return_forecast_weight_function(prices, initial_value=10000):
        return run_return_forecast_ppo(prices, risk_free_rate, initial_value)
    
    return_forecast_backtest, return_forecast_weights = backtest_strategy(
        price_data, return_forecast_weight_function, 'ME', risk_free_rate
    )
    
    # Define a function to use hybrid forecasts in the backtest
    def hybrid_forecast_weight_function(prices, initial_value=10000):
        return run_hybrid_forecast_ppo(prices, risk_free_rate, vol_forecasts, 'hybrid', initial_value)
    
    hybrid_forecast_backtest, hybrid_forecast_weights = backtest_strategy(
        price_data, hybrid_forecast_weight_function, 'ME', risk_free_rate
    )
    
    # Plot portfolio value results
    plt.figure(figsize=(12, 8))
    plt.plot(ppo_backtest['portfolio_value'], label='Base PPO Portfolio')
    plt.plot(equal_backtest['portfolio_value'], label='Equal Weight')
    plt.plot(vol_forecast_backtest['portfolio_value'], label='Volatility Forecast PPO')
    plt.plot(return_forecast_backtest['portfolio_value'], label='Return Forecast PPO')
    plt.plot(hybrid_forecast_backtest['portfolio_value'], label='Hybrid Forecast PPO')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/ppo_portfolio_comparison.png')
    plt.close()
    
    # Plot cumulative returns comparison
    plot_backtest_comparison(
        [ppo_backtest, equal_backtest, vol_forecast_backtest, return_forecast_backtest, hybrid_forecast_backtest],
        ['Base PPO', 'Equal Weight', 'Volatility Forecast PPO', 'Return Forecast PPO', 'Hybrid Forecast PPO'],
        'results/ppo_cumulative_returns_comparison.png'
    )
    
    # Additional result analysis
    compare_strategies(ppo_backtest, equal_backtest, vol_forecast_backtest, 
                      return_forecast_backtest, hybrid_forecast_backtest, risk_free_rate)

if __name__ == "__main__":
    main()
    
    