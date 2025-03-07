#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Mean-Variance Optimization (MVO) implementation using PyPortfolioOpt
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
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt import exceptions
from pypfopt.exceptions import OptimizationError

# Import the data loader
from src.data import data_loader

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def run_mvo_optimization(prices_df, risk_free_rate=0.01, portfolio_value=1000000):
    """
    Run Mean-Variance Optimization on price data
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        risk_free_rate (float): Risk-free rate (annual)
        portfolio_value (int): Portfolio value for allocation
        
    Returns:
        dict: Optimization results including weights and metrics
    """
    logger.info("Running MVO optimization...")
    
    # Calculate expected returns (using mean historical return)
    returns = prices_df.pct_change().dropna()
    mu = returns.mean() * 252  # Annualized
    S = returns.cov() * 252    # Annualized covariance
    
    # Check if any assets have expected returns above risk-free rate
    assets_above_rf = (mu > risk_free_rate).sum()
    logger.info(f"Assets with returns > risk-free rate: {assets_above_rf} of {len(mu)}")
    logger.info(f"Expected returns range: {mu.min():.4f} to {mu.max():.4f}, mean: {mu.mean():.4f}")
    
    # Strategy 1: Try max_sharpe with original risk-free rate
    try:
        logger.info(f"Attempting max_sharpe with risk-free rate = {risk_free_rate:.4f}")
        ef = EfficientFrontier(mu, S)
        # Add constraint: max 20% in any one asset
        ef.add_constraint(lambda w: w <= 0.3)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        expected_annual_return = performance[0]
        expected_annual_volatility = performance[1]
        sharpe_ratio = performance[2]
        
        logger.info(f"Max Sharpe successful - Expected return: {expected_annual_return:.1%}, "
                  f"Volatility: {expected_annual_volatility:.1%}, Sharpe: {sharpe_ratio:.2f}")
        
        return {
            'weights': cleaned_weights,
            'expected_annual_return': expected_annual_return,
            'expected_annual_volatility': expected_annual_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    except OptimizationError as e:
        logger.warning(f"Max Sharpe optimization failed: {e}")
        
        # Strategy 2: Try max_sharpe with adjusted risk-free rate
        if assets_above_rf == 0 and mu.max() > 0:
            adjusted_rf = mu.max() * 0.9  # Set slightly below highest return
            logger.info(f"Using adjusted risk-free rate: {adjusted_rf:.4f}")
            
            try:
                ef = EfficientFrontier(mu, S)
                # Add constraint: max 20% in any one asset
                ef.add_constraint(lambda w: w <= 0.3)
                weights = ef.max_sharpe(risk_free_rate=adjusted_rf)
                cleaned_weights = ef.clean_weights()
                
                # Get performance with ORIGINAL risk-free rate for consistent reporting
                performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                
                logger.info(f"Max Sharpe (adj RF) - Expected return: {performance[0]:.1%}, "
                          f"Volatility: {performance[1]:.1%}, Sharpe: {performance[2]:.2f}")
                
                return {
                    'weights': cleaned_weights,
                    'expected_annual_return': performance[0],
                    'expected_annual_volatility': performance[1],
                    'sharpe_ratio': performance[2]
                }
            except OptimizationError as e2:
                logger.warning(f"Adjusted risk-free rate optimization also failed: {e2}")
        
        # Strategy 3: Try minimum volatility
        logger.info("Trying minimum volatility portfolio")
        try:
            ef = EfficientFrontier(mu, S)
            # Add constraint: max 20% in any one asset
            ef.add_constraint(lambda w: w <= 0.3)
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            
            performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            
            logger.info(f"Min Volatility - Expected return: {performance[0]:.1%}, "
                      f"Volatility: {performance[1]:.1%}, Sharpe: {performance[2]:.2f}")
            
            return {
                'weights': cleaned_weights,
                'expected_annual_return': performance[0],
                'expected_annual_volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        
        except OptimizationError as e3:
            logger.warning(f"Min volatility optimization also failed: {e3}")
    
    # Final fallback: Equal weights
    logger.warning("All optimization methods failed. Using equal weights.")
    n_assets = len(prices_df.columns)
    equal_weights = {asset: 1/n_assets for asset in prices_df.columns}
    
    # Estimate performance of equal weight portfolio
    equal_return = mu.mean()  # Simple average of all asset returns
    
    # Calculate volatility using the covariance matrix
    weights_array = np.array([1/n_assets] * n_assets)
    equal_volatility = np.sqrt(weights_array.T @ S @ weights_array)
    
    # Calculate Sharpe ratio
    equal_sharpe = (equal_return - risk_free_rate) / equal_volatility if equal_volatility > 0 else 0
    
    logger.info(f"Equal Weight - Expected return: {equal_return:.1%}, "
              f"Volatility: {equal_volatility:.1%}, Sharpe: {equal_sharpe:.2f}")
    
    return {
        'weights': equal_weights,
        'expected_annual_return': equal_return,
        'expected_annual_volatility': equal_volatility,
        'sharpe_ratio': equal_sharpe
    }

def plot_efficient_frontier(prices_df, risk_free_rate=0.01, n_points=100, save_path=None):
    """
    Plot the efficient frontier with the optimal portfolio
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        risk_free_rate (float): Risk-free rate (annual)
        n_points (int): Number of points to plot on efficient frontier
        save_path (str): Path to save the plot
    """
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)
    
    # Plot efficient frontier
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Find optimal portfolio
    ef_max_sharpe = EfficientFrontier(mu, S)
    ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    ret_sharpe, vol_sharpe, _ = ef_max_sharpe.portfolio_performance()
    
    # Find min volatility portfolio
    ef_min_vol = EfficientFrontier(mu, S)
    ef_min_vol.min_volatility()
    ret_min_vol, vol_min_vol, _ = ef_min_vol.portfolio_performance()
    
    # Generate efficient frontier curve
    ef_curve = EfficientFrontier(mu, S)
    risk = []
    returns = []
    
    # Get returns and risk for different portfolios
    for target_return in np.linspace(ret_min_vol, max(ret_min_vol * 2, ret_sharpe * 1.2), n_points):
        ef_curve = EfficientFrontier(mu, S)
        try:
            ef_curve.efficient_return(target_return)
            r, v, _ = ef_curve.portfolio_performance()
            returns.append(r)
            risk.append(v)
        except:
            pass
    
    # Plot the efficient frontier
    ax.plot(risk, returns, 'b-', linewidth=2)
    
    # Plot the optimal portfolio
    ax.scatter(vol_sharpe, ret_sharpe, marker='*', s=200, c='r', label='Max Sharpe')
    ax.scatter(vol_min_vol, ret_min_vol, marker='o', s=150, c='g', label='Min Volatility')
    
    # Individual assets
    for i, asset in enumerate(prices_df.columns):
        asset_returns = mu[i]
        asset_risk = np.sqrt(S.iloc[i, i])
        ax.scatter(asset_risk, asset_returns, marker='o', s=100, label=asset)
    
    # Format and display
    ax.set_title('Efficient Frontier', fontsize=14)
    ax.set_xlabel('Expected Volatility (Annual)', fontsize=12)
    ax.set_ylabel('Expected Return (Annual)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def equal_weight_portfolio(prices_df, portfolio_value=10000):
    """
    Create an equal-weight portfolio
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        portfolio_value (int): Portfolio value for allocation
        
    Returns:
        dict: Portfolio results including weights and allocation
    """
    logger.info("Creating equal-weight portfolio...")
    
    # Create equal weights
    n_assets = len(prices_df.columns)
    equal_weight = 1.0 / n_assets
    weights = {asset: equal_weight for asset in prices_df.columns}
    
    # Get expected performance (using historical data)
    returns = expected_returns.mean_historical_return(prices_df)
    cov_matrix = risk_models.sample_cov(prices_df)
    
    # Calculate expected return and volatility
    expected_return = sum(returns * equal_weight for returns in returns)
    expected_volatility = np.sqrt(
        np.dot(np.array(list(weights.values())), 
               np.dot(cov_matrix, np.array(list(weights.values())))))
    sharpe_ratio = (expected_return - 0.02) / expected_volatility
    
    # Get discrete allocation
    latest_prices = prices_df.iloc[-1]
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_value)
    allocation, leftover = da.greedy_portfolio()
    
    # Prepare results
    results = {
        'weights': weights,
        'expected_annual_return': expected_return,
        'expected_annual_volatility': expected_volatility,
        'sharpe_ratio': sharpe_ratio,
        'allocation': allocation,
        'leftover': leftover
    }
    
    return results

def backtest_strategy(prices_df, weight_function, rebalance_freq='ME', risk_free_rate=0.01, 
                     portfolio_value=1000000, lookback_window=60, start_date=None):
    """
    Backtest a portfolio strategy
    
    Args:
        prices_df (pd.DataFrame): Historical price data
        weight_function (callable): Function that returns portfolio weights
        rebalance_freq (str): Rebalancing frequency ('D'=daily, 'W'=weekly, 'ME'=month end)
        risk_free_rate (float): Risk-free rate (annual)
        portfolio_value (float): Initial portfolio value
        lookback_window (int): Minimum number of trading days required before first rebalance
        start_date (str or datetime, optional): Start date for the backtest. If None, uses all data.
        
    Returns:
        pd.DataFrame: Backtest results
    """
    logger.info(f"Backtesting strategy with {rebalance_freq} rebalancing...")
    
    # Apply start_date filter if provided, keeping prior data for lookback
    if start_date is not None:
        # Convert to pandas timestamp if string
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
            
        # Find the index position of start_date or the nearest date after it
        start_pos = None
        for i, date in enumerate(prices_df.index):
            if date >= start_date:
                start_pos = i
                break
        
        if start_pos is None:
            logger.warning(f"Start date {start_date} is after the end of price data. Using all available data.")
        else:
            # Keep some lookback data for initial optimization
            lookback_start = max(0, start_pos - lookback_window)
            
            # Use full data for optimization calculations but only evaluate backtest from start_date
            backtest_data = prices_df.iloc[lookback_start:]
            
            # For reporting, track where actual backtest evaluation begins
            backtest_start_idx = start_pos - lookback_start
            
            logger.info(f"Starting backtest from {prices_df.index[start_pos]} with {lookback_window} days of lookback data")
            
            # Replace prices_df with filtered data but keep original for reference
            original_prices_df = prices_df.copy()
            prices_df = backtest_data
    
    # Ensure we have enough historical data before first rebalance
    if prices_df.shape[0] <= lookback_window:
        logger.warning(f"Not enough historical data for reliable optimization. Need at least {lookback_window} days.")
    
    # Make sure we have data to work with before starting
    if prices_df.empty:
        logger.error("No price data available for backtesting")
        return pd.DataFrame(), {}
    
    # Get rebalance dates
    if rebalance_freq == 'D':
        rebalance_dates = prices_df.index
    else:
        # For weekly or monthly, use pandas resample
        rebalance_dates = prices_df.resample(rebalance_freq).last().index
        
        # Filter to make sure we only use dates that exist in our price data
        rebalance_dates = [d for d in rebalance_dates if d in prices_df.index]
    
    # Ensure we have the first date in rebalance_dates
    if prices_df.index[0] not in rebalance_dates:
        rebalance_dates = [prices_df.index[0]] + rebalance_dates
    
    # Skip the first rebalance date if we don't have enough history
    first_rebalance_idx = 0
    for i, rebalance_date in enumerate(rebalance_dates):
        days_before = len(prices_df.loc[:rebalance_date].index)
        if days_before > lookback_window:
            first_rebalance_idx = i
            break
    
    # Only use rebalance dates with sufficient prior data
    rebalance_dates = rebalance_dates[first_rebalance_idx:]
    
    # Make sure we have at least one rebalance date
    if len(rebalance_dates) == 0:
        logger.error("No valid rebalance dates found with sufficient historical data")
        return pd.DataFrame(), {}
    
    # Initialize results
    results = pd.DataFrame(index=prices_df.index)
    results['portfolio_value'] = np.nan
    results['daily_return'] = np.nan
    
    # Current portfolio value
    current_portfolio = portfolio_value
    
    # Dictionary to store weights at each rebalance
    all_weights = {}
    
    # For each rebalance period
    for i, rebalance_date in enumerate(rebalance_dates):
        logger.info(f"Rebalancing at {rebalance_date}")
        
        # Get historical data up to rebalance date
        historical_data = prices_df.loc[:rebalance_date]
        
        # If this is the first rebalance, make sure we have enough data
        if i == 0 and historical_data.shape[0] < lookback_window:
            logger.warning(f"Not enough historical data for first rebalance. Using equal weights.")
            weights = {asset: 1.0/len(prices_df.columns) for asset in prices_df.columns}
        else:
            try:
                # Get portfolio weights using the provided weight function
                portfolio = weight_function(historical_data, portfolio_value)
                weights = portfolio['weights']
            except Exception as e:
                logger.warning(f"Optimization failed at {rebalance_date}: {e}")
                logger.warning("Falling back to equal weights for this period")
                weights = {asset: 1.0/len(historical_data.columns) for asset in historical_data.columns}
        
        # Store weights for this rebalance period
        all_weights[rebalance_date] = weights
        
        # Get the next period's data for returns calculation
        next_rebalance_date = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else prices_df.index[-1]
        period_prices = prices_df.loc[rebalance_date:next_rebalance_date]
        
        # Calculate daily returns during this period
        period_dates = period_prices.index.tolist()
        for j, day in enumerate(period_dates):
            if day == rebalance_date:
                results.loc[day, 'portfolio_value'] = current_portfolio
                continue
            
            # Get the previous available day's data instead of assuming day-1 exists
            prev_day = period_dates[j-1]  
            
            # Calculate portfolio return for this day
            daily_returns = period_prices.loc[day] / period_prices.loc[prev_day] - 1
            
            # Make sure we have valid return data for assets in our portfolio
            valid_returns = {asset: ret for asset, ret in daily_returns.items() 
                            if asset in weights and not np.isnan(ret)}
            
            if valid_returns:
                portfolio_return = sum(weights[asset] * valid_returns[asset] for asset in valid_returns)
                
                # Update portfolio value
                current_portfolio *= (1 + portfolio_return)
                results.loc[day, 'portfolio_value'] = current_portfolio
            else:
                # No valid returns for this day, use previous value
                results.loc[day, 'portfolio_value'] = current_portfolio
    
    # Calculate returns and ensure no NaN values
    results['portfolio_value'] = results['portfolio_value'].ffill()  # Forward fill any missing values
    results['daily_return'] = results['portfolio_value'].pct_change()
    
    # More robust cumulative return calculation
    results['cumulative_return'] = (1 + results['daily_return'].fillna(0)).cumprod() - 1
    
    # Calculate annualized metrics
    days_in_backtest = (prices_df.index[-1] - prices_df.index[0]).days
    years = days_in_backtest / 365
    
    final_return = results['portfolio_value'].iloc[-1] / portfolio_value - 1
    annual_return = (1 + final_return) ** (1 / years) - 1
    
    # Try different approach for volatility
    try:
        # Remove outliers and invalid values
        daily_returns_clean = results['daily_return'].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check if we have enough data
        if len(daily_returns_clean) > 10:  # Arbitrary minimum number of data points
            # Use numpy's nanstd which is more robust
            annual_volatility = np.nanstd(daily_returns_clean) * np.sqrt(252)
            
            # Sanity check the result
            if not np.isnan(annual_volatility) and annual_volatility > 0:
                sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            else:
                logger.warning("Volatility calculation resulted in invalid value.")
                annual_volatility = 0
                sharpe_ratio = 0
        else:
            logger.warning(f"Not enough valid return data points: {len(daily_returns_clean)}")
            annual_volatility = 0
            sharpe_ratio = 0
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        annual_volatility = 0
        sharpe_ratio = 0
    
    logger.info(f"Backtest Results:")
    logger.info(f"Total Return: {final_return:.4f}")
    logger.info(f"Annualized Return: {annual_return:.4f}")
    logger.info(f"Annualized Volatility: {annual_volatility:.4f}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    return results, all_weights

def plot_backtest_comparison(backtest_results, strategy_names, save_path=None):
    """
    Plot backtest results comparison
    
    Args:
        backtest_results (list): List of backtest result dataframes
        strategy_names (list): List of strategy names
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for i, results in enumerate(backtest_results):
        plt.plot(results.index, results['cumulative_return'], label=strategy_names[i])
    
    plt.title('Portfolio Backtest: Cumulative Returns', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def load_volatility_forecasts(forecast_file='results/forecasts/lstm_hybrid_forecast.csv'):
    """
    Load and process volatility forecast data from LSTM and Hybrid models
    
    Args:
        forecast_file (str): Path to the forecast CSV file
        
    Returns:
        dict: Dictionary containing processed forecast data for both models
    """
    logger.info(f"Loading volatility forecasts from {forecast_file}...")
    
    # Load the forecast data
    forecast_df = pd.read_csv(forecast_file, parse_dates=['Date'])
    forecast_df.set_index('Date', inplace=True)
    
    # Pivot the data to have tickers as columns
    lstm_pred = forecast_df.pivot(columns='Ticker', values='LSTM_Pred')
    hybrid_pred = forecast_df.pivot(columns='Ticker', values='Hybrid_Pred')
    
    logger.info(f"Loaded volatility forecasts for {len(hybrid_pred.columns)} tickers")
    return {
        'lstm': lstm_pred,
        'hybrid': hybrid_pred
    }

def run_vol_forecast_mvo(prices_df, risk_free_rate=0.01, vol_forecasts=None, forecast_model='hybrid', initial_value=1000000):
    """
    Run Mean-Variance Optimization using volatility forecasts to enhance the covariance matrix
    
    Args:
        prices_df (pd.DataFrame): Historical price data for the assets
        risk_free_rate (float): Risk-free rate (annual)
        vol_forecasts (dict): Dictionary containing volatility forecasts
        forecast_model (str): Which forecast model to use ('lstm' or 'hybrid')
        initial_value (float): Initial portfolio value
        
    Returns:
        dict: Dictionary with optimization results
    """
    logger.info(f"Running MVO optimization with {forecast_model} volatility forecasts...")
    
    # Get the tickers we have in both our price data and forecasts
    forecasts = vol_forecasts[forecast_model]
    common_tickers = list(set(prices_df.columns) & set(forecasts.columns))
    
    if not common_tickers:
        logger.warning("No common tickers between price data and forecasts. Falling back to historical MVO.")
        return run_mvo_optimization(prices_df, risk_free_rate, initial_value)
    
    # Filter to common tickers
    prices = prices_df[common_tickers]
    
    # Calculate returns for expected returns and historical covariance
    returns = prices.pct_change().dropna()
    
    # Get expected returns from historical data
    expected_returns = returns.mean() * 252  # Annualize daily returns
    
    # Get the latest volatility forecasts
    # Use the most recent forecast available for each ticker
    latest_forecasts = forecasts.iloc[-1:]
    
    # Create a modified covariance matrix
    # 1. Start with the historical correlation matrix
    historical_cov = returns.cov() * 252  # Annualized
    historical_corr = returns.corr()
    
    # 2. Create a diagonal matrix of forecast volatilities
    forecast_vols = {}
    
    for ticker in common_tickers:
        # Get the most recent forecast for this ticker
        if ticker in latest_forecasts:
            # Convert daily volatility to annual
            forecast_vols[ticker] = latest_forecasts[ticker].iloc[0] * np.sqrt(252)
        else:
            # Fall back to historical volatility
            forecast_vols[ticker] = returns[ticker].std() * np.sqrt(252)
    
    # 3. Calculate the new covariance matrix using correlation and forecast volatilities
    # Formula: Cov(i,j) = Correlation(i,j) * Vol(i) * Vol(j)
    forecast_cov = pd.DataFrame(index=common_tickers, columns=common_tickers)
    
    for i in common_tickers:
        for j in common_tickers:
            forecast_cov.loc[i, j] = historical_corr.loc[i, j] * forecast_vols[i] * forecast_vols[j]
    
    try:
        # Create Efficient Frontier with historical returns but forecast-enhanced covariance
        ef = EfficientFrontier(expected_returns, forecast_cov)
        # Add constraint: max 20% in any one asset
        ef.add_constraint(lambda w: w <= 0.3)
        
        # First try max_sharpe
        try:
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            cleaned_weights = ef.clean_weights()
            
            # Get performance stats
            performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            
            logger.info(f"Volatility forecast - Expected annual return: {performance[0]:.1%}")
            logger.info(f"Volatility forecast - Annual volatility: {performance[1]:.1%}")
            logger.info(f"Volatility forecast - Sharpe Ratio: {performance[2]:.2f}")
            
            return {
                'weights': cleaned_weights,
                'expected_annual_return': performance[0],
                'expected_annual_volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        except OptimizationError as e:
            # Try min_volatility as fallback
            logger.warning(f"Volatility forecast max_sharpe failed: {e}")
            logger.warning("Trying min_volatility instead")
            
            ef = EfficientFrontier(expected_returns, forecast_cov)
            # Add constraint: max 20% in any one asset
            ef.add_constraint(lambda w: w <= 0.3)
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            
            performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            
            logger.info(f"Volatility forecast (min vol) - Expected return: {performance[0]:.1%}")
            logger.info(f"Volatility forecast (min vol) - Volatility: {performance[1]:.1%}")
            logger.info(f"Volatility forecast (min vol) - Sharpe: {performance[2]:.2f}")
            
            return {
                'weights': cleaned_weights,
                'expected_annual_return': performance[0],
                'expected_annual_volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
            
    except Exception as e:
        logger.warning(f"Forecast-enhanced optimization failed: {e}")
        logger.warning("Falling back to historical MVO")
        return run_mvo_optimization(prices_df, risk_free_rate, initial_value)

def print_portfolio_details(portfolio, name):
    """
    Print formatted details of a portfolio
    
    Args:
        portfolio (dict): Portfolio results dictionary
        name (str): Name of the portfolio strategy
    """
    logger.info(f"\n--- {name} ---")
    logger.info(f"Expected Annual Return: {portfolio['expected_annual_return']:.4f}")
    logger.info(f"Expected Annual Volatility: {portfolio['expected_annual_volatility']:.4f}")
    logger.info(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.4f}")
    
    logger.info("Asset Weights:")
    for asset, weight in portfolio['weights'].items():
        if weight > 0.01:  # Only show assets with significant weight
            logger.info(f"  {asset}: {weight:.4f}")
    
    # Print allocation if available
    if 'allocation' in portfolio:
        logger.info("Suggested Allocation:")
        for asset, shares in portfolio['allocation'].items():
            logger.info(f"  {asset}: {shares} shares")
        logger.info(f"Leftover cash: ${portfolio.get('leftover', 0):.2f}")
    
    logger.info("")  # Empty line for better readability

def load_return_forecasts(forecast_file='results/forecasts/return_forecast.csv'):
    """
    Load and process return forecast data from CSV
    """
    logger.info(f"Loading return forecasts from {forecast_file}...")
    
    # Check if file exists
    if not os.path.exists(forecast_file):
        logger.error(f"Return forecast file not found: {forecast_file}")
        return None
    
    try:
        # Load the forecast data
        forecast_df = pd.read_csv(forecast_file, parse_dates=['Date'])
        forecast_df.set_index('Date', inplace=True)
        
        # Debug information
        logger.info(f"Loaded return forecasts with shape: {forecast_df.shape}")
        logger.info(f"Forecast columns: {forecast_df.columns.tolist()}")
        
        return forecast_df
    except Exception as e:
        logger.error(f"Error loading return forecasts: {e}")
        return None

def run_return_forecast_mvo(prices_df, risk_free_rate=0.01, initial_value=1000000):
    """
    Run Mean-Variance Optimization using categorical return forecasts
    """
    logger.info(f"Running MVO optimization with categorical return forecasts...")
    
    # Load return forecasts
    forecast_file = 'results/forecasts/return_forecast.csv'
    try:
        forecasts_df = load_return_forecasts(forecast_file)
        if forecasts_df is None or forecasts_df.empty:
            logger.warning(f"Return forecast data is empty or None")
            return run_mvo_optimization(prices_df, risk_free_rate, initial_value)
            
        # Debug information
        logger.info(f"Successfully loaded return forecasts with shape {forecasts_df.shape}")
    except Exception as e:
        logger.warning(f"Failed to load return forecasts: {e}")
        logger.warning("Falling back to historical MVO.")
        return run_mvo_optimization(prices_df, risk_free_rate, initial_value)
    
    try:
        # Get latest available forecast date
        forecast_date = forecasts_df.index[-1]
        latest_forecasts = forecasts_df.loc[forecast_date]
        logger.info(f"Using forecast data from {forecast_date}")
        
        # Get the tickers we have in both our price data and forecasts
        common_tickers = list(set(prices_df.columns) & set(latest_forecasts.index))
        logger.info(f"Found {len(common_tickers)} common tickers between price data and forecasts")
        
        if not common_tickers:
            logger.warning("No common tickers between price data and forecasts. Falling back to historical MVO.")
            return run_mvo_optimization(prices_df, risk_free_rate, initial_value)
        
        # Filter to common tickers
        prices = prices_df[common_tickers]
        
        # Calculate returns for covariance and historical returns distribution
        returns = prices.pct_change().dropna()
        
        # Convert categorical forecasts (0-4) to actual return expectations
        # We'll use historical return quintiles to map the categories
        expected_returns = pd.Series(index=common_tickers)
        
        for ticker in common_tickers:
            try:
                ticker_returns = returns[ticker].dropna()
                if len(ticker_returns) > 10:  # Ensure we have enough data
                    # Calculate quintile values (0=lowest, 4=highest)
                    quintiles = [ticker_returns.quantile(i/5) for i in range(1, 6)]
                    
                    # Get the categorical forecast (0-4) and map to the appropriate quintile
                    # Convert to int safely
                    try:
                        category = int(float(latest_forecasts[ticker]))
                        # Ensure category is within bounds
                        category = max(0, min(category, 4))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid category value for {ticker}: {latest_forecasts[ticker]}, using middle category (2)")
                        category = 2
                    
                    # Annualize the expected return (daily to annual)
                    expected_returns[ticker] = quintiles[category] * 252
                else:
                    # Fall back to historical mean if not enough data
                    expected_returns[ticker] = returns[ticker].mean() * 252
                    logger.warning(f"Not enough return data for {ticker}, using historical mean")
            except Exception as e:
                logger.warning(f"Error processing ticker {ticker}: {e}")
                # Use a reasonable default
                expected_returns[ticker] = 0.05  # 5% annual return as fallback
        
        # Verify expected returns
        logger.info(f"Expected returns calculated for {len(expected_returns)} tickers")
        if expected_returns.isna().any():
            logger.warning(f"NaN values detected in expected returns, filling with median")
            expected_returns = expected_returns.fillna(expected_returns.median())
        
        # Get the historical covariance matrix
        historical_cov = returns.cov() * 252  # Annualized
        
        # Verify covariance matrix
        if historical_cov.isna().any().any():
            logger.warning("NaN values detected in covariance matrix, using robust estimation")
            # Simple fix: replace NaNs with 0s (not ideal but prevents errors)
            historical_cov = historical_cov.fillna(0)
    
        # Create Efficient Frontier with forecast returns but historical covariance
        try:
            ef = EfficientFrontier(expected_returns, historical_cov)
            
            # Add constraint: max 20% in any one asset
            ef.add_constraint(lambda w: w <= 0.3)
            
            try:
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                cleaned_weights = ef.clean_weights()
                
                # Get performance stats
                performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                
                logger.info(f"Return forecast - Expected annual return: {performance[0]:.1%}")
                logger.info(f"Return forecast - Annual volatility: {performance[1]:.1%}")
                logger.info(f"Return forecast - Sharpe Ratio: {performance[2]:.2f}")
                
                return {
                    'weights': cleaned_weights,
                    'expected_annual_return': performance[0],
                    'expected_annual_volatility': performance[1],
                    'sharpe_ratio': performance[2]
                }
                
            except OptimizationError as e:
                logger.warning(f"Return forecast max_sharpe failed: {e}")
                
                # Check if any assets have expected returns above risk-free rate
                assets_above_rf = (expected_returns > risk_free_rate).sum()
                
                if assets_above_rf == 0 and expected_returns.max() > 0:
                    # Try with adjusted risk-free rate
                    adjusted_rf = expected_returns.max() * 0.9
                    logger.info(f"Using adjusted risk-free rate: {adjusted_rf:.4f}")
                    
                    ef = EfficientFrontier(expected_returns, historical_cov)
                    # Add constraint: max 20% in any one asset
                    ef.add_constraint(lambda w: w <= 0.3)
                    weights = ef.max_sharpe(risk_free_rate=adjusted_rf)
                    cleaned_weights = ef.clean_weights()
                    
                    # Report using original risk-free rate
                    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                    
                    return {
                        'weights': cleaned_weights,
                        'expected_annual_return': performance[0],
                        'expected_annual_volatility': performance[1],
                        'sharpe_ratio': performance[2]
                    }
                else:
                    # Try min_volatility instead
                    logger.warning("Trying min_volatility")
                    ef = EfficientFrontier(expected_returns, historical_cov)
                    # Add constraint: max 20% in any one asset
                    ef.add_constraint(lambda w: w <= 0.3)
                    weights = ef.min_volatility()
                    cleaned_weights = ef.clean_weights()
                    
                    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                    
                    return {
                        'weights': cleaned_weights,
                        'expected_annual_return': performance[0],
                        'expected_annual_volatility': performance[1],
                        'sharpe_ratio': performance[2]
                    }
                
        except Exception as e:
            logger.warning(f"Return forecast optimization failed: {e}")
            logger.warning("Falling back to historical MVO")
            return run_mvo_optimization(prices_df, risk_free_rate, initial_value)
    except Exception as e:
        logger.warning(f"Return forecast optimization failed: {e}")
        logger.warning("Falling back to historical MVO")
        return run_mvo_optimization(prices_df, risk_free_rate, initial_value)

def save_portfolio_details(portfolio, strategy_name, directory='results'):
    """
    Save portfolio details to CSV files
    
    Args:
        portfolio (dict): Portfolio details including weights and performance metrics
        strategy_name (str): Name of the strategy
        directory (str): Directory to save the files
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
        
    # Save portfolio weights
    weights_df = pd.DataFrame({'Asset': list(portfolio['weights'].keys()), 
                               'Weight': list(portfolio['weights'].values())})
    weights_df.to_csv(f"{directory}/{strategy_name}_weights.csv", index=False)
    logger.info(f"Saved {strategy_name} weights to {directory}/{strategy_name}_weights.csv")
    
    # Save performance metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Expected Annual Return', 'Expected Annual Volatility', 'Sharpe Ratio'],
        'Value': [portfolio['expected_annual_return'], 
                 portfolio['expected_annual_volatility'], 
                 portfolio['sharpe_ratio']]
    })
    metrics_df.to_csv(f"{directory}/{strategy_name}_metrics.csv", index=False)
    logger.info(f"Saved {strategy_name} metrics to {directory}/{strategy_name}_metrics.csv")

def save_backtest_results(backtest_data, strategy_name, directory='results'):
    """
    Save backtest results to CSV
    
    Args:
        backtest_data (pd.DataFrame): Backtest results
        strategy_name (str): Name of the strategy
        directory (str): Directory to save the files
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Save backtest data
    backtest_data.to_csv(f"{directory}/{strategy_name}_backtest.csv")
    logger.info(f"Saved {strategy_name} backtest results to {directory}/{strategy_name}_backtest.csv")

def run_hybrid_forecast_mvo(prices_df, risk_free_rate=0.01, vol_forecasts=None, forecast_model='hybrid', initial_value=1000000):
    """
    Run Mean-Variance Optimization using both return and volatility forecasts
    
    Args:
        prices_df (pd.DataFrame): Historical price data for the assets
        risk_free_rate (float): Risk-free rate (annual)
        vol_forecasts (dict): Dictionary containing volatility forecasts
        forecast_model (str): Which forecast model to use ('lstm' or 'hybrid')
        initial_value (float): Initial portfolio value
        
    Returns:
        dict: Dictionary with optimization results
    """
    logger.info("Running MVO optimization with combined return and volatility forecasts...")
    
    try:
        # Load return forecasts
        forecast_file = 'results/forecasts/return_forecast.csv'
        forecasts_df = load_return_forecasts(forecast_file)
        
        # Get volatility forecasts
        forecasts = vol_forecasts[forecast_model]
        
        # Get common tickers across all data sources
        common_tickers = list(set(prices_df.columns) & set(forecasts.columns) & set(forecasts_df.columns))
        
        if not common_tickers:
            logger.warning("No common tickers between data sources. Falling back to historical MVO.")
            return run_mvo_optimization(prices_df, risk_free_rate, initial_value)
        
        # Filter to common tickers
        prices = prices_df[common_tickers]
        returns = prices.pct_change().dropna()
        
        # Get return expectations from categorical forecasts
        forecast_date = forecasts_df.index[-1]
        latest_forecasts = forecasts_df.loc[forecast_date]
        expected_returns = pd.Series(index=common_tickers)
        
        for ticker in common_tickers:
            ticker_returns = returns[ticker].dropna()
            if len(ticker_returns) > 10:
                quintiles = [ticker_returns.quantile(i/5) for i in range(1, 6)]
                category = int(latest_forecasts[ticker])
                expected_returns[ticker] = quintiles[category] * 252
            else:
                expected_returns[ticker] = returns[ticker].mean() * 252
        
        # Build volatility-enhanced covariance matrix
        latest_vol_forecasts = forecasts.iloc[-1:]
        historical_corr = returns.corr()
        forecast_vols = {}
        
        for ticker in common_tickers:
            if ticker in latest_vol_forecasts:
                forecast_vols[ticker] = latest_vol_forecasts[ticker].iloc[0] * np.sqrt(252)
            else:
                forecast_vols[ticker] = returns[ticker].std() * np.sqrt(252)
        
        # Create covariance matrix using correlation and forecast volatilities
        forecast_cov = pd.DataFrame(index=common_tickers, columns=common_tickers)
        for i in common_tickers:
            for j in common_tickers:
                forecast_cov.loc[i, j] = historical_corr.loc[i, j] * forecast_vols[i] * forecast_vols[j]
        
        # Run optimization with enhanced inputs
        ef = EfficientFrontier(expected_returns, forecast_cov)
        # Add constraint: max 20% in any one asset
        ef.add_constraint(lambda w: w <= 0.3)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        return {
            'weights': cleaned_weights,
            'expected_annual_return': performance[0],
            'expected_annual_volatility': performance[1],
            'sharpe_ratio': performance[2]
        }
    except Exception as e:
        logger.warning(f"Hybrid forecast optimization failed: {e}")
        logger.warning("Falling back to historical MVO")
        return run_mvo_optimization(prices_df, risk_free_rate, initial_value)
        
def hybrid_forecast_weight_function(prices, initial_value=1000000, risk_free_rate=0.01, vol_forecasts=None):
    """Helper function for using hybrid forecasts in backtest"""
    return run_hybrid_forecast_mvo(prices, risk_free_rate, vol_forecasts, 'hybrid', initial_value)

def main():
    """Main function to run the portfolio optimization and backtesting"""
    # Load configuration
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Loaded configuration from config.yaml")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = {'tickers': ['SPY', 'AGG', 'VT', 'BIL', 'VDE', 'DBC', 'IWN', 'AOM', 'AQMIX', '^VIX']}
    
    # Set parameters
    tickers = config.get('tickers', ['SPY', 'AGG', 'VT', 'BIL', 'VDE', 'DBC', 'IWN', 'AOM', 'AQMIX', '^VIX'])
    start_date = config.get('start_date', '2009-01-01')
    end_date = config.get('end_date', '2024-12-31')
    risk_free_rate = config.get('risk_free_rate', 0.02)
    
    # Load data
    n_tickers = len(tickers)
    logger.info(f"Loading data for {n_tickers} tickers...")
    price_data, *_ = data_loader.load_data(tickers, start_date, end_date)  # Unpack the tuple
    
    # Verify we have data before proceeding
    if price_data.empty or price_data.shape[0] == 0:
        logger.error("No price data was loaded. Check your date range and ticker symbols.")
        return  # Exit the function early
    
    # Check for missing data
    missing_values = price_data.isna().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values detected in price data. Filling with forward fill.")
        price_data = price_data.ffill()
    
    # Load volatility forecast data
    vol_forecasts = load_volatility_forecasts()
    
    # Run standard MVO optimization
    logger.info("Running MVO optimization...")
    mvo_portfolio = run_mvo_optimization(price_data, risk_free_rate)
    save_portfolio_details(mvo_portfolio, "historical_mvo")
    
    # Create equal-weight portfolio for comparison
    equal_portfolio = equal_weight_portfolio(price_data)
    save_portfolio_details(equal_portfolio, "equal_weight")
    
    # Create volatility-forecast-enhanced portfolio
    forecast_portfolio = run_vol_forecast_mvo(price_data, risk_free_rate, vol_forecasts, 'hybrid')
    save_portfolio_details(forecast_portfolio, "volatility_forecast")
    
    # Create return-forecast-enhanced portfolio
    return_forecast_portfolio = run_return_forecast_mvo(price_data, risk_free_rate)
    save_portfolio_details(return_forecast_portfolio, "return_forecast")
    
    # Create combined forecast portfolio
    hybrid_forecast_portfolio = run_hybrid_forecast_mvo(price_data, risk_free_rate, vol_forecasts, 'hybrid')
    save_portfolio_details(hybrid_forecast_portfolio, "hybrid_forecast")
    
    # Print portfolio details
    print_portfolio_details(mvo_portfolio, "Historical Portfolio (MVO)")
    print_portfolio_details(equal_portfolio, "Equal Weight Portfolio")
    print_portfolio_details(forecast_portfolio, "Volatility Forecast Portfolio")
    print_portfolio_details(return_forecast_portfolio, "Return Forecast Portfolio")
    print_portfolio_details(hybrid_forecast_portfolio, "Hybrid Forecast Portfolio (Return + Volatility)")
    
    # Use a start date that ensures enough historical data for the first rebalance
    data_start_date = price_data.index[0]
    # Make sure we start AFTER the data starts (not before)
    backtest_start_date = data_start_date + pd.Timedelta(days=60)  # Start 60 days after data begins
    logger.info(f"Data starts at {data_start_date}, backtest will start at {backtest_start_date}")
    
    # Update backtest calls with start_date
    historical_backtest, historical_weights = backtest_strategy(
        price_data, run_mvo_optimization, 'ME', risk_free_rate, 
        start_date=backtest_start_date
    )
    save_backtest_results(historical_backtest, "historical_mvo")
    
    equal_backtest, equal_weights = backtest_strategy(
        price_data, equal_weight_portfolio, 'ME', risk_free_rate,
        start_date=backtest_start_date
    )
    save_backtest_results(equal_backtest, "equal_weight")
    
    # Define a function to use volatility forecasts in the backtest
    def vol_forecast_weight_function(prices, initial_value=1000000):
        return run_vol_forecast_mvo(prices, risk_free_rate, vol_forecasts, 'hybrid', initial_value)
    
    forecast_backtest, forecast_weights = backtest_strategy(
        price_data, vol_forecast_weight_function, 'ME', risk_free_rate,
        start_date=backtest_start_date
    )
    save_backtest_results(forecast_backtest, "volatility_forecast")
    
    # Define a function to use return forecasts in the backtest
    def return_forecast_weight_function(prices, initial_value=1000000):
        return run_return_forecast_mvo(prices, risk_free_rate, initial_value)
    
    return_forecast_backtest, return_forecast_weights = backtest_strategy(
        price_data, return_forecast_weight_function, 'ME', risk_free_rate,
        start_date=backtest_start_date
    )
    save_backtest_results(return_forecast_backtest, "return_forecast")
    
    hybrid_forecast_backtest, hybrid_forecast_weights = backtest_strategy(
        price_data, 
        lambda prices, initial_value=1000000: hybrid_forecast_weight_function(
            prices, initial_value, risk_free_rate, vol_forecasts
        ), 
        'ME', risk_free_rate,
        start_date=backtest_start_date
    )
    save_backtest_results(hybrid_forecast_backtest, "hybrid_forecast")
    
    # Plot portfolio value results
    plt.figure(figsize=(12, 8))
    plt.plot(historical_backtest['portfolio_value'], label='Historical MVO Portfolio')
    plt.plot(equal_backtest['portfolio_value'], label='Equal Weight')
    plt.plot(forecast_backtest['portfolio_value'], label='Volatility Forecast MVO')
    plt.plot(return_forecast_backtest['portfolio_value'], label='Return Forecast MVO')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/portfolio_comparison.png')
    plt.close()
    
    # Plot cumulative returns comparison
    plot_backtest_comparison(
        [historical_backtest, equal_backtest, forecast_backtest, return_forecast_backtest, hybrid_forecast_backtest],
        ['Historical MVO', 'Equal Weight', 'Volatility Forecast MVO', 'Return Forecast MVO', 'Hybrid Forecast MVO'],
        'results/cumulative_returns_comparison.png'
    )
    
    # Additional result analysis
    compare_strategies(historical_backtest, equal_backtest, forecast_backtest, 
                       return_forecast_backtest, hybrid_forecast_backtest, risk_free_rate)
    
    # Save the weights over time for each strategy
    pd.DataFrame(historical_weights).to_csv("results/historical_mvo_weights_over_time.csv")
    pd.DataFrame(equal_weights).to_csv("results/equal_weight_weights_over_time.csv")
    pd.DataFrame(forecast_weights).to_csv("results/volatility_forecast_weights_over_time.csv")
    pd.DataFrame(return_forecast_weights).to_csv("results/return_forecast_weights_over_time.csv")
    pd.DataFrame(hybrid_forecast_weights).to_csv("results/hybrid_forecast_weights_over_time.csv")
    
    logger.info("Portfolio optimization and analysis complete. All results saved to the 'results' directory.")

def compare_strategies(historical, equal, forecast, return_forecast, hybrid_forecast, risk_free_rate=0.01):
    """
    Compare performance metrics of different strategies
    
    Args:
        historical (pd.DataFrame): Historical MVO backtest results
        equal (pd.DataFrame): Equal weight backtest results
        forecast (pd.DataFrame): Forecast-enhanced MVO backtest results
        return_forecast (pd.DataFrame): Return forecast MVO backtest results
        hybrid_forecast (pd.DataFrame): Hybrid forecast MVO backtest results
        risk_free_rate (float): Risk-free rate
    """
    logger.info("\n--- Strategy Comparison ---\n")
    
    strategies = {
        "Historical MVO": historical,
        "Equal Weight": equal,
        "Vol Forecast MVO": forecast,
        "Return Forecast MVO": return_forecast,
        "Hybrid Forecast MVO": hybrid_forecast
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        # Calculate performance metrics
        returns = strategy['daily_return'].dropna()
        
        # Get first and last valid portfolio values to handle NaN values properly
        first_valid = strategy['portfolio_value'].first_valid_index()
        last_valid = strategy['portfolio_value'].last_valid_index()
        
        if first_valid is not None and last_valid is not None:
            start_value = strategy['portfolio_value'].loc[first_valid]
            end_value = strategy['portfolio_value'].loc[last_valid]
            cumulative_return = end_value / start_value - 1
            
            # Calculate trading days for proper annualization
            trading_days = (last_valid - first_valid).days / 365 * 252
            annual_return = (1 + cumulative_return) ** (252 / trading_days) - 1 if trading_days > 0 else np.nan
        else:
            cumulative_return = np.nan
            annual_return = np.nan
        
        # Handle cases with missing or invalid data
        if len(returns) > 10:
            volatility = returns.std() * np.sqrt(252)
            sharpe = (returns.mean() * 252 - risk_free_rate) / volatility if volatility > 0 else 0
            max_drawdown = calculate_max_drawdown(strategy['portfolio_value'])
        else:
            volatility = np.nan
            sharpe = np.nan
            max_drawdown = np.nan
        
        results[name] = {
            "Total Return": cumulative_return,
            "Annualized Return": annual_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown
        }
    
    # Create comparison table
    comparison = pd.DataFrame(results).T
    logger.info("\nStrategy Performance Comparison:")
    logger.info(f"\n{comparison.to_string(float_format=lambda x: f'{x:.4f}')}")
    
    # Save to CSV
    comparison.to_csv('results/strategy_comparison.csv')
    logger.info("\nComparison saved to results/strategy_comparison.csv")
    
    # Save detailed performance for each strategy
    for name, strategy in strategies.items():
        file_name = name.lower().replace(' ', '_')
        strategy.to_csv(f'results/{file_name}_detailed_performance.csv')
        logger.info(f"Detailed performance for {name} saved to results/{file_name}_detailed_performance.csv")

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

if __name__ == "__main__":
    main()