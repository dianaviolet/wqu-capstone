#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for portfolio optimization pipeline.
This script orchestrates the entire process:
1. Generate forecasts (if needed)
2. Create portfolio strategies
3. Compare and evaluate strategies
"""

import os
import argparse
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Import forecast modules
from src.models.categorical_returns_forecast import run_categorical_returns_forecast
from src.models.volatility_forecast import run_volatility_forecast

# Import portfolio optimization modules
from src.models.mvo_portfolio import (
    run_mvo, run_vol_forecast_mvo, run_return_forecast_mvo, 
    run_hybrid_forecast_mvo, equal_weight_portfolio,
    save_portfolio_details, compare_strategies as mvo_compare
)
from src.models.ppo_portfolio import (
    run_ppo, run_vol_forecast_ppo, run_return_forecast_ppo,
    run_hybrid_forecast_ppo, equal_weight_portfolio as ppo_equal_weight,
    save_portfolio_details as ppo_save_details
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def ensure_directories(directories):
    """Ensure all required directories exist"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def forecasts_exist(forecast_dir='results/forecasts'):
    """Check if all required forecast files exist"""
    required_files = [
        'return_forecast.csv',
        'volatility_forecast_metrics.csv',
        'volatility_forecasts.csv',
        'return_class_forecasts.csv',
        'return_forecasts_future.csv'
    ]
    
    for file in required_files:
        file_path = os.path.join(forecast_dir, file)
        if not os.path.exists(file_path):
            logger.info(f"Missing forecast file: {file_path}")
            return False
    
    logger.info("All forecast files found, skipping forecast generation")
    return True

def generate_forecasts(config_path='config.yaml', forecast_dir='results/forecasts', 
                       retrain_returns=False, retrain_vol=False):
    """Generate returns and volatility forecasts if needed"""
    # Ensure forecast directory exists
    os.makedirs(forecast_dir, exist_ok=True)
    
    config = load_config(config_path)
    forecast_horizon = config.get('forecast_horizon', 30)
    num_classes = config.get('num_classes', 5)
    window_size = config.get('window_size', 30)
    
    # Generate return forecasts
    logger.info("Generating return forecasts...")
    return_forecasts = run_categorical_returns_forecast(
        config_path=config_path,
        output_dir=forecast_dir,
        forecast_horizon=forecast_horizon,
        num_classes=num_classes,
        forecast_periods=forecast_horizon,
        mode='full_train' if retrain_returns else 'predict'
    )
    
    # Generate volatility forecasts
    logger.info("Generating volatility forecasts...")
    vol_forecasts = run_volatility_forecast(
        config_path=config_path,
        output_dir=forecast_dir,
        window_size=window_size,
        forecast_periods=forecast_horizon,
        mode='full_train' if retrain_vol else 'predict',
        skip_tuning=not retrain_vol
    )
    
    logger.info("Forecasts generated successfully")
    return return_forecasts, vol_forecasts

def load_price_data(config_path='config.yaml'):
    """Load historical price data"""
    config = load_config(config_path)
    data_dir = config.get('data_dir', 'data')
    
    price_file = os.path.join(data_dir, 'processed', 'price_data.csv')
    if not os.path.exists(price_file):
        logger.error(f"Price data file not found: {price_file}")
        raise FileNotFoundError(f"Price data file not found: {price_file}")
    
    price_data = pd.read_csv(price_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded price data with shape {price_data.shape}")
    return price_data

def load_forecast_data(forecast_dir='results/forecasts'):
    """Load forecast data from CSV files"""
    forecasts = {}
    
    # Load return forecasts
    return_forecast_file = os.path.join(forecast_dir, 'return_forecast.csv')
    if os.path.exists(return_forecast_file):
        forecasts['return'] = pd.read_csv(return_forecast_file, index_col=0, parse_dates=True)
    
    # Load return class forecasts
    return_class_file = os.path.join(forecast_dir, 'return_class_forecasts.csv')
    if os.path.exists(return_class_file):
        forecasts['return_class'] = pd.read_csv(return_class_file, index_col=0, parse_dates=True)
    
    # Load volatility forecasts
    vol_forecast_file = os.path.join(forecast_dir, 'volatility_forecasts.csv')
    if os.path.exists(vol_forecast_file):
        forecasts['volatility'] = pd.read_csv(vol_forecast_file, index_col=0, parse_dates=True)
    
    # Load volatility metrics
    vol_metrics_file = os.path.join(forecast_dir, 'volatility_forecast_metrics.csv')
    if os.path.exists(vol_metrics_file):
        forecasts['volatility_metrics'] = pd.read_csv(vol_metrics_file, index_col=0)
    
    # Load future return forecasts
    future_forecast_file = os.path.join(forecast_dir, 'return_forecasts_future.csv')
    if os.path.exists(future_forecast_file):
        forecasts['future'] = pd.read_csv(future_forecast_file, index_col=0, parse_dates=True)
    
    return forecasts

def run_mvo_strategies(price_data, forecasts, risk_free_rate=0.01, initial_value=1000000):
    """Run all MVO portfolio strategies"""
    logger.info("Running MVO portfolio strategies...")
    
    # Initialize results dictionary
    portfolios = {}
    
    # Historical Mean-Variance Optimization
    logger.info("Running historical MVO...")
    portfolios['historical'] = run_mvo(price_data, risk_free_rate, initial_value)
    save_portfolio_details(portfolios['historical'], "historical_mvo")
    
    # Equal weight portfolio
    logger.info("Running equal weight portfolio...")
    portfolios['equal'] = equal_weight_portfolio(price_data, initial_value)
    save_portfolio_details(portfolios['equal'], "equal_weight")
    
    # Volatility forecast-enhanced MVO
    if 'volatility' in forecasts:
        logger.info("Running volatility forecast MVO...")
        portfolios['vol_forecast'] = run_vol_forecast_mvo(
            price_data, 
            risk_free_rate, 
            forecasts['volatility'],
            'hybrid', 
            initial_value
        )
        save_portfolio_details(portfolios['vol_forecast'], "volatility_forecast_mvo")
    
    # Return forecast-enhanced MVO
    if 'return' in forecasts and 'return_class' in forecasts:
        logger.info("Running return forecast MVO...")
        portfolios['return_forecast'] = run_return_forecast_mvo(
            price_data, 
            risk_free_rate, 
            initial_value
        )
        save_portfolio_details(portfolios['return_forecast'], "return_forecast_mvo")
    
    # Hybrid forecast-enhanced MVO
    if 'volatility' in forecasts and 'return' in forecasts and 'return_class' in forecasts:
        logger.info("Running hybrid forecast MVO...")
        portfolios['hybrid_forecast'] = run_hybrid_forecast_mvo(
            price_data, 
            risk_free_rate, 
            forecasts['volatility'],
            'hybrid', 
            initial_value
        )
        save_portfolio_details(portfolios['hybrid_forecast'], "hybrid_forecast_mvo")
    
    return portfolios

def run_ppo_strategies(price_data, forecasts, risk_free_rate=0.02, initial_value=10000, retrain_ppo=False):
    """Run all PPO portfolio strategies"""
    logger.info("Running PPO portfolio strategies...")
    
    # Initialize results dictionary
    portfolios = {}
    
    # Base PPO without forecasts
    logger.info("Running base PPO strategy...")
    portfolios['ppo_base'] = run_ppo(price_data, risk_free_rate, initial_value, retrain=retrain_ppo)
    ppo_save_details(portfolios['ppo_base'], "ppo_base")
    
    # Equal weight portfolio for comparison
    logger.info("Running equal weight portfolio for PPO comparison...")
    portfolios['equal'] = ppo_equal_weight(price_data, initial_value)
    ppo_save_details(portfolios['equal'], "equal_weight_ppo")
    
    # Volatility forecast-enhanced PPO
    if 'volatility' in forecasts:
        logger.info("Running volatility forecast PPO...")
        portfolios['vol_forecast'] = run_vol_forecast_ppo(
            price_data, 
            risk_free_rate, 
            forecasts['volatility'],
            'hybrid', 
            initial_value
        )
        ppo_save_details(portfolios['vol_forecast'], "volatility_forecast_ppo")
    
    # Return forecast-enhanced PPO
    if 'return' in forecasts and 'return_class' in forecasts:
        logger.info("Running return forecast PPO...")
        portfolios['return_forecast'] = run_return_forecast_ppo(
            price_data, 
            risk_free_rate, 
            initial_value
        )
        ppo_save_details(portfolios['return_forecast'], "return_forecast_ppo")
    
    # Hybrid forecast-enhanced PPO
    if 'volatility' in forecasts and 'return' in forecasts and 'return_class' in forecasts:
        logger.info("Running hybrid forecast PPO...")
        portfolios['hybrid_forecast'] = run_hybrid_forecast_ppo(
            price_data, 
            risk_free_rate, 
            forecasts['volatility'],
            'hybrid', 
            initial_value
        )
        ppo_save_details(portfolios['hybrid_forecast'], "hybrid_forecast_ppo")
    
    return portfolios

def plot_cumulative_returns(strategies, output_dir='results', title="Cumulative Returns Comparison"):
    """Create cumulative returns comparison chart"""
    plt.figure(figsize=(14, 8))
    
    for name, data in strategies.items():
        if isinstance(data, dict) and 'cumulative_return' in data:
            plt.plot(data['cumulative_return'], label=name)
        elif isinstance(data, pd.Series):
            plt.plot(data, label=name)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add date and save
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f'cumulative_returns_comparison_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Cumulative returns plot saved to {output_file}")
    
    return output_file

def calculate_portfolio_metrics(strategies):
    """Calculate key portfolio metrics for all strategies"""
    metrics = {}
    
    for name, data in strategies.items():
        # Extract returns data
        if isinstance(data, dict) and 'daily_return' in data:
            returns = data['daily_return'].dropna()
        elif isinstance(data, pd.Series):
            returns = data.pct_change().dropna()
        else:
            logger.warning(f"Cannot calculate metrics for {name} - invalid data format")
            continue
        
        # Convert to numpy array for calculations
        returns_array = np.array(returns)
        
        # Calculate metrics
        total_return = (1 + returns_array).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns_array)) - 1
        annual_volatility = returns_array.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino ratio (downside risk)
        downside_returns = returns_array[returns_array < 0]
        downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = annual_return / downside_risk if downside_risk > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns_array).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Store metrics
        metrics[name] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown
        }
    
    # Convert to DataFrame for easier display and comparison
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.sort_values('sharpe_ratio', ascending=False)
    
    return metrics_df

def save_metrics_table(metrics_df, output_dir='results'):
    """Save portfolio metrics to CSV and generate formatted table"""
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f'portfolio_metrics_comparison_{timestamp}.csv')
    
    # Format metrics for readability
    formatted_df = metrics_df.copy()
    formatted_df['total_return'] = formatted_df['total_return'].map('{:.2%}'.format)
    formatted_df['annual_return'] = formatted_df['annual_return'].map('{:.2%}'.format)
    formatted_df['annual_volatility'] = formatted_df['annual_volatility'].map('{:.2%}'.format)
    formatted_df['sharpe_ratio'] = formatted_df['sharpe_ratio'].map('{:.2f}'.format)
    formatted_df['sortino_ratio'] = formatted_df['sortino_ratio'].map('{:.2f}'.format)
    formatted_df['max_drawdown'] = formatted_df['max_drawdown'].map('{:.2%}'.format)
    
    # Save raw values
    metrics_df.to_csv(output_file)
    
    # Create visualization of the metrics
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df.sort_values('sharpe_ratio', ascending=False), 
                annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('Portfolio Strategy Metrics Comparison', fontsize=16)
    plt.tight_layout()
    
    heatmap_file = os.path.join(output_dir, f'portfolio_metrics_heatmap_{timestamp}.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Portfolio metrics saved to {output_file} and heatmap to {heatmap_file}")
    
    return formatted_df, output_file, heatmap_file

def main():
    """Main function to run the entire portfolio optimization pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Portfolio Optimization Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--mvo_only', action='store_true', 
                        help='Run only Mean-Variance Optimization strategies')
    parser.add_argument('--ppo_only', action='store_true', 
                        help='Run only PPO reinforcement learning strategies')
    parser.add_argument('--retrain_returns', action='store_true', 
                        help='Force retraining of returns forecast models')
    parser.add_argument('--retrain_vol', action='store_true', 
                        help='Force retraining of volatility forecast models')
    parser.add_argument('--retrain_ppo', action='store_true', 
                        help='Force retraining of PPO models')
    parser.add_argument('--forecasts_only', action='store_true', 
                        help='Generate forecasts only, skip portfolio optimization')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set up directories
    forecast_dir = os.path.join('results', 'forecasts')
    output_dir = 'results'
    ensure_directories([forecast_dir, output_dir])
    
    # Set risk-free rate from config
    risk_free_rate = config.get('risk_free_rate', 0.01)
    initial_value = config.get('initial_value', 1000000)
    
    # Step 1: Check and generate forecasts if needed
    if not forecasts_exist(forecast_dir) or args.retrain_returns or args.retrain_vol:
        logger.info("Generating forecasts...")
        generate_forecasts(args.config, forecast_dir, args.retrain_returns, args.retrain_vol)
    
    # Exit if only forecasts requested
    if args.forecasts_only:
        logger.info("Forecasts generated. Exiting as requested.")
        return
    
    # Step 2: Load data
    price_data = load_price_data(args.config)
    forecasts = load_forecast_data(forecast_dir)
    
    # Step 3: Run portfolio optimization strategies
    all_strategies = {}
    
    if not args.ppo_only:
        # Run MVO strategies
        logger.info("Running Mean-Variance Optimization strategies...")
        mvo_portfolios = run_mvo_strategies(price_data, forecasts, risk_free_rate, initial_value)
        
        # Add MVO strategies to all strategies
        for key, value in mvo_portfolios.items():
            all_strategies[f"MVO_{key}"] = value
    
    if not args.mvo_only:
        # Run PPO strategies
        logger.info("Running PPO reinforcement learning strategies...")
        ppo_portfolios = run_ppo_strategies(price_data, forecasts, risk_free_rate, 
                                           initial_value, args.retrain_ppo)
        
        # Add PPO strategies to all strategies
        for key, value in ppo_portfolios.items():
            all_strategies[f"PPO_{key}"] = value
    
    # Step 4: Compare and visualize results
    logger.info("Comparing strategies and generating visualizations...")
    
    # Plot cumulative returns
    cumulative_returns_plot = plot_cumulative_returns(all_strategies, output_dir)
    
    # Calculate and save portfolio metrics
    metrics_df = calculate_portfolio_metrics(all_strategies)
    formatted_metrics, metrics_file, heatmap_file = save_metrics_table(metrics_df, output_dir)
    
    # Print summary to console
    logger.info("\n===== Portfolio Optimization Results =====")
    logger.info("\nStrategy Performance Metrics:")
    logger.info("\n" + formatted_metrics.to_string())
    logger.info(f"\nCumulative returns plot: {cumulative_returns_plot}")
    logger.info(f"Metrics table: {metrics_file}")
    logger.info(f"Metrics heatmap: {heatmap_file}")
    logger.info("\nPortfolio optimization complete!")

if __name__ == "__main__":
    main()