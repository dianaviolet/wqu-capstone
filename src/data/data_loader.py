#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preparation utilities.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

def load_data(tickers, start_date, end_date):
    """
    Download price and volume data for specified tickers.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        tuple: (prices_df, volume_df)
    """
    logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    try:
        prices = yf.download(tickers=tickers, start=start_date, end=end_date, interval="1d")["Close"]
        volume = yf.download(tickers=tickers, start=start_date, end=end_date, interval="1d")["Volume"]
        
        # Handle case where only one ticker is downloaded
        if len(tickers) == 1:
            prices = pd.DataFrame(prices, columns=tickers)
            volume = pd.DataFrame(volume, columns=tickers)
            
        # Check for missing data
        missing_data = prices.isna().sum()
        if missing_data.sum() > 0:
            logger.warning(f"Missing data detected: {missing_data[missing_data > 0]}")
            
            # Fill missing values using forward fill method
            prices = prices.ffill()
            volume = volume.fillna(0)
            
        logger.info(f"Downloaded data for {len(tickers)} tickers: shape {prices.shape}")
        return prices, volume
    
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise

def split_dataset(df, train_size, valid_size=0):
    """
    Split a time series dataset into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame to split
        train_size (float): Proportion of data for training
        valid_size (float): Proportion of data for validation
        
    Returns:
        tuple: (train_df, valid_df, test_df) or (train_df, test_df) if valid_size=0
    """
    train_end = int(df.shape[0] * train_size)
    valid_end = int(df.shape[0] * (train_size + valid_size))

    train_df = df.iloc[:train_end]
    test_df = df.iloc[valid_end:]
    
    if valid_size == 0:
        return train_df, test_df
    else:
        valid_df = df.iloc[train_end:valid_end]
        return train_df, valid_df, test_df

def create_lagged_features(returns_df, target_col, lags=5):
    """
    Create lagged features for time series prediction.
    
    Args:
        returns_df (pd.DataFrame): Returns data
        target_col (str): Target column to predict
        lags (int or list): Number of lags or list of lag values
        
    Returns:
        pd.DataFrame: DataFrame with lagged features
    """
    if isinstance(lags, int):
        lags = range(1, lags + 1)
        
    result_df = pd.DataFrame(index=returns_df.index)
    
    # Add target variable
    result_df[target_col] = returns_df[target_col]
    
    # Add lagged features for all columns
    for col in returns_df.columns:
        for lag in lags:
            result_df[f"{col}_lag_{lag}"] = returns_df[col].shift(lag)
            
    # Drop rows with NaN values
    result_df = result_df.dropna()
    
    return result_df

def calculate_technical_indicators(prices_df):
    """
    Calculate technical indicators for price data.
    
    Args:
        prices_df (pd.DataFrame): Price data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    tech_indicators = pd.DataFrame(index=prices_df.index)
    
    # Calculate indicators for each asset
    for col in prices_df.columns:
        # Simple Moving Averages
        tech_indicators[f"{col}_SMA_5"] = prices_df[col].rolling(window=5).mean()
        tech_indicators[f"{col}_SMA_20"] = prices_df[col].rolling(window=20).mean()
        tech_indicators[f"{col}_SMA_50"] = prices_df[col].rolling(window=50).mean()
        
        # Exponential Moving Averages
        tech_indicators[f"{col}_EMA_5"] = prices_df[col].ewm(span=5, adjust=False).mean()
        tech_indicators[f"{col}_EMA_20"] = prices_df[col].ewm(span=20, adjust=False).mean()
        
        # Relative price to moving averages
        tech_indicators[f"{col}_Rel_SMA_20"] = prices_df[col] / tech_indicators[f"{col}_SMA_20"] - 1
        
        # Volatility (standard deviation)
        tech_indicators[f"{col}_Vol_20"] = prices_df[col].rolling(window=20).std()
        
        # Rate of change
        tech_indicators[f"{col}_ROC_5"] = prices_df[col].pct_change(periods=5)
        tech_indicators[f"{col}_ROC_20"] = prices_df[col].pct_change(periods=20)
        
        # Momentum
        tech_indicators[f"{col}_Mom_10"] = prices_df[col] - prices_df[col].shift(10)
        
        # RSI (simplified calculation)
        delta = prices_df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        tech_indicators[f"{col}_RSI_14"] = 100 - (100 / (1 + rs))
        
    # Drop rows with NaN values
    tech_indicators = tech_indicators.dropna()
    
    return tech_indicators