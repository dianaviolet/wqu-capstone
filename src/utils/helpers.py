#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper utilities for portfolio analysis and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def clear_memory():
    """Clear unused memory."""
    gc.collect()

def plot_portfolio_weights(weights, title="Portfolio Weights"):
    """
    Plot portfolio weights as a bar chart.
    
    Args:
        weights (pd.Series): Portfolio weights
        title (str): Chart title
    """
    plt.figure(figsize=(10, 6))
    weights.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Weight')
    plt.xlabel('Asset')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def plot_cumulative_returns(returns_dict, title="Cumulative Returns Comparison", figsize=(12, 7)):
    """
    Plot cumulative returns for multiple strategies.
    
    Args:
        returns_dict (dict): Dictionary of returns series with strategy names as keys
        title (str): Chart title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        plt.plot(cumulative, label=name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt

def evaluate_classification(y_true, y_pred, labels=None):
    """
    Evaluate classification predictions.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (list, optional): Label names
        
    Returns:
        dict: Evaluation metrics
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    cr = classification_report(y_true, y_pred, output_dict=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return {
        'confusion_matrix': cm,
        'classification_report': cr
    }

def calculate_drawdowns(returns):
    """
    Calculate drawdown statistics from returns.
    
    Args:
        returns (pd.Series): Returns series
        
    Returns:
        tuple: (drawdowns, max_drawdown, max_drawdown_duration)
    """
    # Calculate wealth index
    wealth_index = (1 + returns).cumprod()
    
    # Calculate previous peaks
    previous_peaks = wealth_index.cummax()
    
    # Calculate drawdowns
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Find maximum drawdown
    max_drawdown = drawdowns.min()
    
    # Calculate drawdown duration
    drawdown_started = False
    current_drawdown_duration = 0
    max_drawdown_duration = 0
    
    for dd in drawdowns:
        if dd < 0:
            drawdown_started = True
            current_drawdown_duration += 1
        elif drawdown_started:
            drawdown_started = False
            max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
            current_drawdown_duration = 0
    
    # In case we're still in a drawdown at the end
    if drawdown_started:
        max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
    
    return drawdowns, max_drawdown, max_drawdown_duration

def create_rolling_predictions(model, data, window_size, target_col, scaler=None):
    """
    Create rolling predictions for time series data.
    
    Args:
        model: Trained model
        data (pd.DataFrame): Data to predict on
        window_size (int): Size of the rolling window
        target_col (str): Target column name
        scaler (object, optional): Scaler for preprocessing
        
    Returns:
        pd.DataFrame: Predictions DataFrame
    """
    # Create a copy of the data to avoid modifying the original
    predictions = pd.DataFrame(index=data.index)
    predictions['actual'] = data[target_col]
    
    for i in range(window_size, len(data)):
        # Get window data
        window_data = data.iloc[i-window_size:i].copy()
        
        # Scale data if scaler provided
        if scaler is not None:
            X = scaler.transform(window_data.drop(columns=[target_col]))
        else:
            X = window_data.drop(columns=[target_col])
        
        # Get prediction
        pred = model.predict(X)
        
        # Store results
        predictions.loc[data.index[i], 'predicted'] = pred[0]
    
    return predictions

def get_config_param(config, param_path, default_value):
    """
    Get parameter from nested config with fallback to default value.
    
    Args:
        config (dict): Configuration dictionary
        param_path (str): Dot-separated path to parameter (e.g., "lstm.learning_rate")
        default_value: Default value if parameter not found
    
    Returns:
        Parameter value from config or default
    """
    keys = param_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default_value
    
    return current 