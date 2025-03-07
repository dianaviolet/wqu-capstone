#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volatility forecasting script using LSTM-GARCH hybrid model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import os
import argparse
import logging
import yaml
import traceback
import sys
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from arch import arch_model
import pickle

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import from src
from src.data.data_loader import load_data, split_dataset, create_lagged_features, calculate_technical_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volatility_forecast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class GARCHModel:
    def __init__(self, model_type):
        self.model_type = model_type

    def metrics(self, obs, predict):
        mse = np.mean((obs - predict) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(obs - predict))
        mape = np.mean(np.abs((obs - predict) / np.where(obs != 0, obs, np.nan))) * 100
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}

    def build_model(self, ticker, p, q, o, dist, returns):
        """Build and fit a GARCH model without plotting."""
        train_df, test_df = split_dataset(returns, 0.8, 0)
        
        model = arch_model(returns, vol=self.model_type, p=p, o=o, q=q, dist=dist)
        fit = model.fit(disp='off', show_warning=False, last_obs=train_df.index[-1])

        # Get conditional volatility
        cond_vol = fit.conditional_volatility
        forecast = fit.forecast(horizon=1, start=train_df.index[-1], align='target')
        
        # Ensure forecast variance is properly shaped for Series creation
        forecast_variance = np.sqrt(forecast.variance[1:].values).flatten()

        train_perf = self.metrics(train_df.abs(), cond_vol.loc[train_df.index])
        test_perf = self.metrics(test_df.abs(), forecast_variance)

        return fit, train_perf, test_perf, cond_vol, forecast_variance, test_df

class LSTMHyperparameterTuner:
    def __init__(self, act_fun='relu', seed=1234, window_size=30, max_epochs=50,
                 lstm_units=(16, 49), dropout_rate=(0.1, 0.2), lstm_layers=(0, 2),
                 dense_layers=(0, 2), learning_rate=(1e-4, 5e-4)):
        self.act_fun = act_fun
        self.seed = seed
        self.window_size = window_size
        self.max_epochs = max_epochs
        self.model = None
        self.scaler_input = None
        self.scaler_output = None
        self.tuner = None
        self.n_features = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.perf = {}
        self.feature_names = None

        # Hyperparameter values
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.learning_rate = learning_rate

    def prepare_data(self, train_df, valid_df, test_df, target):
        # Split data into X and y
        X_train_df = train_df.drop(columns=[target])
        y_train_df = train_df[[target]]

        X_valid_df = valid_df.drop(columns=[target])
        y_valid_df = valid_df[[target]]

        X_test_df = test_df.drop(columns=[target])
        y_test_df = test_df[[target]]

        self.n_features = X_train_df.shape[1]
        self.feature_names = X_train_df.columns.tolist()

        # Scale data
        self.scaler_input = MinMaxScaler(feature_range=(0, 1))
        self.scaler_input.fit(X_train_df)
        X_train_scaled = self.scaler_input.transform(X_train_df)
        X_valid_scaled = self.scaler_input.transform(X_valid_df)
        X_test_scaled = self.scaler_input.transform(X_test_df)

        self.scaler_output = MinMaxScaler(feature_range=(0, 1))
        self.scaler_output.fit(y_train_df)
        y_train_scaled = self.scaler_output.transform(y_train_df)
        y_valid_scaled = self.scaler_output.transform(y_valid_df)
        y_test_scaled = self.scaler_output.transform(y_test_df)

        # Create time series data
        def create_sequences(X, y):
            Xs, ys = [], []
            for i in range(self.window_size, len(X)):
                Xs.append(X[i - self.window_size:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        self.X_train, self.y_train = create_sequences(X_train_scaled, y_train_scaled)
        self.X_valid, self.y_valid = create_sequences(X_valid_scaled, y_valid_scaled)
        self.X_test, self.y_test = create_sequences(X_test_scaled, y_test_scaled)

    def build_model(self, hp):
        n_dropout = hp.Choice('n_dropout', values=self.dropout_rate)
        hp_lr = hp.Choice("learning_rate", values=self.learning_rate)

        model = Sequential()
        model.add(LSTM(units=hp.Int('units_LSTM_0', min_value=self.lstm_units[0], max_value=self.lstm_units[1], step=16),
                       activation="tanh", return_sequences=True, input_shape=(self.X_train.shape[1], self.n_features)))

        for i in range(1, hp.Int("num_lstm", self.lstm_layers[0], self.lstm_layers[1])):
            model.add(LSTM(
                units=hp.Int("units_LSTM_" + str(i), min_value=self.lstm_units[0], max_value=self.lstm_units[1], step=16),
                activation="tanh",
                return_sequences=True))
            model.add(Dropout(n_dropout))

        # Final LSTM layer
        model.add(LSTM(units=8, activation="tanh", return_sequences=False))
        model.add(Dropout(n_dropout, seed=self.seed))

        # Dense layers
        for i in range(1, hp.Int("num_dense", self.dense_layers[0], self.dense_layers[1])):
            model.add(Dense(units=16, activation=self.act_fun))
            model.add(Dropout(n_dropout))

        model.add(Dense(1))

        adam = tf.keras.optimizers.Adam(learning_rate=hp_lr)
        model.compile(optimizer=adam, loss="mean_absolute_error")
        return model

    def tune(self, max_trials=5):
        directory = os.path.abspath('tuner_dir')
        os.makedirs(directory, exist_ok=True)
        tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=kt.Objective("val_loss", direction="min"),
            max_epochs=self.max_epochs,
            factor=3,
            directory='tuner_dir',
            project_name='lstm_tuning',
            overwrite=True
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

        # Silence excessive logging
        original_verbosity = tf.get_logger().getEffectiveLevel()
        tf.get_logger().setLevel('ERROR')

        logger.info("Tuning hyperparameters...")
        tuner.search(
            self.X_train, self.y_train,
            epochs=self.max_epochs,
            validation_data=(self.X_valid, self.y_valid),
            callbacks=[early_stopping],
            verbose=0
        )

        # Restore original verbosity
        tf.get_logger().setLevel(original_verbosity)

        self.tuner = tuner
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters: {best_hps.values}")
        return best_hps

    def train(self):
        if self.tuner is None:
            logger.error("Must tune model first")
            return None

        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        model = self.tuner.hypermodel.build(best_hps)

        # Train with best hyperparameters
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        
        # Silence excessive logging
        original_verbosity = tf.get_logger().getEffectiveLevel()
        tf.get_logger().setLevel('ERROR')
        
        logger.info("Training model with best hyperparameters...")
        history = model.fit(
            self.X_train, self.y_train,
            epochs=self.max_epochs,
            batch_size=32,
            validation_data=(self.X_valid, self.y_valid),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Restore original verbosity
        tf.get_logger().setLevel(original_verbosity)
        
        self.model = model
        return model, history

    def predict(self):
        if self.model is None:
            logger.error("Must train model first")
            return None

        predictions_scaled = self.model.predict(self.X_test, verbose=0)
        predictions = self.scaler_output.inverse_transform(predictions_scaled)
        
        actual = self.scaler_output.inverse_transform(self.y_test.reshape(-1, 1))
        
        return predictions, actual

    def prepare_single_window(self, window_data, target_col):
        """Prepare a single window of data for prediction."""
        # Create X (features)
        X_window = window_data.drop(columns=[target_col]).copy()
        
        # Scale the data
        X_window_scaled = self.scaler_input.transform(X_window)
        
        # Reshape for LSTM (samples, time steps, features)
        X_window_reshaped = X_window_scaled.reshape(1, X_window_scaled.shape[0], X_window_scaled.shape[1])
        
        return X_window_reshaped
    
    def predict_single(self, X_window):
        """Make a prediction for a single window."""
        if self.model is None:
            logger.error("Must train model first")
            return None
        
        # Get prediction
        prediction_scaled = self.model.predict(X_window, verbose=0)
        
        # Inverse transform the prediction
        prediction = self.scaler_output.inverse_transform(prediction_scaled)[0][0]
        
        return prediction
    
    def metrics(self, actual, predicted):
        """Calculate performance metrics."""
        # MSE
        mse = np.mean((actual - predicted) ** 2)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(actual - predicted))
        
        # MAPE
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, np.nan))) * 100
        
        # R-squared
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        metrics_dict = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R-squared": r_squared
        }
        
        return metrics_dict

    def save_model(self, model_path):
        """Save the trained model to disk."""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Add .keras extension if not present
            if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
                model_path = f"{model_path}.keras"
            
            # Save the model
            self.model.save(model_path)
            
            # Create metadata dictionary
            metadata = {
                'feature_names': self.feature_names,
                'window_size': self.window_size,
                'n_features': self.n_features
            }
            
            # Save metadata alongside the model
            metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            # Save scalers
            scaler_path = f"{os.path.splitext(model_path)[0]}_scalers.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump({'input': self.scaler_input, 'output': self.scaler_output}, f)
            
            logger.info(f"Model and metadata saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_path):
        """Load a model from disk with proper optimizer handling."""
        try:
            # Add .keras extension if not present
            if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
                model_path = f"{model_path}.keras"
            
            # Load the model
            self.model = tf.keras.models.load_model(
                model_path, 
                compile=False  # Don't load optimizer state
            )
            
            # Recompile the model with a fresh optimizer
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='mean_absolute_error'
            )
            
            # Load metadata
            metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            # Set metadata attributes
            self.feature_names = metadata.get('feature_names')
            self.window_size = metadata.get('window_size')
            self.n_features = metadata.get('n_features')
            
            # Load scalers
            scaler_path = f"{os.path.splitext(model_path)[0]}_scalers.pkl"
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler_input = scalers.get('input')
                self.scaler_output = scalers.get('output')
            
            logger.info(f"Model and metadata loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def build_model_with_params(self, params):
        """Build model with explicit hyperparameters without tuning."""
        model = Sequential()
        
        # First LSTM layer (always included)
        model.add(LSTM(units=params.get('units_LSTM_0', 16),
                    activation="tanh", return_sequences=True, 
                    input_shape=(self.X_train.shape[1], self.n_features)))
        
        # Additional LSTM layers
        for i in range(1, params.get('num_lstm', 0) + 1):
            units = params.get(f'units_LSTM_{i}', 16)
            model.add(LSTM(units=units, activation="tanh", return_sequences=True))
            model.add(Dropout(params.get('n_dropout', 0.1)))
        
        # Final LSTM layer
        model.add(LSTM(units=8, activation="tanh", return_sequences=False))
        model.add(Dropout(params.get('n_dropout', 0.1), seed=self.seed))
        
        # Dense layers
        for i in range(params.get('num_dense', 0)):
            model.add(Dense(units=16, activation=self.act_fun))
            model.add(Dropout(params.get('n_dropout', 0.1)))
        
        model.add(Dense(1))
        
        # Compile model
        adam = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.0005))
        model.compile(optimizer=adam, loss="mean_absolute_error")
        
        self.model = model
        return model
    
    def train_with_params(self, params):
        """Train model with specific parameters without tuning."""
        model = self.build_model_with_params(params)
        
        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        
        # Silence excessive logging
        original_verbosity = tf.get_logger().getEffectiveLevel()
        tf.get_logger().setLevel('ERROR')
        
        logger.info("Training model with specified hyperparameters...")
        history = model.fit(
            self.X_train, self.y_train,
            epochs=self.max_epochs,
            batch_size=32,
            validation_data=(self.X_valid, self.y_valid),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Restore original verbosity
        tf.get_logger().setLevel(original_verbosity)
        
        self.model = model
        return model, history

def run_volatility_forecast(config_path='config.yaml', output_dir='forecasts', window_size=30,
                           forecast_periods=30, start_forecast_date=None, end_forecast_date=None,
                           mode='full_train', model_dir='models', skip_tuning=False):
    """Run volatility forecasting with LSTM-GARCH hybrid model."""
    # Ensure paths are relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Convert relative paths to absolute paths based on project root
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
        
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(project_root, model_dir)
    
    # Create output and model directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model directory: {model_dir}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract parameters
    start_date = config.get('start_date', '2009-01-01')
    end_date = config.get('end_date', '2024-12-31')
    tickers = config.get('tickers', ["SPY", "AOM", "VT", "AGG", "BIL", "^VIX"])
    
    # Dictionary to store all forecasts for each ticker
    all_forecasts = {}
    all_metrics = {}
    
    # Load data directly using data_loader.load_data
    logger.info(f"Loading data for {len(tickers)} tickers from {start_date} to {end_date}")
    prices_df, volume_df = load_data(tickers, start_date, end_date)
    
    if prices_df is None or prices_df.empty:
        logger.error("Failed to load price data")
        return None
    
    # Calculate returns for all tickers
    returns_df = prices_df.pct_change().dropna()
    
    # Prepare GARCH models for all tickers (if needed)
    if mode != 'predict':
        logger.info("Fitting GARCH models to generate volatility features")
        garch_volatilities = {}
        
        for ticker in tickers:
            try:
                # Use best GARCH parameters or default
                p, q, o = 1, 1, 1  # Default EGARCH params
                dist = 'normal'
                
                # Create EGARCH model
                garch_model = GARCHModel("EGARCH")
                
                # Fit GARCH model without plots
                _, _, _, cond_vol, forecast, test_df = garch_model.build_model(
                    ticker, p, q, o, dist, returns_df[ticker]
                )
                
                # Store conditional volatility and forecast
                garch_volatilities[ticker] = pd.concat([
                    cond_vol, 
                    pd.Series(forecast, index=test_df.index)
                ]).rename('EGARCH')
                
            except Exception as e:
                logger.error(f"Error fitting GARCH model for {ticker}: {str(e)}")
                logger.error(traceback.format_exc())
    
    # Process each ticker
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}")
            
            # Create target column (realized volatility)
            target_col = f"{ticker}_vol"
            
            # Calculate realized volatility (21-day rolling standard deviation of returns)
            prices_df[target_col] = returns_df[ticker].rolling(21).std() * np.sqrt(252)
            
            # Split dataset
            train_data, valid_data, test_data = split_dataset(prices_df, train_size=0.7, valid_size=0.1)
            
            # Create technical indicators
            train_features = calculate_technical_indicators(train_data[[ticker]])
            valid_features = calculate_technical_indicators(valid_data[[ticker]])
            test_features = calculate_technical_indicators(test_data[[ticker]])
            
            # Add GARCH volatility as a feature if available
            if mode != 'predict' and ticker in garch_volatilities:
                garch_vol = garch_volatilities[ticker]
                
                # Replace the problematic lines with a safer merge approach
                # Instead of using .loc[train_features.index] which requires exact matches
                train_features = pd.merge(train_features, 
                                         garch_vol.to_frame(), 
                                         left_index=True, 
                                         right_index=True, 
                                         how='left')
                
                valid_features = pd.merge(valid_features, 
                                         garch_vol.to_frame(), 
                                         left_index=True, 
                                         right_index=True, 
                                         how='left')
                
                test_features = pd.merge(test_features, 
                                        garch_vol.to_frame(), 
                                        left_index=True, 
                                        right_index=True, 
                                        how='left')
                
                # Fill any NaN values that might result from the merge
                train_features = train_features.ffill()
                valid_features = valid_features.ffill()
                test_features = test_features.ffill()
            
            # Make sure the target column is preserved
            train_features[target_col] = train_data[target_col]
            valid_features[target_col] = valid_data[target_col]
            test_features[target_col] = test_data[target_col]
            
            # Create lagged features for the target
            train_features = create_lagged_features(train_features, target_col, lags=5)
            valid_features = create_lagged_features(valid_features, target_col, lags=5)
            test_features = create_lagged_features(test_features, target_col, lags=5)
            
            # Drop NaN values
            train_features = train_features.dropna()
            valid_features = valid_features.dropna()
            test_features = test_features.dropna()
            
            # Initialize a Series to store all forecasts for this ticker
            ticker_forecasts = pd.Series(dtype=float)
            
            # Initialize hyperparameter tuner
            lstm_model_tuner = LSTMHyperparameterTuner(
                act_fun='relu',
                seed=1234,
                window_size=window_size,
                max_epochs=50,
                lstm_units=(16, 33),
                dropout_rate=[0.1, 0.2],
                lstm_layers=(0, 2),
                dense_layers=(0, 2),
                learning_rate=[1e-4, 5e-4]
            )
            
            # Define model path for this ticker
            model_path = os.path.join(model_dir, f"{ticker}_lstm_volatility_model")
            
            # If in predict mode, try to load existing model first
            model_loaded = False
            if mode == 'predict':
                logger.info(f"Attempting to load existing model for {ticker}")
                lstm_model_tuner.prepare_data(
                    train_features, valid_features, test_features, 
                    target=target_col
                )
                model_loaded = lstm_model_tuner.load_model(model_path)
            
            # If not in predict mode or model loading failed, train a new model
            if mode != 'predict' or not model_loaded:
                # Log data sizes
                logger.info(f"Training size: {len(train_features)}, validation size: {len(valid_features)}, test size: {len(test_features)}")
                
                # Prepare data for training
                lstm_model_tuner.prepare_data(
                    train_features, valid_features, test_features, 
                    target=target_col
                )
                
                # Access the attributes directly
                if hasattr(lstm_model_tuner, 'X_train') and lstm_model_tuner.X_train is not None and len(lstm_model_tuner.X_train) > 0:
                    # Train the model with or without tuning
                    if skip_tuning:
                        # Use default hyperparameters without tuning
                        default_params = {
                            'units_LSTM_0': 32,
                            'units_LSTM_1': 16,
                            'n_dropout': 0.1,
                            'num_lstm': 1,
                            'num_dense': 1,
                            'learning_rate': 0.0005
                        }
                        logger.info(f"Skipping tuning and using default hyperparameters: {default_params}")
                        lstm_model_tuner.train_with_params(default_params)
                    else:
                        # Tune and train the model
                        lstm_model_tuner.tune(max_trials=5)
                        lstm_model_tuner.train()
                    
                    # Save the trained model
                    if mode != 'predict':
                        logger.info(f"Saving trained model for {ticker}")
                        lstm_model_tuner.save_model(model_path)
                else:
                    logger.warning(f"Could not prepare data for {ticker}")
            
            # Forecast for the entire test set at once
            prediction_result = lstm_model_tuner.predict()
            
            # Create a Series with the predictions
            if prediction_result is not None and len(prediction_result) > 0:
                # Extract predictions from the tuple returned by predict()
                test_predictions = prediction_result[0]  # First element of the tuple
                
                # Get the dates of the test set (adjusted for window size if needed)
                if hasattr(lstm_model_tuner, 'test_dates') and lstm_model_tuner.test_dates is not None:
                    test_dates = lstm_model_tuner.test_dates
                else:
                    # Fallback to using test_features index
                    test_dates = test_features.index[-len(test_predictions):]
                
                # Now flatten the actual predictions array
                forecast_series = pd.Series(test_predictions.flatten(), index=test_dates)
                logger.info(f"Made {len(forecast_series)} forecasts for {ticker}")
                ticker_forecasts = forecast_series
            else:
                logger.warning(f"No forecasts generated for {ticker}")
            
            # Store all forecasts for this ticker
            if not ticker_forecasts.empty:
                all_forecasts[ticker] = ticker_forecasts
            
            # Calculate metrics if actuals are available
            if not ticker_forecasts.empty:
                try:
                    # Fix issue with index duplication by rebuilding both Series with unique indices
                    # First, create a new DataFrame with one row per date
                    forecast_dates = ticker_forecasts.index.unique()
                    
                    # Create new Series for clean comparison
                    clean_forecasts = pd.Series(dtype=float)
                    clean_actuals = pd.Series(dtype=float)
                    
                    # For each unique date, get values from both series
                    for date in forecast_dates:
                        if date in test_features.index:
                            # Get forecast value (use mean if multiple exist)
                            forecast_val = ticker_forecasts.loc[date].mean() if isinstance(ticker_forecasts.loc[date], pd.Series) else ticker_forecasts.loc[date]
                            
                            # Get actual value (use first value if multiple exist)
                            if date in test_features.index:
                                actual_val = test_features.loc[date, target_col].iloc[0] if hasattr(test_features.loc[date, target_col], 'iloc') else test_features.loc[date, target_col]
                                
                                # Add to our clean series
                                clean_forecasts[date] = forecast_val
                                clean_actuals[date] = actual_val
                    
                    logger.info(f"Calculating metrics using {len(clean_actuals)} data points")
                    
                    # Now the lengths should match
                    if len(clean_actuals) == len(clean_forecasts):
                        # Use numpy arrays for calculations to avoid index issues
                        metrics_dict = GARCHModel("EGARCH").metrics(
                            clean_actuals.values, 
                            clean_forecasts.values
                        )
                        all_metrics[ticker] = metrics_dict
                        
                        logger.info(f"Performance metrics for {ticker}:")
                        for metric, value in metrics_dict.items():
                            logger.info(f"{metric}: {value:.6f}")
                    else:
                        logger.error(f"Length mismatch after cleanup: actual values ({len(clean_actuals)}) != forecasts ({len(clean_forecasts)})")
                except Exception as e:
                    logger.error(f"Error calculating metrics for {ticker}: {str(e)}")
                    logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # Save all forecasts
    if all_forecasts:
        # Create DataFrame from all forecasts
        forecast_df = pd.DataFrame(all_forecasts)
        test_file = os.path.join(output_dir, 'volatility_forecasts.csv')
        forecast_df.to_csv(test_file)
        logger.info(f"Volatility forecasts saved to {test_file}")
        
        # Save metrics
        if all_metrics:
            metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
            metrics_file = os.path.join(output_dir, 'volatility_forecast_metrics.csv')
            metrics_df.to_csv(metrics_file)
            logger.info(f"Volatility forecast metrics saved to {metrics_file}")
        
    return all_forecasts, all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Volatility Forecasting with LSTM-GARCH')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--window', type=int, default=30, help='Lookback window size (days)')
    parser.add_argument('--periods', type=int, default=30, help='Number of forecast periods')
    parser.add_argument('--start-date', type=str, default=None, help='Start date for forecasting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for forecasting (YYYY-MM-DD)')
    
    # Execution mode arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full_train', action='store_true', help='Run full walk-forward training')
    group.add_argument('--predict', action='store_true', help='Make predictions without retraining')
    
    # Add option to skip hyperparameter tuning
    parser.add_argument('--no_tune', action='store_true', help='Skip hyperparameter tuning and use default parameters')
    
    args = parser.parse_args()
    
    if args.full_train:
        run_mode = 'full_train'
    elif args.predict:
        run_mode = 'predict'
    else:
        logger.error("Invalid execution mode. Please use --full_train or --predict.")
        sys.exit(1)

    result = run_volatility_forecast(
        config_path=args.config,
        output_dir=args.output,
        window_size=args.window,
        forecast_periods=args.periods,
        start_forecast_date=args.start_date,
        end_forecast_date=args.end_date,
        mode=run_mode,
        model_dir=args.model_dir,
        skip_tuning=args.no_tune
    )
    
    if result is None:
        print("\nVolatility forecasting failed. Check the logs for details.")
    else:
        forecasts, metrics = result
        print("\nVolatility forecasting completed. Results saved to:", args.output)

