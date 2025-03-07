#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Categorical returns forecasting script.
Generates 5-class return forecasts using CNN-LSTM-GRU architecture and saves results to CSV.
"""

import argparse
import logging
import os
import yaml
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import traceback
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas_market_calendars as mcal
import warnings
import inspect
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed, Conv1D, MaxPooling1D # type: ignore
from tensorflow.keras.layers import Flatten, GRU, concatenate, LayerNormalization, Bidirectional # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import tensorflow.keras.backend as K # type: ignore
import pickle
import re
import traceback
import types

# Fix path to properly locate src modules
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data.data_loader import load_data, split_dataset, create_lagged_features, calculate_technical_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('categorical_returns_forecast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class CNNLSTMGRU_Model:
    """CNNLSTMGRU model for time series forecasting with categorical output."""
    
    def __init__(self, window_size=30, num_classes=5, max_epochs=50):
        """
        Initialize CNNLSTMGRU model.
        
        Args:
            window_size (int): Size of the lookback window
            num_classes (int): Number of categorical classes to predict
            max_epochs (int): Maximum number of training epochs
        """
        self.window_size = window_size
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.model = None
        self.scaler = None
        self.discretizer = None
        self.class_labels = None
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def prepare_data(self, train_df, valid_df, test_df, target, sequence_length=30, features=None):
        """
        Prepare data for training and testing.
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data
        valid_df : pd.DataFrame
            Validation data
        test_df : pd.DataFrame
            Test data
        target : str
            Target column name
        sequence_length : int
            Length of input sequences
        features : list
            Features to use (if None, use all columns except target)
        """
        self.target = target
        self.window_size = sequence_length
        
        # Select features if provided, otherwise use all columns except target
        if features is None:
            features = [col for col in train_df.columns if col != target and not pd.isna(col)]
            
            if not features:
                # Create lagged features of the target itself
                for lag in range(1, 6):  # Create 5 lags
                    lag_col = f"{target}_lag{lag}"
                    train_df[lag_col] = train_df[target].shift(lag)
                    valid_df[lag_col] = valid_df[target].shift(lag)
                    test_df[lag_col] = test_df[target].shift(lag)
                
                # Update features list with new lag columns
                features = [col for col in train_df.columns if col != target and not pd.isna(col)]
        
        # Log initial feature stats to diagnose issues
        logger.info(f"Features before cleaning: {len(features)}")
        for df_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            inf_count = np.isinf(df[features]).sum().sum()
            nan_count = np.isnan(df[features]).sum().sum()
            logger.info(f"{df_name} dataframe - inf values: {inf_count}, NaN values: {nan_count}")
        
        # Clean the dataframes by replacing inf values with NaN and then dropping NaN rows
        for df in [train_df, valid_df, test_df]:
            # Replace infinity values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Identify and drop problematic columns (more than 50% NaN or inf values)
            bad_cols = []
            for col in features:
                if col in df.columns:
                    null_pct = df[col].isna().mean()
                    if null_pct > 0.5:
                        bad_cols.append(col)
            
            if bad_cols:
                logger.warning(f"Dropping {len(bad_cols)} columns with >50% missing values: {bad_cols}")
                features = [col for col in features if col not in bad_cols]
        
        # Drop rows with NaN values
        train_df = train_df.dropna()
        valid_df = valid_df.dropna()
        test_df = test_df.dropna()
        
        # Log data shapes after cleaning
        logger.info(f"Data shapes after cleaning - train: {train_df.shape}, valid: {valid_df.shape}, test: {test_df.shape}")
        
        # Convert target to categorical
        self.discretizer = KBinsDiscretizer(n_bins=self.num_classes, encode='ordinal', strategy='quantile')
        
        # Check if we have enough training data after cleaning
        if len(train_df) < 10:  # Arbitrary minimum sample threshold
            raise ValueError(f"Not enough training data after cleaning: {len(train_df)} samples")
        
        y_train = train_df[target].values.reshape(-1, 1)
        self.discretizer.fit(y_train)
        
        # Transform target to categorical
        y_train_cat = self.discretizer.transform(y_train).flatten().astype(int)
        
        if len(valid_df) > 0:
            y_valid = valid_df[target].values.reshape(-1, 1)
            y_valid_cat = self.discretizer.transform(y_valid).flatten().astype(int)
        else:
            y_valid_cat = np.array([])
            
        if len(test_df) > 0:
            y_test = test_df[target].values.reshape(-1, 1)
            y_test_cat = self.discretizer.transform(y_test).flatten().astype(int)
        else:
            y_test_cat = np.array([])
        
        # Scale features with robust error handling
        try:
            self.scaler = StandardScaler()
            # First verify no inf values remain
            if np.isinf(train_df[features].values).any():
                raise ValueError("Infinity values still present in training data after cleaning")
            
            X_train = self.scaler.fit_transform(train_df[features])
            
            if len(valid_df) > 0:
                # Replace any remaining inf values with feature means
                valid_features_data = valid_df[features].copy()
                valid_features_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                valid_features_data.fillna(valid_features_data.mean(), inplace=True)
                X_valid = self.scaler.transform(valid_features_data)
            else:
                X_valid = np.array([])
            
            if len(test_df) > 0:
                # Replace any remaining inf values with feature means  
                test_features_data = test_df[features].copy()
                test_features_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                test_features_data.fillna(test_features_data.mean(), inplace=True)
                X_test = self.scaler.transform(test_features_data)
            else:
                X_test = np.array([])
        
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            # Get problematic columns for debugging
            problematic_cols = []
            for col in features:
                if col in train_df.columns:
                    if np.isinf(train_df[col]).any() or np.isnan(train_df[col]).any():
                        max_val = train_df[col].max() if not np.isinf(train_df[col].max()) else "inf"
                        min_val = train_df[col].min() if not np.isinf(train_df[col].min()) else "-inf"
                        nan_count = train_df[col].isna().sum()
                        problematic_cols.append(f"{col}: range [{min_val}, {max_val}], NaNs: {nan_count}")
            
            if problematic_cols:
                logger.error(f"Problematic columns: {problematic_cols}")
            raise
        
        # Create sequences
        self.n_features = X_train.shape[1]
        
        # Store data for training
        self.X_train = X_train
        self.y_train = tf.keras.utils.to_categorical(y_train_cat, num_classes=self.num_classes)
        self.X_valid = X_valid
        self.y_valid = tf.keras.utils.to_categorical(y_valid_cat, num_classes=self.num_classes) if len(y_valid_cat) > 0 else np.array([])
        self.X_test = X_test
        self.y_test = tf.keras.utils.to_categorical(y_test_cat, num_classes=self.num_classes) if len(y_test_cat) > 0 else np.array([])
        
        # Save feature names for later use
        self.feature_names = features
        
        logger.info(f"Prepared data with {self.n_features} features, {len(self.X_train)} training samples")
        return self
    
    def build_model(self, conv_filters=64, kernel_size=3, pool_size=2, 
                   lstm_units=50, gru_units=50, dropout_rate=0.3,
                   use_layer_norm=True, use_bidirectional=True):
        """
        Build the CNNLSTMGRU model according to the specified architecture.
        
        Parameters:
        -----------
        conv_filters : int
            Number of CNN filters
        kernel_size : int
            CNN kernel size
        pool_size : int
            MaxPooling1D pool size
        lstm_units : int
            Number of LSTM units
        gru_units : int
            Number of GRU units
        dropout_rate : float
            Dropout rate
        use_layer_norm : bool
            Whether to use layer normalization
        use_bidirectional : bool
            Whether to use bidirectional LSTM/GRU
        """
        if self.n_features < 1:
            raise ValueError(f"Not enough features to build model: {self.n_features}")
        
        # Define input shape (window_size, n_features, 1)
        input_shape = (self.window_size, self.n_features, 1)
        
        # Define the model architecture
        input_layer = Input(shape=input_shape)
        
        # CNN part (shared)
        cnn_part = TimeDistributed(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu'))(input_layer)
        cnn_part = TimeDistributed(MaxPooling1D(pool_size=pool_size))(cnn_part)
        cnn_part = TimeDistributed(Flatten())(cnn_part)
        
        # LSTM branch
        lstm_branch = LSTM(lstm_units, return_sequences=True)(cnn_part)
        if use_layer_norm:
            lstm_branch = LayerNormalization()(lstm_branch)
        lstm_branch = Dropout(dropout_rate)(lstm_branch)  # Dropout after LayerNorm
        if use_bidirectional:
            lstm_branch = Bidirectional(LSTM(lstm_units))(lstm_branch)  # Bidirectional LSTM
        else:
            lstm_branch = LSTM(lstm_units)(lstm_branch)  # Regular LSTM
        
        # GRU branch
        gru_branch = GRU(gru_units, return_sequences=True)(cnn_part)
        if use_layer_norm:
            gru_branch = LayerNormalization()(gru_branch)
        gru_branch = Dropout(dropout_rate)(gru_branch)  # Dropout after LayerNorm
        if use_bidirectional:
            gru_branch = Bidirectional(GRU(gru_units))(gru_branch)  # Bidirectional GRU
        else:
            gru_branch = GRU(gru_units)(gru_branch)  # Regular GRU
        
        # Concatenate
        merged = concatenate([lstm_branch, gru_branch])
        if use_layer_norm:
            merged = LayerNormalization()(merged)  # LayerNorm after concatenate
        merged = Dropout(dropout_rate)(merged)  # Dropout after concatenate
        
        # Output layer
        output_layer = Dense(self.num_classes, activation='softmax')(merged)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, epochs=None, batch_size=32, patience=10):
        """
        Train the model on the prepared data.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared. Call prepare_data first.")
            return None
        
        # Build the model if it hasn't been built yet
        if self.model is None:
            logger.info("Building model before training")
            self.build_model()
        
        if self.model is None:
            logger.error("Failed to build model")
            return None
        
        if epochs is None:
            epochs = self.max_epochs
        
        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        callbacks = [early_stopping]
        
        # Create sequences from the data
        X_sequences = []
        y_sequences = []
        sequence_length = self.window_size
        
        # Create sequences for training
        for i in range(len(self.X_train) - sequence_length + 1):
            X_sequences.append(self.X_train[i:i+sequence_length])
            y_sequences.append(self.y_train[i+sequence_length-1])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Log the shapes of training data
        logger.info(f"Training sequences shape: X={X_sequences.shape}, y={y_sequences.shape}")
        
        # Create validation sequences if available
        validation_data = None
        if len(self.X_valid) > 0 and len(self.y_valid) > 0:
            X_valid_sequences = []
            y_valid_sequences = []
            
            for i in range(len(self.X_valid) - sequence_length + 1):
                X_valid_sequences.append(self.X_valid[i:i+sequence_length])
                y_valid_sequences.append(self.y_valid[i+sequence_length-1])
            
            if len(X_valid_sequences) > 0 and len(y_valid_sequences) > 0:
                X_valid_sequences = np.array(X_valid_sequences)
                y_valid_sequences = np.array(y_valid_sequences)
                validation_data = (X_valid_sequences, y_valid_sequences)
                
                logger.info(f"Validation sequences shape: X={X_valid_sequences.shape}, y={y_valid_sequences.shape}")
        
        # Train the model
        try:
            logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")
            
            history = self.model.fit(
                X_sequences, y_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0
            )
            
            # Log training results
            final_epoch = len(history.history['loss'])
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            
            logger.info(f"Training completed after {final_epoch} epochs")
            logger.info(f"Final training loss: {final_loss:.4f}, accuracy: {final_accuracy:.4f}")
            
            if validation_data is not None:
                final_val_loss = history.history['val_loss'][-1]
                final_val_accuracy = history.history['val_accuracy'][-1]
                logger.info(f"Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_accuracy:.4f}")
            
            return history
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def predict(self, test_df, features=None):
        """
        Make predictions on test data.
        
        Parameters:
        -----------
        test_df : pd.DataFrame
            Test data
        features : list
            Features to use (if None, use the same features as in training)
        
        Returns:
        --------
        np.ndarray
            Predicted probabilities for each class
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            return np.zeros((len(test_df), self.num_classes))
        
        if features is None:
            features = [col for col in test_df.columns if col != self.target]
        
        try:
            # Make sure test_df has the same features as training data
            missing_cols = [col for col in features if col not in test_df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in test data: {missing_cols}")
                for col in missing_cols:
                    test_df[col] = 0
            
            # Clean data - handle inf and NaN values
            test_df_clean = test_df.replace([np.inf, -np.inf], np.nan)
            if test_df_clean[features].isna().any().any():
                logger.warning("NaN values in test data, filling with means")
                test_df_clean[features] = test_df_clean[features].fillna(test_df_clean[features].mean())
            
            # Transform features
            X_test = self.scaler.transform(test_df_clean[features])
            
            # Create sequences
            X_sequences = []
            sequence_length = self.window_size
            
            # Create overlapping sequences
            for i in range(len(X_test) - sequence_length + 1):
                X_sequences.append(X_test[i:i+sequence_length])
            
            if not X_sequences:
                logger.warning("Not enough data to create sequences for prediction")
                return np.zeros((0, self.num_classes))
            
            X_sequences = np.array(X_sequences)
            
            # Make predictions
            y_pred = self.model.predict(X_sequences)
            
            # Log the shapes
            logger.info(f"Input test data shape: {test_df.shape}")
            logger.info(f"After sequence creation: {X_sequences.shape}")
            logger.info(f"Prediction shape: {y_pred.shape}")
            
            return y_pred
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros((len(test_df), self.num_classes))

    def prepare_single_window(self, window):
        try:
            # Clean and prepare the window data
            window_clean = window.copy()
            
            # Get the expected feature names from when the scaler was fitted
            expected_features = self.scaler.feature_names_in_
            
            # Check if any expected features are missing and add them with zeros
            missing_features = set(expected_features) - set(window_clean.columns)
            if missing_features:
                for feature in missing_features:
                    window_clean[feature] = 0  # Fill with zeros or another appropriate value
                    logger.info(f"Added missing feature {feature} with default values")
            
            # Ensure columns are in the same order as during fit
            window_clean = window_clean[expected_features]
            
            # Apply scaling
            X = self.scaler.transform(window_clean)
            
            # Get the right number of features expected by the model (16 in this case)
            expected_features_count = 16
            if X.shape[1] < expected_features_count:
                # Pad with zeros if needed
                padding = np.zeros((X.shape[0], expected_features_count - X.shape[1]))
                X = np.concatenate([X, padding], axis=1)
            elif X.shape[1] > expected_features_count:
                # Truncate if too many features
                X = X[:, :expected_features_count]
                
            # Reshape for CNN input - ensure correct dimensions
            X = X.reshape(1, self.window_size, expected_features_count, 1)
            return X
        except Exception as e:
            logger.error(f"Error preparing single window: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def predict_single(self, X_sequence):
        try:
            # Properly reshape the input to match the expected shape
            # The model expects (None, 30, 16, 1) but is getting (1, 1, 30, 16) or similar
            if hasattr(X_sequence, 'shape'):
                if len(X_sequence.shape) == 4:
                    # Try to reshape to match expected dimensions
                    if X_sequence.shape[2] == 30 and X_sequence.shape[3] == 16:
                        X_sequence = X_sequence.reshape((1, 30, 16, 1))
                    elif X_sequence.shape[1] == 30 and X_sequence.shape[2] == 16:
                        X_sequence = X_sequence.reshape((1, 30, 16, 1))
                    logger.info(f"Reshaped input to {X_sequence.shape}")
            
            # Predict using the model
            y_pred = self.model.predict(X_sequence, verbose=0)
            
            # Handle prediction results that might be returned as tuple
            if isinstance(y_pred, tuple):
                logger.info(f"Prediction returned as tuple with {len(y_pred)} elements")
                # Use the first element of the tuple (the actual predictions)
                y_pred = y_pred[0]
            
            # Process the predictions
            if hasattr(y_pred, 'shape'):
                logger.info(f"Prediction shape: {y_pred.shape}")
            
            # Get class probabilities and predicted class
            class_probs = y_pred[0]
            predicted_class = int(np.argmax(class_probs))
            
            # Safely access the bin centers if available, otherwise use a fallback
            predicted_return = None
            try:
                if hasattr(self, 'discretizer') and self.discretizer is not None:
                    if hasattr(self.discretizer, 'bin_centers_'):
                        bin_centers = self.discretizer.bin_centers_
                        if predicted_class < len(bin_centers):
                            predicted_return = bin_centers[predicted_class]
                            logger.info(f"Using bin center {predicted_return} for class {predicted_class}")
                        else:
                            logger.warning(f"Predicted class {predicted_class} out of range for bin_centers with length {len(bin_centers)}")
                    else:
                        logger.warning(f"Discretizer doesn't have bin_centers_ attribute")
                else:
                    logger.warning(f"Model doesn't have discretizer attribute or it is None")
            except Exception as e:
                logger.error(f"Error accessing bin centers: {str(e)}")
            
            # Use fallback if bin centers aren't available
            if predicted_return is None:
                num_classes = len(class_probs)
                predicted_return = (predicted_class - (num_classes // 2)) / 100.0
                logger.info(f"Using fallback return estimate: {predicted_return} for class {predicted_class}")
            
            return predicted_class, class_probs
        except Exception as e:
            logger.error(f"Error during single prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    def evaluate(self, X, y):
        """Evaluate the model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Reshape input for CNN-LSTM
        X_reshaped = X.reshape(-1, self.window_size, self.n_features, 1)
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_reshaped, y, verbose=0)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }

def run_categorical_returns_forecast(config_path='config.yaml', output_dir='forecasts', forecast_horizon=30, 
                                    num_classes=5, forecast_periods=30, start_forecast_date=None,
                                    end_forecast_date=None, mode='full_train', model_dir='models'):
    """Run categorical returns forecasting."""
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
    tickers = config.get('tickers', ["SPY", "AOM", "VT", "AGG", "BIL", "^VIX", "VDE", "DBC", "AQMIX", "IWN"])
    window_size = config.get('window_size', 30)
    
    # Load data
    data_result = load_data(tickers, start_date, end_date)
    
    # Check if load_data returns a tuple or just a DataFrame
    if isinstance(data_result, tuple):
        # If it's a tuple, get the first element which should be the DataFrame
        prices_df = data_result[0]
        logger.info(f"Data loaded as tuple, using first element as DataFrame: {type(prices_df)}")
    else:
        # If it's already a DataFrame, use it directly
        prices_df = data_result
    
    # Dictionary to store all forecasts for each ticker
    all_forecasts = {}
    all_detailed_forecasts = {}
    all_metrics = {}
    
    # All forecasts for the test period
    test_period_forecasts = {}

    # Define file paths for saving/loading
    forecasts_file = os.path.join(output_dir, 'return_class_forecasts.csv')
    future_forecasts_file = os.path.join(output_dir, 'return_forecasts_future.csv')
    
    # If in predict mode and end_forecast_date is provided, calculate the number of periods
    if mode == 'predict' and end_forecast_date is not None:
        try:
            start_date_dt = pd.to_datetime(start_forecast_date) if start_forecast_date else None
            end_date_dt = pd.to_datetime(end_forecast_date)
            
            # Get trading days between start and end date
            calendar = mcal.get_calendar('NYSE')
            
            # If start_date not provided, we'll calculate it later when we have data loaded
            if start_date_dt:
                trading_days = calendar.schedule(start_date=start_date_dt, end_date=end_date_dt)
                forecast_periods = len(trading_days)
                logger.info(f"Calculating forecasts from {start_date_dt.strftime('%Y-%m-%d')} to {end_date_dt.strftime('%Y-%m-%d')}")
                logger.info(f"Total trading days to forecast: {forecast_periods}")
                
                if forecast_periods == 0:
                    logger.error("No trading days found between start and end dates")
                    return None
            else:
                logger.info(f"Will forecast until {end_date_dt.strftime('%Y-%m-%d')}")
                # The actual periods will be calculated after loading the data
                
        except Exception as e:
            logger.error(f"Error calculating forecast periods: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    # Check if we're in predict mode and models exist
    if mode == 'predict':
        missing_models = []
        for ticker in tickers:
            model_path = os.path.join(model_dir, f'categorical_model_{ticker}.keras')
            legacy_model_path = os.path.join(model_dir, f'categorical_model_{ticker}.h5')
            
            # Check for new format first, then legacy format
            if os.path.exists(model_path):
                model_file_to_use = model_path
            elif os.path.exists(legacy_model_path):
                model_file_to_use = legacy_model_path
                logger.warning(f"Using legacy model format for {ticker}. Consider retraining to update format.")
            else:
                missing_models.append(ticker)
        
        if missing_models:
            logger.error(f"Cannot run in predict mode. Missing models for: {', '.join(missing_models)}")
            logger.error("Please run with --full_train or --incremental_train first.")
            return None
    
    # For predict-only mode, we skip the walk-forward training
    if mode == 'predict':
        # Load data
        data_result = load_data(tickers, start_date, end_date)
        
        # Check if load_data returns a tuple or just a DataFrame
        if isinstance(data_result, tuple):
            # If it's a tuple, get the first element which should be the DataFrame
            data = data_result[0]
            logger.info(f"Data loaded as tuple in predict mode, using first element as DataFrame: {type(data)}")
        else:
            # If it's already a DataFrame, use it directly
            data = data_result
        
        # Dictionary to store future forecasts
        future_forecasts = {}
        detailed_forecasts = {}
        
        # If end_forecast_date is provided but no start_forecast_date, use the last data date
        if end_forecast_date is not None and start_forecast_date is None:
            last_data_date = data.index[-1]
            start_date_dt = last_data_date + pd.Timedelta(days=1)
            
            # Calculate forecast periods using trading days
            calendar = mcal.get_calendar('NYSE')
            trading_days = calendar.schedule(start_date=start_date_dt, end_date=pd.to_datetime(end_forecast_date))
            forecast_periods = len(trading_days)
            
            logger.info(f"Calculating forecasts from {start_date_dt.strftime('%Y-%m-%d')} to {end_date_dt.strftime('%Y-%m-%d')}")
            logger.info(f"Total trading days to forecast: {forecast_periods}")
            
            if forecast_periods == 0:
                logger.error("No trading days found between start and end dates")
                return None
            
            # Update start_forecast_date
            start_forecast_date = start_date_dt.strftime('%Y-%m-%d')
        
        # Generate future forecasts
        future_forecasts = generate_future_forecasts(
            tickers, 
            test_period_forecasts, 
            forecast_periods, 
            num_classes, 
            window_size, 
            output_dir, 
            start_forecast_date,
            end_forecast_date,
            model_dir=model_dir,
            historical_data=data
        )
        
        return future_forecasts
    
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}")
            
            # Create target column (forward returns)
            target_col = f"{ticker}_fwd_{forecast_horizon}d"
            prices_df[target_col] = prices_df[ticker].pct_change(periods=forecast_horizon, fill_method=None).shift(-forecast_horizon)
            
            # Split dataset
            train_data, valid_data, test_data = split_dataset(prices_df, train_size=0.7, valid_size=0.1)
            
            # Create technical indicators
            lags = 5  # Number of lag features to create
            train_features = calculate_technical_indicators(train_data[[ticker]])
            valid_features = calculate_technical_indicators(valid_data[[ticker]])
            test_features = calculate_technical_indicators(test_data[[ticker]])
            
            # Make sure the target column is preserved
            train_features[target_col] = train_data[target_col]
            valid_features[target_col] = valid_data[target_col]
            test_features[target_col] = test_data[target_col]
            
            # Create lagged features for the target
            for lag in range(1, lags + 1):
                lag_col = f"{target_col}_lag{lag}"
                train_features[lag_col] = train_features[target_col].shift(lag)
                valid_features[lag_col] = valid_features[target_col].shift(lag)
                test_features[lag_col] = test_features[target_col].shift(lag)
            
            # Drop NaN values
            train_features = train_features.dropna()
            valid_features = valid_features.dropna()
            test_features = test_features.dropna()
            
            # Split the test data into monthly chunks - use groupby with Grouper instead of resample
            test_months = []
            for month_end, group in test_features.groupby(pd.Grouper(freq='ME')):
                if not group.empty:
                    test_months.append(group)
            
            logger.info(f"Split test data into {len(test_months)} monthly chunks")
            
            # Initialize a Series to store all forecasts for this ticker
            ticker_forecasts = pd.Series(dtype=float)
            
            # Current training and validation sets
            current_train = train_features.copy()
            current_valid = valid_features.copy()
            
            # Log initial data sizes
            logger.info(f"Initial training size: {len(current_train)}, validation size: {len(current_valid)}")
            
            # For each month in the test set
            for i, month_data in enumerate(test_months):
                if len(month_data) == 0:
                    continue
                    
                logger.info(f"Processing month {i+1}/{len(test_months)} for {ticker}")
                logger.info(f"Month dates: {month_data.index[0]} to {month_data.index[-1]}, {len(month_data)} days")
                
                # Initialize model
                model = CNNLSTMGRU_Model(
                    window_size=window_size,
                    num_classes=num_classes,
                    max_epochs=50
                )
                
                # Prepare data for this iteration
                model.prepare_data(
                    current_train, current_valid, month_data, 
                    target=target_col, sequence_length=window_size
                )
                
                # Instead of unpacking, access the attributes directly
                if hasattr(model, 'X_train') and model.X_train is not None and len(model.X_train) > 0:
                    # Train the model
                    model.build_model()  # Build the model first
                    model.train(epochs=10, batch_size=32)
                    
                    # Forecast for this month
                    month_forecasts = []
                    month_dates = []
                    
                    # For each day in the month, make a prediction
                    for j, date in enumerate(month_data.index):
                        try:
                            # Get window of data before this date
                            # We need at least window_size days of data before the current date
                            all_data = pd.concat([current_train, current_valid])
                            
                            # Find data up to but not including the current date
                            prev_data = all_data[all_data.index < date]
                            
                            if len(prev_data) >= window_size:
                                # Get the last window_size days
                                window_data = prev_data.iloc[-window_size:].copy()
                                
                                # Scale data using the model's scaler
                                scaled_data = model.scaler.transform(window_data[model.feature_names])
                                
                                # Reshape for model input
                                X_window = np.expand_dims(scaled_data, axis=0)
                                
                                # Make prediction
                                pred_probs = model.model.predict(X_window, verbose=0)
                                pred_class = np.argmax(pred_probs, axis=1)[0]
                                
                                # Map the predicted class to a return value
                                class_ranges = model.discretizer.bin_edges_[0]
                                if pred_class < len(class_ranges) - 1:
                                    class_min = class_ranges[pred_class]
                                    class_max = class_ranges[pred_class + 1]
                                    predicted_return = (class_min + class_max) / 2
                                else:
                                    class_min = class_ranges[pred_class]
                                    prev_interval = class_ranges[pred_class] - class_ranges[pred_class - 1]
                                    predicted_return = class_min + (prev_interval / 2)
                                
                                month_forecasts.append(predicted_return)
                                month_dates.append(date)
                            else:
                                logger.warning(f"Insufficient data for date {date}: only have {len(prev_data)} observations but need {window_size}")
                        except Exception as e:
                            logger.error(f"Error predicting for date {date}: {str(e)}")
                    
                    # Create forecast series if we have any forecasts
                    if month_forecasts:
                        forecast_series = pd.Series(month_forecasts, index=month_dates)
                        logger.info(f"Made {len(forecast_series)} forecasts for month {i+1}")
                        ticker_forecasts = pd.concat([ticker_forecasts, forecast_series])
                    else:
                        logger.warning(f"No forecasts generated for month {i+1}")
                    
                    # After forecasting, add this month's data to training for next iteration
                    combined_data = pd.concat([current_train, current_valid, month_data])
                    
                    # Recalculate validation set size (10% of the combined data)
                    valid_size_rows = int(len(combined_data) * 0.1)
                    
                    # Update training and validation sets
                    current_valid = combined_data.iloc[-valid_size_rows:]
                    current_train = combined_data.iloc[:-valid_size_rows]
                    
                    logger.info(f"Updated training size: {len(current_train)}, validation size: {len(current_valid)}")
                    
                    # After model training is complete
                    if hasattr(model, 'model') and model.model is not None:
                        # Save the model using the newer .keras format
                        model_path = os.path.join(model_dir, f'categorical_model_{ticker}.keras')
                        model.model.save(model_path)
                        logger.info(f"Saved model for {ticker} to {model_path}")
                        
                        # Save the discretizer
                        if hasattr(model, 'discretizer') and model.discretizer is not None:
                            discretizer_path = os.path.join(model_dir, f'discretizer_{ticker}.pkl')
                            with open(discretizer_path, 'wb') as f:
                                pickle.dump(model.discretizer, f)
                            logger.info(f"Saved discretizer for {ticker}")
                        
                        # Save the scaler
                        if hasattr(model, 'scaler') and model.scaler is not None:
                            scaler_path = os.path.join(model_dir, f'scaler_{ticker}.pkl')
                            with open(scaler_path, 'wb') as f:
                                pickle.dump(model.scaler, f)
                            logger.info(f"Saved scaler for {ticker}")
                else:
                    logger.warning(f"Could not prepare data for {ticker}, month {i+1}")
            
            # Store all forecasts for this ticker
            if not ticker_forecasts.empty:
                test_period_forecasts[ticker] = ticker_forecasts
                logger.info(f"Generated {len(ticker_forecasts)} total forecasts for {ticker}")
            else:
                logger.warning(f"No forecasts generated for {ticker}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            continue
    
    # Save test period forecasts
    if test_period_forecasts:
        test_forecast_df = pd.DataFrame(test_period_forecasts)
        test_file = os.path.join(output_dir, 'return_class_forecasts.csv')
        test_forecast_df.to_csv(test_file)
        logger.info(f"Test period forecasts saved to {test_file}")
    else:
        logger.warning("No test period forecasts generated")
    
    # Generate future forecasts if requested
    future_forecasts = {}
    if forecast_periods > 0:
        # Call the generate_future_forecasts function with the historical data
        future_forecasts = generate_future_forecasts(
            tickers, 
            test_period_forecasts, 
            forecast_periods, 
            num_classes, 
            window_size, 
            output_dir, 
            start_forecast_date,
            end_forecast_date,
            model_dir=model_dir,
            historical_data=data
        )
    
    # Return the test period forecasts
    return test_period_forecasts

def generate_future_forecasts(tickers, test_forecasts, forecast_periods, num_classes, window_size, 
                             output_dir, start_forecast_date=None, end_forecast_date=None, model_dir='models', historical_data=None):
    """Generate future forecasts using trained models."""
    # Ensure paths are relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
        
    # Initialize empty DataFrame for all forecasts
    all_forecasts = pd.DataFrame()
    
    # Load existing forecasts if they exist
    forecast_file = os.path.join(output_dir, 'return_forecasts_future.csv')
    categorical_forecast_file = os.path.join(output_dir, 'categorical_return_forecasts_future.csv')
    
    if os.path.exists(forecast_file):
        try:
            all_forecasts = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded existing future forecasts from {forecast_file}")
        except Exception as e:
            logger.warning(f"Could not load existing forecasts: {str(e)}")
    
    # Initialize categorical forecast dataframes
    all_categorical_forecasts = pd.DataFrame()
    
    # For each ticker
    for ticker in tickers:
        try:
            logger.info(f"Processing forecasts for {ticker}")
            
            # Get the last forecast date for this ticker
            last_forecast_date = None
            if ticker in all_forecasts.columns:
                # Get the last date that's not NaN for this ticker
                ticker_forecasts = all_forecasts[ticker].dropna()
                if not ticker_forecasts.empty:
                    last_forecast_date = ticker_forecasts.index[-1]
            
            # If no last forecast date, use start_forecast_date
            if last_forecast_date is None:
                logger.warning(f"No past forecasts for {ticker}, using {start_forecast_date} as start date")
                last_forecast_date = start_forecast_date
            
            # Load the model for this ticker
            model_path = os.path.join(project_root, model_dir, f"categorical_model_{ticker}.keras")
            
            if not os.path.exists(model_path):
                logger.warning(f"No model found for {ticker}, skipping forecasts")
                continue
                
            try:
                # Initialize model without ticker parameter (it's not accepted by the constructor)
                model = CNNLSTMGRU_Model(num_classes=num_classes, window_size=window_size)
                
                # Set ticker as an attribute after creation for reference
                model.ticker = ticker
                
                # Load the model - define a custom load method since it doesn't exist
                model.model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded model for {ticker} from {model_path}")
                
                # Load discretizer if available
                discretizer_path = os.path.join(project_root, model_dir, f"discretizer_{ticker}.pkl")
                if os.path.exists(discretizer_path):
                    with open(discretizer_path, 'rb') as f:
                        model.discretizer = pickle.load(f)
                    logger.info(f"Loaded discretizer for {ticker}")
                
                # Load scaler if available
                scaler_path = os.path.join(project_root, model_dir, f"scaler_{ticker}.pkl")
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        model.scaler = pickle.load(f)
                    logger.info(f"Loaded scaler for {ticker}")
                
                # Define prepare_single_window function for the model if it doesn't have one
                if not hasattr(model, 'prepare_single_window'):
                    def prepare_single_window_func(self, window):
                        """Prepare a single window of data for prediction."""
                        try:
                            # Clean and prepare the window data
                            window_clean = window.copy()
                            
                            # Get the expected feature names from when the scaler was fitted
                            expected_features = self.scaler.feature_names_in_
                            
                            # Check if any expected features are missing and add them with zeros
                            missing_features = set(expected_features) - set(window_clean.columns)
                            if missing_features:
                                for feature in missing_features:
                                    window_clean[feature] = 0  # Fill with zeros or another appropriate value
                                    logger.info(f"Added missing feature {feature} with default values")
                            
                            # Ensure columns are in the same order as during fit
                            window_clean = window_clean[expected_features]
                            
                            # Apply scaling
                            X = self.scaler.transform(window_clean)
                            
                            # Get the right number of features expected by the model (16 in this case)
                            expected_features_count = 16
                            if X.shape[1] < expected_features_count:
                                # Pad with zeros if needed
                                padding = np.zeros((X.shape[0], expected_features_count - X.shape[1]))
                                X = np.concatenate([X, padding], axis=1)
                            elif X.shape[1] > expected_features_count:
                                # Truncate if too many features
                                X = X[:, :expected_features_count]
                                
                            # Reshape for CNN input - ensure correct dimensions
                            X = X.reshape(1, self.window_size, expected_features_count, 1)
                            return X
                        except Exception as e:
                            logger.error(f"Error preparing single window: {str(e)}")
                            logger.error(traceback.format_exc())
                            return None
                    
                    # Attach the function to the model instance
                    model.prepare_single_window = types.MethodType(prepare_single_window_func, model)
                
                # Define predict_single method for the model if it doesn't have one
                if not hasattr(model, 'predict_single'):
                    def predict_single_func(self, X_sequence):
                        """Make a prediction for a single preprocessed window."""
                        try:
                            if X_sequence is None:
                                return None
                                
                            # Reshape the input to match the expected shape
                            # The model expects (None, 30, 16, 1) but is getting (1, 1, 30, 16)
                            if hasattr(X_sequence, 'shape') and len(X_sequence.shape) == 4:
                                if X_sequence.shape == (1, 1, 30, 16):
                                    # Reshape to match expected dimensions
                                    X_sequence = X_sequence.reshape(1, 30, 16, 1)
                                    logger.info(f"Reshaped input from {(1, 1, 30, 16)} to {X_sequence.shape}")
                            
                            # Predict using the model
                            y_pred = self.model.predict(X_sequence, verbose=0)
                            
                            # Rest of your existing code
                            predicted_class = np.argmax(y_pred[0])
                            
                            # Use the discretizer to get the return value for this class
                            if hasattr(self, 'discretizer') and self.discretizer is not None:
                                bin_centers = getattr(self.discretizer, 'bin_centers_', None)
                                if bin_centers is not None and predicted_class < len(bin_centers):
                                    predicted_return = bin_centers[predicted_class]
                                else:
                                    # Fallback if bin_centers not available or index out of range
                                    predicted_return = (predicted_class - (self.num_classes // 2)) / 100.0
                                    
                            # Create results dict with class probabilities
                            results = {
                                f"{self.ticker}_PredictedClass": predicted_class,
                                f"{self.ticker}_PredictedReturn": predicted_return
                            }
                            
                            # Add class probabilities
                            for i, prob in enumerate(y_pred[0]):
                                results[f"{self.ticker}_Class{i}_Prob"] = float(prob)
                                
                            return results
                        except Exception as e:
                            logger.error(f"Error during single prediction: {str(e)}")
                            logger.error(traceback.format_exc())
                            return None
                    
                    # Attach the function to the model instance  
                    model.predict_single = types.MethodType(predict_single_func, model)
                
                # Initialize empty series for this ticker's forecasts
                ticker_forecasts = pd.Series(dtype=float, name=ticker)
                
                # Create a list for categorical forecast data
                categorical_forecast_data = []
                
                # Start with historical data if available
                if historical_data is not None and ticker in historical_data.columns:
                    logger.info(f"Using downloaded historical data for {ticker}")
                    
                    try:
                        # Create a deep copy to avoid modifying the original data
                        ticker_data = pd.DataFrame()
                        ticker_data[ticker] = historical_data[ticker].copy()
                        
                        # Calculate technical indicators
                        technical_features = calculate_technical_indicators(ticker_data)
                        
                        # Ensure we have enough data
                        if len(technical_features) >= window_size:
                            # Get most recent window_size days for prediction
                            current_window = technical_features.iloc[-window_size:]
                            
                            # Generate forecasts for each period
                            for i in range(forecast_periods):
                                # Calculate start and end dates for this period
                                if i == 0:
                                    start_date = last_forecast_date
                                else:
                                    # Add one month to previous start date
                                    start_date = (pd.to_datetime(start_date) + pd.DateOffset(months=i)).strftime('%Y-%m-%d')
                                
                                end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
                                
                                # Limit by end_forecast_date if specified
                                if end_forecast_date is not None and pd.to_datetime(end_date) > pd.to_datetime(end_forecast_date):
                                    end_date = end_forecast_date
                                
                                # Skip if start_date is after end_date
                                if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                                    continue
                                
                                # Get the list of trading days in this period
                                month_dates = pd.date_range(start=start_date, end=end_date, freq='B')
                                
                                # Fix: Check if test_forecasts is a dict and handle accordingly
                                if isinstance(test_forecasts, dict):
                                    # If it's a dictionary of DataFrames/Series, try to get all dates
                                    all_dates = set()
                                    for ticker_data in test_forecasts.values():
                                        if hasattr(ticker_data, 'index'):
                                            all_dates.update(ticker_data.index)
                                    if all_dates:
                                        # Convert to DataFrame index and filter
                                        all_dates_idx = pd.DatetimeIndex(sorted(all_dates))
                                        month_dates = month_dates[month_dates.isin(all_dates_idx)]
                                    else:
                                        # If no dates found, just use all business days
                                        logger.info(f"No dates found in test_forecasts, using all business days")
                                else:
                                    # Original code if test_forecasts is a DataFrame
                                    month_dates = month_dates[month_dates.isin(test_forecasts.index)]
                                
                                if len(month_dates) == 0:
                                    logger.info(f"No new dates to forecast for {ticker} in period {i+1}")
                                    continue
                                
                                # Make prediction for each date in the month
                                for prediction_date in month_dates:
                                    # Make prediction using the model
                                    try:
                                        logger.info(f"Preparing window for {ticker} for date {prediction_date}")
                                        X_sequence = model.prepare_single_window(current_window)
                                        
                                        if X_sequence is not None:
                                            # Make the prediction
                                            predicted_class, class_probs = model.predict_single(X_sequence)
                                            
                                            if predicted_class is not None and class_probs is not None:
                                                # Safely access the bin centers if available, otherwise use a fallback
                                                predicted_return = None
                                                try:
                                                    if hasattr(model, 'discretizer') and model.discretizer is not None:
                                                        if hasattr(model.discretizer, 'bin_centers_'):
                                                            bin_centers = model.discretizer.bin_centers_
                                                            if predicted_class < len(bin_centers):
                                                                predicted_return = bin_centers[predicted_class]
                                                                logger.info(f"Using bin center {predicted_return} for class {predicted_class}")
                                                            else:
                                                                logger.warning(f"Predicted class {predicted_class} out of range for bin_centers with length {len(bin_centers)}")
                                                        else:
                                                            logger.warning(f"Discretizer doesn't have bin_centers_ attribute")
                                                    else:
                                                        logger.warning(f"Model doesn't have discretizer attribute or it is None")
                                                except Exception as e:
                                                    logger.error(f"Error accessing bin centers: {str(e)}")
                                                
                                                # Use fallback if bin centers aren't available
                                                if predicted_return is None:
                                                    num_classes = len(class_probs)
                                                    predicted_return = (predicted_class - (num_classes // 2)) / 100.0
                                                    logger.info(f"Using fallback return estimate: {predicted_return} for class {predicted_class}")
                                                
                                                # Add to forecast data
                                                categorical_forecast_data.append({
                                                    'Date': prediction_date,
                                                    'Ticker': ticker,
                                                    'PredictedClass': predicted_class,
                                                    'PredictedReturn': predicted_return,
                                                    'Class0_Prob': class_probs[0],
                                                    'Class1_Prob': class_probs[1],
                                                    'Class2_Prob': class_probs[2],
                                                    'Class3_Prob': class_probs[3],
                                                    'Class4_Prob': class_probs[4]
                                                })
                                                
                                                # Also store for return forecasts
                                                ticker_forecasts[prediction_date.strftime('%Y-%m-%d')] = predicted_return
                                                
                                                logger.info(f"Generated forecast for {ticker} on {prediction_date}: Class {predicted_class}, Return {predicted_return:.4f}")
                                            else:
                                                logger.warning(f"No valid prediction for {ticker} on {prediction_date}")
                                        else:
                                            logger.warning(f"Failed to prepare window for {ticker}")
                                    except Exception as e:
                                        logger.error(f"Error in prediction processing for {ticker} on {prediction_date}: {str(e)}")
                                        logger.error(traceback.format_exc())
                        else:
                            logger.warning(f"Not enough data after processing for {ticker} ({len(technical_features)} rows), need {window_size}")
                    except Exception as e:
                        logger.error(f"Error preparing features for {ticker}: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"No historical data for {ticker}, skipping forecasts")
                
                # Add forecasts to the all_forecasts DataFrame
                if not ticker_forecasts.empty:
                    if ticker not in all_forecasts.columns:
                        all_forecasts[ticker] = pd.Series(dtype=float)
                    
                    # Update with new forecasts (preserving existing ones)
                    for date, value in ticker_forecasts.items():
                        all_forecasts.loc[date, ticker] = value
                    
                    logger.info(f"Added {len(ticker_forecasts)} forecasts for {ticker}")
                    
                    # Add categorical forecast data if we have any
                    if categorical_forecast_data:
                        for forecast in categorical_forecast_data:
                            # Check if we have a DataFrame and add as a new row
                            if all_categorical_forecasts.empty:
                                all_categorical_forecasts = pd.DataFrame([forecast])
                            else:
                                # Update existing data or add new columns
                                for key, value in forecast.items():
                                    all_categorical_forecasts.loc[0, key] = value
                else:
                    logger.warning(f"No forecasts generated for {ticker}")
            
            except Exception as e:
                logger.error(f"Error generating future forecasts for {ticker}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
        except Exception as e:
            logger.error(f"Unexpected error processing {ticker}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # Save all forecasts
    try:
        all_forecasts.sort_index(inplace=True)
        all_forecasts.to_csv(forecast_file)
        logger.info(f"Future forecasts saved to {forecast_file}")
        
        # Save categorical forecasts
        if not all_categorical_forecasts.empty:
            all_categorical_forecasts.to_csv(categorical_forecast_file)
            logger.info(f"Categorical forecasts saved to {categorical_forecast_file}")
    except Exception as e:
        logger.error(f"Error saving forecasts: {str(e)}")
    
    return all_forecasts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Categorical Returns Forecasting')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon (days)')
    parser.add_argument('--classes', type=int, default=5, help='Number of return classes')
    parser.add_argument('--periods', type=int, default=30, help='Number of forecast periods (ignored if end_date is provided)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date for forecasting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for forecasting (YYYY-MM-DD)')
    
    # Execution mode arguments (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full_train', action='store_true', help='Run full walk-forward training')
    # group.add_argument('--incremental_train', action='store_true', help='Train on new data only')
    group.add_argument('--predict', action='store_true', help='Make predictions without retraining')
    
    args = parser.parse_args()
    
    if args.full_train:
        run_mode = 'full_train'
    elif args.predict:
        run_mode = 'predict'
    else:
        logger.error("Invalid execution mode. Please use --full_train or --predict.")
        sys.exit(1)

    run_categorical_returns_forecast(
        config_path=args.config,
        output_dir=args.output,
        forecast_horizon=args.horizon,
        num_classes=args.classes,
        forecast_periods=args.periods,
        start_forecast_date=args.start_date,
        end_forecast_date=args.end_date,
        mode=run_mode,
        model_dir=args.model_dir
    )

    print("\nForecasting completed. Results saved to:", args.output)
    
    # Clean up TensorFlow/Keras resources and force exit
    try:
        if hasattr(K, 'clear_session'):
            K.clear_session()
    except:
        pass
        
    # Close any open matplotlib figures
    try:
        plt.close('all')
    except:
        pass
        
    # Force exit
    sys.exit(0)