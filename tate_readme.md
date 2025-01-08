# ETF Volatility Forecasting with LSTM

This project implements a deep learning approach to forecast the volatility of SPDR Sector ETFs using LSTM (Long Short-Term Memory) neural networks. The model is designed to outperform traditional time series models like GARCH in terms of forecast accuracy.

## Project Structure
project_root/
├── Data/
│ ├── train_data.csv
│ ├── val_data.csv
│ └── test_data.csv
├── Models/
│ └── volatility_lstm_model.h5
├── data_preparation.py
└── volatility_forecasting.py

## Features

- Downloads historical data for 11 SPDR Sector ETFs
- Calculates various technical indicators:
  - Simple Moving Averages (10-day and 50-day)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
- Implements LSTM-based deep learning model for volatility forecasting
- Includes early stopping to prevent overfitting
- Provides comprehensive evaluation metrics (MSE, MAE, RMSE)
- Visualizes model performance and predictions

## Requirements
tensorflow
pandas
numpy
scikit-learn
yfinance
matplotlib
arch

## Usage

1. First, prepare the data:
bash
python data_preparation.py

This script will:
- Download historical ETF data
- Calculate technical indicators
- Normalize the features
- Split the data into training, validation, and test sets

2. Then, train and evaluate the model:
bash
python volatility_forecasting.py

This script will:
- Build and train the LSTM model
- Evaluate model performance
- Generate visualization plots
- Save the trained model

## Model Architecture

The LSTM model consists of:
- Input layer with sequence length of 20 time steps
- First LSTM layer with 50 units and return sequences
- Dropout layer (20% dropout rate)
- Second LSTM layer with 50 units
- Dropout layer (20% dropout rate)
- Dense output layer

## Data Processing

- Historical data from 2000 to 2024
- Features include:
  - Close prices
  - Technical indicators
  - Historical volatility
- Data is normalized using MinMaxScaler
- Dataset split:
  - 70% Training
  - 15% Validation
  - 15% Testing

## Performance Metrics

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Visualization

The script generates plots showing:
- Training and validation loss over epochs
- Actual vs predicted volatility comparison

## Future Improvements

- Implement hyperparameter tuning
- Add GARCH model comparison
- Include more technical indicators
- Experiment with different architectures
- Add cross-validation