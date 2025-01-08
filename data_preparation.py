import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# List of SPDR Sector ETFs
etfs = [
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"
]

# Download historical data from 2000
data = {}
for etf in etfs:
    df = yf.download(etf, start="2000-01-01", end="2024-12-31")
    # Access the Close price using the MultiIndex
    close_prices = df[('Close', etf)]
    data[etf] = close_prices

# Create DataFrame with the date index from one of the Series
sample_dates = next(iter(data.values())).index
df = pd.DataFrame(index=sample_dates)

# Add each ETF's close prices to the DataFrame
for etf in etfs:
    df[etf] = data[etf]

print("\nFirst few rows of the combined DataFrame:")
print(df.head())

# Calculate daily log returns
log_returns = np.log(df / df.shift(1))

# Calculate historical volatility (21-day rolling standard deviation)
volatility = log_returns.rolling(window=21).std()

# Calculate technical indicators
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Calculate technical indicators for all ETFs
technical_indicators = {}
for etf in etfs:
    technical_indicators[f'{etf}_SMA10'] = calculate_sma(df[etf], 10)
    technical_indicators[f'{etf}_SMA50'] = calculate_sma(df[etf], 50)
    technical_indicators[f'{etf}_RSI'] = calculate_rsi(df[etf])
    macd, signal = calculate_macd(df[etf])
    technical_indicators[f'{etf}_MACD'] = macd
    technical_indicators[f'{etf}_MACD_Signal'] = signal

# Add technical indicators to the main DataFrame
for indicator_name, indicator_data in technical_indicators.items():
    df[indicator_name] = indicator_data

# Fill missing values using forward fill
df = df.ffill()
df = df.dropna()

# Normalize features
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)

# Split the data
train_size = int(len(normalized_df) * 0.7)
val_size = int(len(normalized_df) * 0.15)

train_data = normalized_df[:train_size]
val_data = normalized_df[train_size:train_size + val_size]
test_data = normalized_df[train_size + val_size:]

# Save the datasets
train_data.to_csv('Data/train_data.csv')
val_data.to_csv('Data/val_data.csv')
test_data.to_csv('Data/test_data.csv')

print("\nDataset shapes:")
print(f"Training set: {train_data.shape}")
print(f"Validation set: {val_data.shape}")
print(f"Test set: {test_data.shape}")