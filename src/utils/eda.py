import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import requests
import seaborn as sns
import pypfopt
from arch import arch_model
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import het_arch
import warnings
import contextlib
import io
import os
warnings.filterwarnings('ignore')

def price_volume(tickers, df_price, df_volume):
    n_rows = int(np.ceil(len(tickers) / 3))
    fig, axs = plt.subplots(n_rows, 3, figsize=(20, 4 * n_rows), sharex=False)
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i, ax in enumerate(axs):
        if i < len(tickers):
            ticker = tickers[i]

            # Plot the closing prices
            ax.plot(df_price[ticker], label=ticker)
            ax.set_ylabel("Price")
            ax.tick_params(axis='y')
            ax.legend()

            # Plot the volume
            ax2 = ax.twinx()
            ax2.bar(df_volume[ticker].index, df_volume[ticker], alpha=0.7, color="gray", label="Volume")
            ax2.set_ylabel("Volume")
            ax2.tick_params(axis='y')

            ax.set_title(f'{ticker} Closing Prices and Volume')
            ax.grid(True)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# Volatilty Clustering
def volatility_clustering_analysis(tickers, time_series):
    n_rows = np.ceil(len(tickers)/3).astype(int)

    plt.figure(figsize=(20, 4*n_rows))
    for i, ticker in enumerate(tickers):
        # Calculate the daily returns
        #daily_returns = time_series[ticker].pct_change().dropna()
        plt.subplot(n_rows, 3, i+1)

        # Plot the daily returns
        plt.plot(time_series[ticker])
        plt.title(f'{ticker} Daily Returns (Volatility Clustering)')
        plt.xlabel('Date')
        plt.ylabel('Daily Returns')
        p_value = het_arch(time_series[ticker]*100)[1]
        plt.text(0.5, 0.9, f'ARCH Test p-value: {p_value:.3f}', ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

# Return Distribtion
def plot_hist(tickers, time_series):

    n_rows = np.ceil(len(tickers)/3).astype(int)
    plt.figure(figsize=(20, 4*n_rows))
    for i, ticker in enumerate(tickers):
        plt.subplot(n_rows, 3, i+1)
        # Plot the daily returns
        time_series[ticker].hist(bins=50)
        kurtosis = time_series[ticker].kurt()
        skew = time_series[ticker].skew()
        plt.text(0.5, 0.9, f'Skew, Kurtosis: {skew:.2f}, {kurtosis:.2f}', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{ticker} Daily Returns Histograms')
        plt.xlabel('Daily Returns')
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def corr_plot(df, target):
    plt.figure(figsize=(8,6))
    corr = df.corr()
    sort_index = corr[target].abs().sort_values(ascending=False).index
    corr = corr.loc[sort_index, sort_index]
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=np.bool_))
    heatmap = sns.heatmap(corr, mask = mask, cmap=cmap, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=10)

def best_garch(daily_returns, max_p=5, max_o=2, max_q=5, dist_list=['normal', 't', 'skewt']):
    acf_list = acf(daily_returns, nlags=max_q)
    pacf_list = pacf(daily_returns, nlags=max_p)

    # Select lags where values exceed a significance threshold
    threshold = 1.96 / np.sqrt(len(daily_returns))
    sig_p = [int(p) for p in np.where(np.abs(acf_list) > threshold)[0] if p > 0]
    sig_q = [int(q) for q in np.where(np.abs(pacf_list) > threshold)[0] if q > 0]

    # Ensure at least one candidate for p and q
    p_list = sig_p or [1]
    q_list = sig_q or [1]

    best_model = None
    best_bic = np.inf

    for p in p_list:
        for q in q_list:
            for o in range(0,max_o):
                for dist in dist_list:
                    garch_model = arch_model(daily_returns, vol='garch', p=p, o=o, q=q, dist = dist)
                    with contextlib.redirect_stdout(io.StringIO()):
                        garch_fit = garch_model.fit(show_warning = False)
                    if garch_fit.bic < best_bic:
                        best_bic = garch_fit.bic
                        best_model = (p, q, o, dist)

    return best_model if best_model else (1, 1)

def plot_acf_pacf(tickers, time_series, best_garch_param):
    fig, axes = plt.subplots(len(tickers), 2, figsize=(10, 3 * len(tickers)))

    for i, ticker in enumerate(tickers):
        #daily_returns = time_series[ticker].pct_change().dropna()
        plot_acf(time_series[ticker]**2, ax=axes[i, 0])
        axes[i, 0].text(0.5, 0.9, f'Best (p,q, o, dist): {best_garch_param.get(ticker)}', ha='center', va='center', transform=axes[i, 0].transAxes)
        axes[i, 0].set_title(f'{ticker} ACF of Returns')
        plot_pacf(time_series[ticker]**2, ax=axes[i, 1])
        axes[i, 1].set_title(f'{ticker} PACF of Returns')

    plt.tight_layout()
    plt.show()
