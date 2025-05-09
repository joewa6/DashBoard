import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
import ccxt
import matplotlib.pyplot as plt

# -------------------------------
# Helper Functions
# -------------------------------

def compute_slope(y):
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

def rolling_slope(series, window):
    return series.rolling(window).apply(lambda y: compute_slope(y.values), raw=False)

def z_score(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def local_entropy(series, window=60, bins=10):
    def entropy_func(x):
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]
        return entropy(hist)
    return series.rolling(window).apply(entropy_func, raw=True)

# -------------------------------
# Feature Engineering
# -------------------------------

def compute_features(df):
    df = df.copy()

    df['grad_15m'] = rolling_slope(df['Close'], 15)
    df['grad_1h'] = rolling_slope(df['Close'], 60)
    df['grad_1d'] = rolling_slope(df['Close'], 240)
    df['grad_ratio_15m_1d'] = df['grad_15m'] / df['grad_1d']
    df['grad_ratio_1h_1d'] = df['grad_1h'] / df['grad_1d']

    df['rolling_mean_1d'] = df['Close'].rolling(240).mean()
    df['rolling_std_1d'] = df['Close'].rolling(240).std()
    df['z_score_1d'] = z_score(df['Close'], 240)

    df['price_diff_15m'] = df['Close'].pct_change(15)
    df['price_diff_1h'] = df['Close'].pct_change(60)

    df['volatility_15m'] = df['Close'].rolling(15).std()
    df['volatility_1h'] = df['Close'].rolling(60).std()
    df['price_range_1h'] = df['Close'].rolling(60).apply(lambda x: x.max() - x.min())

    df['cycle_score'] = (df['Close'] - df['Close'].rolling(60).min()) / (
                         df['Close'].rolling(60).max() - df['Close'].rolling(60).min())

    df['slope_flip_signal'] = np.sign(df['grad_15m']) != np.sign(df['grad_1h'])
    df['price_spike_ratio'] = df['Close'] / df['Close'].rolling(15).max()
    df['drawdown_recovery'] = (df['Close'] - df['Close'].rolling(60).min()) / df['Close'].rolling(60).std()
    df['local_entropy'] = local_entropy(df['Close'].pct_change().fillna(0), window=60)

    return df

# -------------------------------
# Signal Generation
# -------------------------------

def generate_trade_signals(df):
    df = df.copy()
    df['signal'] = "HOLD"

    buy_condition = (
        (df['grad_ratio_15m_1d'] > 2.0) &
        (df['z_score_1d'] < -0.5) &
        (df['slope_flip_signal']) &
        (df['price_spike_ratio'] < 1.02) &
        (df['cycle_score'] < 0.3)
    )

    sell_condition = (
        (df['grad_15m'] < 0) |
        (df['z_score_1d'] > 0.5) |
        (df['drawdown_recovery'] < 0)
    )

    df.loc[buy_condition, 'signal'] = "BUY"
    df.loc[sell_condition, 'signal'] = "SELL"
    return df

# -------------------------------
# Streamlit Dashboard
# -------------------------------

st.set_page_config(page_title="Crypto Signal Dashboard", layout="wide")
st.title("Crypto Trading Signal Dashboard")
st.markdown("**Live BTC/EUR signals powered by statistical gradient-based analysis.**")

# Fetch data
with st.spinner("Fetching data from Binance..."):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/EUR', timeframe='1m', limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.rename(columns=str.capitalize, inplace=True)

    df = df.rename(columns={"Close": "Close"})
    features = compute_features(df)
    signals = generate_trade_signals(features)
    latest = signals.iloc[-1]

# Display signal
st.subheader("Latest Signal")
st.metric(label="Signal", value=latest['signal'])

# Indicator metrics
st.subheader("Latest Metrics")
cols = st.columns(3)
metrics = {
    "Price (EUR)": latest['Close'],
    "Gradient Ratio (15m/1d)": round(latest['grad_ratio_15m_1d'], 3),
    "Z-Score (1d)": round(latest['z_score_1d'], 3),
    "Cycle Score": round(latest['cycle_score'], 3),
    "Spike Ratio": round(latest['price_spike_ratio'], 3),
    "Drawdown Recovery": round(latest['drawdown_recovery'], 3),
    "Local Entropy": round(latest['local_entropy'], 3)
}
for i, (label, value) in enumerate(metrics.items()):
    cols[i % 3].metric(label, value)

# Price chart
st.subheader("Price Chart with Signals")
fig, ax = plt.subplots(figsize=(12, 5))
signals['Close'].plot(ax=ax, label='BTC/EUR Price')
for idx in signals.index[signals['signal'] == "BUY"]:
    ax.axvline(x=idx, color='green', linestyle='--', alpha=0.3)
for idx in signals.index[signals['signal'] == "SELL"]:
    ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
ax.set_ylabel("Price (EUR)")
ax.legend()
st.pyplot(fig)

st.markdown("Refresh the page to update the signals.")
