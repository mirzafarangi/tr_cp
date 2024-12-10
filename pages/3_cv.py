import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import json
import requests


# --- Helper Functions ---
def fetch_live_data(symbol: str, interval: str, limit: int = 30) -> pd.DataFrame:
    """
    Fetch live candlestick data from Binance API for the given symbol and interval.
    
    Parameters:
    - symbol: Trading pair (e.g., 'BTCUSDT').
    - interval: Timeframe (e.g., '4h').
    - limit: Number of candles to fetch.
    
    Returns:
    - A DataFrame containing OHLCV data.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Process data into a DataFrame
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignored"
        ])
        
        # Convert data types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error


def add_indicators(df):
    """Add necessary technical indicators."""
    if df.empty:
        return df

    # Add RSI
    df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # Add Moving Averages
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    
    # Add Ichimoku
    ichimoku = calculate_ichimoku(df, 9, 26, 52)
    df['tenkan_sen'] = ichimoku['tenkan_sen']
    df['kijun_sen'] = ichimoku['kijun_sen']
    df['senkou_span_a'] = ichimoku['senkou_span_a']
    df['senkou_span_b'] = ichimoku['senkou_span_b']
    
    # Add Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'], window=14, smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    return df


def calculate_ichimoku(df, tenkan_period, kijun_period, span_b_period):
    """Calculate Ichimoku components."""
    def period_avg(high, low, period):
        return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2
    
    tenkan_sen = period_avg(df['high'], df['low'], tenkan_period)
    kijun_sen = period_avg(df['high'], df['low'], kijun_period)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = period_avg(df['high'], df['low'], span_b_period)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b
    }


def identify_patterns(df):
    """Identify common candlestick patterns."""
    patterns = []
    for i in range(1, len(df)):
        # Bullish Engulfing
        if (
            df['close'].iloc[i] > df['open'].iloc[i]
            and df['close'].iloc[i - 1] < df['open'].iloc[i - 1]
            and df['close'].iloc[i] > df['open'].iloc[i - 1]
            and df['open'].iloc[i] < df['close'].iloc[i - 1]
        ):
            patterns.append(('Bullish Engulfing', df['timestamp'].iloc[i]))
        
        # Add more patterns here as needed
        
    return patterns


def plot_candles_with_indicators(df):
    """Plot candlestick chart with indicators."""
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candles'
        )
    )

    # Add Moving Averages
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['ema_9'],
            mode='lines', name='EMA 9',
            line=dict(color='blue', width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['ema_21'],
            mode='lines', name='EMA 21',
            line=dict(color='orange', width=1)
        )
    )

    # Add Ichimoku components
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['senkou_span_a'],
            mode='lines', name='Senkou Span A',
            line=dict(color='green', width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['senkou_span_b'],
            mode='lines', name='Senkou Span B',
            line=dict(color='red', width=1)
        )
    )

    fig.update_layout(
        title='4H Candlestick Chart with Indicators',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    return fig


def plot_advanced_patterns(df, patterns):
    """Plot advanced patterns on the candlestick chart."""
    fig = plot_candles_with_indicators(df)
    
    for pattern, timestamp in patterns:
        fig.add_annotation(
            x=timestamp,
            y=df[df['timestamp'] == timestamp]['high'].values[0],
            text=pattern,
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30
        )
    
    fig.update_layout(
        title='Candlestick Chart with Patterns',
    )
    return fig


# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Live 4H Candle Analysis", layout="wide")
    st.title("Live 4H Candle Analysis Dashboard")

    # Sidebar inputs
    st.sidebar.header("Parameters")
    symbol = st.sidebar.text_input("Symbol", value="PEPEUSDT", help="Enter the trading pair (e.g., BTCUSDT)")
    interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "8h", "1w", "7d", "1M"], index=0)

    # Fetch live data
    df = fetch_live_data(symbol, interval)

    # Add indicators
    if not df.empty:
        df = add_indicators(df)

        # 1. Candle Visualization
        st.header("Candlestick Chart with Indicators")
        fig_candles = plot_candles_with_indicators(df)
        st.plotly_chart(fig_candles, use_container_width=True)

        # 2. Pattern Recognition
        st.header("Pattern Recognition")
        patterns = identify_patterns(df)
        if patterns:
            st.write("Identified Patterns:")
            for pattern, timestamp in patterns:
                st.write(f"- {pattern} at {timestamp}")
        else:
            st.write("No significant patterns identified.")

        fig_patterns = plot_advanced_patterns(df, patterns)
        st.plotly_chart(fig_patterns, use_container_width=True)

        # 3. Analysis and Recommendations
        st.header("Analysis and Recommendations")
        st.subheader("Traditional Technical Analysis")
        st.write(f"RSI (14): {df['rsi_14'].iloc[-1]:.2f}")
        st.write(f"Ichimoku Status: {'Bullish' if df['close'].iloc[-1] > df['senkou_span_a'].iloc[-1] else 'Bearish'}")

        st.subheader("Advanced Price Action")
        st.write("Monitor for patterns like bullish engulfing near key support zones.")
    
    else:
        st.error("No data available. Check your symbol or interval.")

if __name__ == "__main__":
    main()