import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="[Page Title]", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

import pandas as pd
import numpy as np
import ta
import json

def get_data_path():
    with open('config.json', 'r') as f:
        config = json.load(f)
        return config['data_path']

def safe_max(*args):
    """Safe maximum calculation handling None values"""
    valid_args = [x for x in args if x is not None]
    return max(valid_args) if valid_args else None

def safe_min(*args):
    """Safe minimum calculation handling None values"""
    valid_args = [x for x in args if x is not None]
    return min(valid_args) if valid_args else None

def load_data(file_path):
    """Load and prepare data from CSV file"""
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = calculate_additional_indicators(df)
    return df.iloc[-1]  # Return last row


def calculate_ichimoku(df, tenkan_period, kijun_period, senkou_span_b_period):
    """Calculate Ichimoku indicators with custom periods using the full DataFrame"""
    # Helper function to calculate period high/low
    def get_period_levels(high, low, period):
        period_high = high.rolling(window=period).max()
        period_low = low.rolling(window=period).min()
        return (period_high + period_low) / 2
    
    # Calculate Tenkan-sen (Conversion Line)
    tenkan_sen = get_period_levels(df['high'], df['low'], tenkan_period)
    
    # Calculate Kijun-sen (Base Line)
    kijun_sen = get_period_levels(df['high'], df['low'], kijun_period)
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Calculate Senkou Span B (Leading Span B)
    senkou_span_b = get_period_levels(df['high'], df['low'], senkou_span_b_period)
    
    # Get the last values, ensuring no NaN values
    ichimoku = {
        'tenkan_sen': float(tenkan_sen.fillna(method='ffill').iloc[-1]),
        'kijun_sen': float(kijun_sen.fillna(method='ffill').iloc[-1]),
        'senkou_span_a': float(senkou_span_a.fillna(method='ffill').iloc[-1]),
        'senkou_span_b': float(senkou_span_b.fillna(method='ffill').iloc[-1])
    }
    
    return ichimoku

def calculate_additional_indicators(df, scalping_rsi_period, stoch_settings):
    """Calculate all technical indicators with user-defined parameters"""
    # Calculate RSI variants based on user selection
    df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    df['rsi_9'] = ta.momentum.RSIIndicator(df['close'], window=9).rsi()
    df[f'rsi_{scalping_rsi_period}'] = ta.momentum.RSIIndicator(df['close'], 
                                                               window=scalping_rsi_period).rsi()
    
    # Calculate EMAs
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    
    # Calculate SMAs
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    
    # Calculate Stochastic based on user settings
    if stoch_settings == "Fast (5,3,3)":
        window, smooth = 5, 3
    else:  # Default (14,3,3)
        window, smooth = 14, 3
        
    stoch_fast = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'], 
        window=window, smooth_window=smooth
    )
    df['stoch_fast_k'] = stoch_fast.stoch()
    df['stoch_fast_d'] = stoch_fast.stoch_signal()
    
    stoch_slow = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'], 
        window=14, smooth_window=3
    )
    df['stoch_slow_k'] = stoch_slow.stoch()
    df['stoch_slow_d'] = stoch_slow.stoch_signal()
    
    # Calculate ADX
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    # Calculate VWAP
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
    ).volume_weighted_average_price()
    
    df = df.fillna(method='ffill')
    return df

# Modify these two functions to accept df as parameter:

def calculate_scalping_scenario(df, risk_pct, reward_pct, atr, scalping_rsi_period):
    """Calculate scalping trading scenario with user parameters"""
    risk = risk_pct / 100
    reward = reward_pct / 100
    
    # Get the current price from the last row
    data = df.iloc[-1]
    close_price = float(data['close'])
    
    # Calculate Ichimoku with full dataset
    scalping_ichimoku = calculate_ichimoku(df, 6, 13, 9)
    
    # Use offset for entry calculation
    entry_price = safe_max(
        scalping_ichimoku['tenkan_sen'],
        close_price + (0.5 * atr)
    )
    
    # Use risk percentage for stop loss
    stop_loss = safe_min(
        scalping_ichimoku['kijun_sen'],
        entry_price * (1 - risk)  # Apply risk percentage
    )
    
    # Use reward percentage for take profit
    take_profit = safe_max(
        scalping_ichimoku['senkou_span_a'],
        entry_price * (1 + reward)  # Apply reward percentage
    )
    
    scalping = {
        'Entry Signals': {
            'Price': close_price,
            'EMA9': float(data['ema_9']),
            'EMA21': float(data['ema_21']),
            f'RSI{scalping_rsi_period}': float(data[f'rsi_{scalping_rsi_period}']),
            'VWAP': float(data['vwap']),
            'Stoch_Fast_K': float(data['stoch_fast_k']),
            'Stoch_Fast_D': float(data['stoch_fast_d'])
        },
        'Entry Price': entry_price,
        'Stop Loss': stop_loss,
        'Take Profit': take_profit
    }
    return scalping

def calculate_swing_scenario(df, risk_pct, reward_pct, atr, swing_rsi_levels):
    """Calculate swing trading scenario with user parameters"""
    risk = risk_pct / 100
    reward = reward_pct / 100
    
    # Get the current price from the last row
    data = df.iloc[-1]
    close_price = float(data['close'])
    
    # Calculate Ichimoku with full dataset
    swing_ichimoku = calculate_ichimoku(df, 9, 26, 52)
    
    # Use RSI levels for entry conditions
    rsi_14 = float(data['rsi_14'])
    rsi_lower, rsi_upper = swing_rsi_levels
    
    # Adjust entry based on RSI levels
    entry_base = safe_max(
        swing_ichimoku['kijun_sen'],
        close_price
    )
    
    # Modify entry price based on RSI conditions
    if rsi_14 < rsi_lower:
        entry_price = entry_base * 0.99  # Discount for oversold
    elif rsi_14 > rsi_upper:
        entry_price = entry_base * 1.01  # Premium for overbought
    else:
        entry_price = entry_base
    
    # Apply risk and reward percentages
    stop_loss = safe_min(
        swing_ichimoku['senkou_span_b'],
        entry_price * (1 - risk)
    )
    
    take_profit = safe_max(
        swing_ichimoku['senkou_span_a'],
        entry_price * (1 + reward)
    )
    
    swing = {
        'Entry Signals': {
            'Price': close_price,
            'SMA20': float(data['sma_20']),
            'EMA50': float(data['ema_50']),
            'RSI14': rsi_14,
            'ADX': float(data['adx']),
            'Stoch_Slow_K': float(data['stoch_slow_k']),
            'Stoch_Slow_D': float(data['stoch_slow_d'])
        },
        'Entry Price': entry_price,
        'Stop Loss': stop_loss,
        'Take Profit': take_profit
    }
    return swing
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_page():
    if not st.session_state.initialized:
        # Do heavy initialization here
        st.session_state.initialized = True

def main():
    initialize_page()
    st.set_page_config(page_title="Enhanced Trading Analysis Dashboard", layout="wide")
    st.title("Enhanced Trading Analysis Dashboard")
    
    # Sidebar inputs
    st.sidebar.header("Parameters")
    
    # General parameters
    
    reward_pct = st.sidebar.number_input("Reward %", min_value=0.0, max_value=100.0, value=6.0, step=0.1)
    risk_pct = st.sidebar.number_input("Risk %", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    
    # RSI parameters
    st.sidebar.subheader("RSI Settings")
    scalping_rsi = st.sidebar.selectbox("Scalping RSI Period", [7, 9], index=0)
    swing_rsi_levels = st.sidebar.slider("Swing RSI Levels", 0, 100, (30, 70))
    
    # Stochastic parameters
    st.sidebar.subheader("Stochastic Settings")
    stoch_fast = st.sidebar.selectbox("Scalping Stochastic", ["Fast (5,3,3)", "Default (14,3,3)"], index=0)
    
    try:
        # Load and prepare data
        df = pd.read_csv(get_data_path(), parse_dates=['timestamp'])
        df = calculate_additional_indicators(df, scalping_rsi, stoch_fast)
        data = df.iloc[-1]
        
        # Calculate scenarios
        scalping_scenario = calculate_scalping_scenario(
            df, risk_pct, reward_pct, float(data['atr']), scalping_rsi
        )
        swing_scenario = calculate_swing_scenario(
            df, risk_pct, reward_pct, float(data['atr']), swing_rsi_levels
        )
        
        # 1. Market Status Display
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Current Market Status")
            st.metric("Current Price", f"{float(data['current_price']):.8f}")
            st.metric("24h Change", f"{float(data['24h_change']):.2f}%")
            st.metric("Period Change", f"{float(data['period_change']):.2f}%")
        
        with col2:
            st.subheader("Technical Indicators")
            st.metric("RSI (14)", f"{float(data['rsi_14']):.2f}")
            st.metric("MACD Signal", data['macd_signal'])
            st.metric("Ichimoku Status", data['ichimoku_status'])
        
        with col3:
            st.subheader("Overall Analysis")
            st.write("Current Signals:")
            signals = eval(data['current_signals'])
            for signal in signals:
                st.write(f"- {signal}")
            st.metric("Recommendation", data['overall_recommendation'])
        
        # 2. Trading Scenarios Analysis
        st.header("Trading Scenarios")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scalping Scenario (1m-15m)")
            with st.expander("Entry Signals", expanded=True):
                for indicator, value in scalping_scenario['Entry Signals'].items():
                    st.metric(indicator, f"{value:.8f}")
            with st.expander("Trade Levels", expanded=True):
                st.metric("Entry Price", f"{scalping_scenario['Entry Price']:.8f}")
                st.metric("Stop Loss", f"{scalping_scenario['Stop Loss']:.8f}")
                st.metric("Take Profit", f"{scalping_scenario['Take Profit']:.8f}")
                risk = abs(scalping_scenario['Entry Price'] - scalping_scenario['Stop Loss'])
                reward = abs(scalping_scenario['Take Profit'] - scalping_scenario['Entry Price'])
                st.metric("Risk/Reward Ratio", f"{(reward/risk):.2f}")
        
        with col2:
            st.subheader("Swing Trading Scenario (4H-1D)")
            with st.expander("Entry Signals", expanded=True):
                for indicator, value in swing_scenario['Entry Signals'].items():
                    st.metric(indicator, f"{value:.8f}")
            with st.expander("Trade Levels", expanded=True):
                st.metric("Entry Price", f"{swing_scenario['Entry Price']:.8f}")
                st.metric("Stop Loss", f"{swing_scenario['Stop Loss']:.8f}")
                st.metric("Take Profit", f"{swing_scenario['Take Profit']:.8f}")
                risk = abs(swing_scenario['Entry Price'] - swing_scenario['Stop Loss'])
                reward = abs(swing_scenario['Take Profit'] - swing_scenario['Entry Price'])
                st.metric("Risk/Reward Ratio", f"{(reward/risk):.2f}")
        
        # 3. Price Levels
        st.header("Price Levels")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Support Levels")
            st.metric("Major Support", f"{float(data['major_support']):.8f}")
            st.metric("Strong Support", f"{float(data['strong_support']):.8f}")
            st.metric("Current Support", f"{float(data['current_support']):.8f}")
        
        with col2:
            st.subheader("Resistance Levels")
            st.metric("Major Resistance", f"{float(data['major_resistance']):.8f}")
            st.metric("Weak Resistance", f"{float(data['weak_resistance']):.8f}")
            st.metric("Current Resistance", f"{float(data['current_resistance']):.8f}")
        
        # 4. Ichimoku Analysis
        st.header("Ichimoku Cloud Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scalping Ichimoku (6, 13, 9)")
            ichimoku_scalp = calculate_ichimoku(df, 6, 13, 9)
            with st.expander("Ichimoku Levels", expanded=True):
                st.metric("Tenkan-sen (Conversion)", f"{ichimoku_scalp['tenkan_sen']:.8f}")
                st.metric("Kijun-sen (Base)", f"{ichimoku_scalp['kijun_sen']:.8f}")
                st.metric("Senkou Span A", f"{ichimoku_scalp['senkou_span_a']:.8f}")
                st.metric("Senkou Span B", f"{ichimoku_scalp['senkou_span_b']:.8f}")
                
                cloud_top = max(ichimoku_scalp['senkou_span_a'], ichimoku_scalp['senkou_span_b'])
                cloud_bottom = min(ichimoku_scalp['senkou_span_a'], ichimoku_scalp['senkou_span_b'])
                cloud_status = "Price Above Cloud (Bullish)" if float(data['close']) > cloud_top else \
                             "Price Below Cloud (Bearish)" if float(data['close']) < cloud_bottom else \
                             "Price In Cloud (Neutral)"
                st.metric("Cloud Status", cloud_status)
                
                tenkan_kijun_signal = "Bullish" if ichimoku_scalp['tenkan_sen'] > ichimoku_scalp['kijun_sen'] else "Bearish"
                st.metric("Tenkan/Kijun Signal", tenkan_kijun_signal)
        
        with col2:
            st.subheader("Swing Ichimoku (9, 26, 52)")
            ichimoku_swing = calculate_ichimoku(df, 9, 26, 52)
            with st.expander("Ichimoku Levels", expanded=True):
                st.metric("Tenkan-sen (Conversion)", f"{ichimoku_swing['tenkan_sen']:.8f}")
                st.metric("Kijun-sen (Base)", f"{ichimoku_swing['kijun_sen']:.8f}")
                st.metric("Senkou Span A", f"{ichimoku_swing['senkou_span_a']:.8f}")
                st.metric("Senkou Span B", f"{ichimoku_swing['senkou_span_b']:.8f}")
                
                cloud_top = max(ichimoku_swing['senkou_span_a'], ichimoku_swing['senkou_span_b'])
                cloud_bottom = min(ichimoku_swing['senkou_span_a'], ichimoku_swing['senkou_span_b'])
                cloud_status = "Price Above Cloud (Bullish)" if float(data['close']) > cloud_top else \
                             "Price Below Cloud (Bearish)" if float(data['close']) < cloud_bottom else \
                             "Price In Cloud (Neutral)"
                st.metric("Cloud Status", cloud_status)
                
                tenkan_kijun_signal = "Bullish" if ichimoku_swing['tenkan_sen'] > ichimoku_swing['kijun_sen'] else "Bearish"
                st.metric("Tenkan/Kijun Signal", tenkan_kijun_signal)
                
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")

if __name__ == "__main__":
    main()