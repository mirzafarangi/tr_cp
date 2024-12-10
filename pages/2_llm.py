import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from datetime import datetime

pd.set_option('display.float_format', lambda x: '{:.16g}'.format(x))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 16)

# Load OpenAI API Key
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API key not found. Please configure it in Streamlit Secrets.")
    st.stop()

# Constants
DATA_FOLDER = "data"
REQUIRED_COLUMNS = [
    'rsi_14', 'histogram', 'ADX_swing', 'Stoch_Slow_K_14_3_3_swing',
    'current_price', 'atr', 'EMA50_swing', 'high', 'low', 'close',
    'VWAP_scalping', 'EMA21_scalping', 'tenkan_sen', 'kijun_sen'
]

def get_last_k_rows_with_json(df, k=1):
    """Returns the last k rows of DataFrame as both table and JSON."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    last_k_rows = df.tail(k)
    clean_table = last_k_rows.reset_index(drop=True)
    json_output = clean_table.to_json(orient="records", date_format="iso")
    
    return clean_table, json_output

def extract_trading_indexes(row_dict):
    """Extract and format trading indexes from data."""
    last_4h_index = [
        ('MCI', row_dict['MCI']),
        ('TMI', row_dict['TMI']),
        ('VAMO', row_dict['VAMO']),
        ('Pivot', row_dict['Pivot']),
        ('Resistance_1', row_dict['Resistance_1']),
        ('Support_1', row_dict['Support_1']),
        ('Support_4h', row_dict['Support_4h']),
        ('Support_1d', row_dict['Support_1d']),
        ('Support_1w', row_dict['Support_1w']),
        ('MT_SR_Weighted_Level', row_dict['MT_SR_Weighted_Level']),
        ('PEI', row_dict['PEI']),
        ('VWDI', row_dict['VWDI'])
    ]

    other_index = [
        ('EMA50_swing', row_dict['EMA50_swing']),
        ('RSI14_swing', row_dict['rsi_14']),
        ('ADX_swing', row_dict['ADX_swing']),
        ('Stoch_Slow_K_14_3_3_swing', row_dict['Stoch_Slow_K_14_3_3_swing']),
        ('current_price', row_dict['current_price']),
        ('VWAP_scalping', row_dict['VWAP_scalping']),
        ('EMA21_scalping', row_dict['EMA21_scalping'])
    ]

    return last_4h_index, other_index

def load_latest_file():
    """Load the latest CSV file based on the last modified time."""
    if not os.path.exists(DATA_FOLDER):
        st.error(f"Data folder '{DATA_FOLDER}' not found.")
        return None
    
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    if not files:
        st.error("No CSV files found in the data folder.")
        return None
    
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)))
    return os.path.join(DATA_FOLDER, latest_file)

def read_csv_file(file_path):
    """Read a CSV file into a DataFrame with proper number formatting."""
    try:
        df = pd.read_csv(file_path, float_precision='high')
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file {file_path}: {str(e)}")
        return None

def calculate_trading_indexes(df):
    """Calculate all trading indexes from input data."""
    try:
        # 1. Momentum Composite Index (MCI)
        df['MCI'] = ((df['rsi_14'] - 50) / 50 + (df['Stoch_Slow_K_14_3_3_swing'] - 50) / 50 + df['histogram']) * df['ADX_swing']

        # 2. Trend-Momentum Index (TMI)
        df['TMI'] = (df['rsi_14'] / 100) * df['ADX_swing'] * ((df['current_price'] - df['EMA50_swing']) / df['EMA50_swing'])

        # 3. Volatility-Adjusted Momentum Oscillator (VAMO)
        df['VAMO'] = (df['rsi_14'] - 50) * (df['current_price'] / df['atr'])

        # 4. Dynamic Pivot Bands
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['Resistance_1'] = 2 * df['Pivot'] - df['low']
        df['Support_1'] = 2 * df['Pivot'] - df['high']

        # 5. Multi-Timeframe Support and Resistance (MT-SR)
        # Simulated data for 4h, 1d, and 1w support levels; replace with actual calculations if available
        df['Support_4h'] = df['low'] * 0.95  # Replace with actual support levels
        df['Support_1d'] = df['low'] * 0.92  # Replace with actual support levels
        df['Support_1w'] = df['low'] * 0.90  # Replace with actual support levels
        df['MT_SR_Weighted_Level'] = (df['Support_4h'] + 2 * df['Support_1d'] + 3 * df['Support_1w']) / 6

        # 6. Probabilistic Entry Indicator (PEI)
        df['PEI'] = (df['Stoch_Slow_K_14_3_3_swing'] + df['rsi_14'] + (df['tenkan_sen'] - df['kijun_sen'])) / 3

        # 7. Volume-Wave Divergence Index (VWDI)
        df['VWDI'] = df['VWAP_scalping'] - df['EMA21_scalping']

        # Clean any NaN values generated during calculations
        
        
        # Ensure small numbers are preserved
        for col in df.select_dtypes(include=[np.float64]).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error calculating indexes: {str(e)}")
        return None

def generate_gpt_analysis(data_dict, timeframe, last_4h_index, other_index):
    """Generate analysis using OpenAI's GPT-4 model with enhanced data."""
    table, json_data = get_last_k_rows_with_json(pd.DataFrame([data_dict]))
    
    report_text = f"""
    This is the last {timeframe} candle data of PEPEUSDT: {json_data}
    Give me a clean, complete, comprehensive analysis and interpretation, and concrete examples for entry, take-profit, stop-loss points, 
    specially regarding these sections of my data:
    
    Last 4H Trading Indexes:
    {', '.join([f"{k}: {v:.16g}" for k, v in last_4h_index])}
    
    Other Important Indexes:
    {', '.join([f"{k}: {v:.16g}" for k, v in other_index])}

    The report should looks like this example ---Comprehensive Analysis and Strategy
        Let’s break down and interpret the new indexes and their values with a focus on actionable strategies for short-term (4h trading), mid-term (1-7 days), and long-term (7-30 days) horizons. The interpretation includes identifying entry, take-profit, and stop-loss levels, along with momentum and trend signals.

        1. Interpreting the New Indexes

        Momentum Composite Index (MCI = 37.70):
        Interpretation: A value of 37.70 indicates strong upward momentum. MCI values above 20 suggest that the market is trending strongly upward.
        Actionable Insight:
        Short-term: Momentum supports buying dips. Watch for pullbacks near support levels for entries.
        Mid-term: Use the trend strength to ride the wave with higher targets.
        Trend-Momentum Index (TMI = 3.89):
        Interpretation: A positive TMI indicates an ongoing bullish trend with strong momentum. The closer the value is to 5 or above, the stronger the trend.
        Actionable Insight:
        Short-term: Trend momentum aligns with bullish bias; buying opportunities are favored on dips.
        Mid-term: Continue holding positions unless a reversal signal (e.g., RSI divergence) appears.
        Volatility-Adjusted Momentum Oscillator (VAMO = 406.66):
        Interpretation: A high VAMO suggests strong momentum adjusted for volatility. Values significantly above 100 indicate a robust bullish setup.
        Actionable Insight:
        Short-term: Favor quick entries during dips as volatility is working in favor of price appreciation.
        Mid-term: Plan for a larger stop-loss as volatility-driven momentum can extend the price swings.
        Pivot (0.00002668), Resistance_1 (0.00002754), Support_1 (0.00002621):
        Interpretation:
        The pivot level is the price equilibrium.
        Resistance_1 and Support_1 define the short-term trading range.
        Actionable Insight:
        Short-term: Enter near Support_1 (0.00002621) with a target near Resistance_1 (0.00002754).
        Use the pivot level (0.00002668) as a benchmark for intraday sentiment.
        Multi-Timeframe Support and Resistance (MT-SR Weighted Level = 0.00002362):
        Interpretation: This is an aggregated support level considering the broader timeframes. It suggests a strong support zone around 0.00002362.
        Actionable Insight:
        Mid-term: Accumulate long positions near MT-SR Weighted Level for stronger risk-reward setups.
        Long-term: Monitor the price's reaction at this level for potential breakdowns or sustained support.
        Probabilistic Entry Indicator (PEI = 56.84):
        Interpretation: A value >50 indicates a strong buy signal. The current value confirms bullish conditions.
        Actionable Insight:
        Short-term: Immediate buying opportunities with confirmation from momentum indicators.
        Mid-term: Continue holding positions or add on breakouts.
        Volume-Wave Divergence Index (VWDI = 7.75e-07):
        Interpretation: Positive VWDI indicates a bullish bias in volume trends. While the value is low, it aligns with the bullish outlook.
        Actionable Insight:
        Use VWDI to confirm volume-supported breakouts or trend continuations.
        2. Short-Term (4h Trading) Strategy

        Bias: Strong bullish momentum and trend supported by MCI, TMI, and PEI.
        Entry Point:
        Look for entries near Support_1 (0.00002621) or on pullbacks to Pivot (0.00002668).
        Confirm entry with short-term indicators like RSI7 and VWDI showing continued bullish momentum.
        Stop Loss:
        Place the stop loss just below Support_1 (0.00002621) at 0.00002600 to limit downside risk.
        Take Profit:
        Use Resistance_1 (0.00002754) as the primary target for short-term trades.
        Use ATR to dynamically adjust targets:
    
        Take Profit=Entry Price+(1.5×ATR)
        Example: If ATR = 1.46e-6, add approximately 0.00000219 to the entry price.
        3. Mid-Term (1-7 Days) Strategy

        Bias: Sustained bullish trend with room for higher highs.
        Entry Point:
        Enter near MT-SR Weighted Level (0.00002362) if a correction occurs.
        Accumulate near Support_4h (0.00002452) or Support_1d (0.00002374) for safer entries.
        Stop Loss:
        Place the stop loss slightly below MT-SR Weighted Level (0.00002362) to account for broader volatility. Example: 0.00002340.
        Take Profit:
        Use Fibonacci extensions for targets:
        First target: 1.618 × ATR above the entry price.
        Second target: Breakout above Resistance_1 (0.00002754) toward Major Resistance (0.00003169).
        4. Long-Term (7-30 Days) Strategy

        Bias: The market shows a strong long-term uptrend with momentum and volatility favoring further upside.
        Entry Point:
        Consider accumulation if prices retrace to MT-SR Weighted Level (0.00002362) or near Major Support (0.00002034).
        Stop Loss:
        Place the stop loss just below Major Support (0.00002034) to secure long-term positions.
        Take Profit:
        Use long-term resistance levels:
        Primary: 0.00003169 (Major Resistance).
        Secondary: 0.000035 (psychological round number).
        5. Key Metrics to Monitor

        Short-Term Signals:
        RSI14, VAMO, and VWDI for quick trend shifts.
        Mid-Term Signals:
        MCI and PEI for sustained momentum.
        Monitor Ichimoku cloud levels for breakdowns.
        Long-Term Signals:
        TMI and MT-SR for trend strength.
        Use VWDI to confirm volume-supported breakouts.
        Final Action Plan:
        Short-Term Entry Plan:
        Enter: 0.00002621 (Support_1) or 0.00002668 (Pivot).
        Stop Loss: 0.00002600.
        Take Profit: 0.00002754 (Resistance_1) or dynamically adjust with ATR.
        Mid-Term Hold Plan:
        Accumulate on dips near MT-SR Weighted Level (0.00002362).
        Stop Loss: 0.00002340.
        Targets: 0.00002754 (Resistance_1) and 0.00003169 (Major Resistance).
        Long-Term Hold Plan:
        Accumulate if price retraces to Major Support (0.00002034).
        Stop Loss: 0.00002000.
        Targets: 0.00003169 and 0.000035.
        This comprehensive strategy ensures you capitalize on immediate opportunities while aligning with broader market trends.---
                    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
    You are a professional trading assistant skilled in swing and scalping trading and analyzing technical analysis reports for cryptocurrencies.
    The user will provide data from their trading tools for specific coins and timeframes. Your role is to analyze this
    data, provide actionable insights, analyse price action and make recommendations on entry, stop-loss, and take-profit points.
    """},
                {"role": "user", "content": report_text}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def main():
    # Page configuration
    st.set_page_config(page_title="LLM Trading Insights", layout="wide")
    st.title("LLM Trading Insights")
    
    # Load and process data
    latest_file = load_latest_file()
    if not latest_file:
        return
    
    df = read_csv_file(latest_file)
    if df is None:
        return
    
    df = calculate_trading_indexes(df)
    if df is None:
        return
    
    # Display options in sidebar
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["4h", "1d", "7d", "30d"],
        index=0
    )
    
    # Get latest data with preserved small numbers
    latest_data = df.tail(1).to_dict('records')[0]
    last_4h_index, other_index = extract_trading_indexes(latest_data)
    
    # Display latest data with proper formatting
    st.subheader(f"Latest {timeframe.upper()} Data")
    
    # Create formatted DataFrame for display
    formatted_df = df.tail(1).copy()
    
    # Format all float columns to preserve small numbers
    for col in formatted_df.select_dtypes(include=['float64']).columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: '{:.16g}'.format(x))
    
    # Display the formatted DataFrame
    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Generate analysis button and display
    if st.button("Analysis Last Candle"):
        with st.spinner("generating report..."):
            analysis = generate_gpt_analysis(latest_data, timeframe, last_4h_index, other_index)
            st.subheader("Trading Analysis Report")
            st.markdown(analysis)

if __name__ == "__main__":
    main()