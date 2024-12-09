import streamlit as st
import pandas as pd
import os
import openai
from datetime import datetime

# Ensure Pandas displays small numbers correctly
pd.set_option('display.float_format', lambda x: '{:.12g}'.format(x))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load OpenAI API Key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
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
    """Read a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        # Validate required columns
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
        df['MCI'] = ((df['rsi_14'] - 50) / 50 + 
                     (df['Stoch_Slow_K_14_3_3_swing'] - 50) / 50 + 
                     df['histogram']) * df['ADX_swing']
        
        # 2. Trend-Momentum Index (TMI)
        df['TMI'] = (df['rsi_14'] / 100) * df['ADX_swing'] * \
                    ((df['current_price'] - df['EMA50_swing']) / df['EMA50_swing'])
        
        # 3. Volatility-Adjusted Momentum Oscillator (VAMO)
        df['VAMO'] = (df['rsi_14'] - 50) * (df['current_price'] / df['atr'])
        
        # 4. Dynamic Pivot Bands
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['Resistance_1'] = 2 * df['Pivot'] - df['low']
        df['Support_1'] = 2 * df['Pivot'] - df['high']
        
        # 5. Multi-Timeframe Support and Resistance (MT-SR)
        df['Support_4h'] = df['low'] * 0.95
        df['Support_1d'] = df['low'] * 0.92
        df['Support_1w'] = df['low'] * 0.90
        df['MT_SR_Weighted_Level'] = (df['Support_4h'] + 2 * df['Support_1d'] + 
                                     3 * df['Support_1w']) / 6
        
        # 6. Probabilistic Entry Indicator (PEI)
        df['PEI'] = (df['Stoch_Slow_K_14_3_3_swing'] + df['rsi_14'] + 
                     (df['tenkan_sen'] - df['kijun_sen'])) / 3
        
        # 7. Volume-Wave Divergence Index (VWDI)
        df['VWDI'] = df['VWAP_scalping'] - df['EMA21_scalping']
        
        # Clean any NaN values
        df.fillna(0, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error calculating indexes: {str(e)}")
        return None

def save_processed_data(df):
    """Save the processed DataFrame to a CSV file."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(DATA_FOLDER, f"processed_data_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        return output_file
    except Exception as e:
        st.error(f"Error saving processed data: {str(e)}")
        return None

def generate_gpt_analysis(data_dict, timeframe):
    """Generate analysis using OpenAI's GPT model with your specific prompt format."""
    report_text = f"""
    This is the last {timeframe} candle data of PEPEUSDT: {data_dict}
    Give me a clean, complete, comprehensive analysis and interpretation, and concrete examples for entry, take-profit, stop-loss points, 
    specially regarding these sections of my data:
    
    1. Momentum and Trend Indicators:
    - MCI: {data_dict.get('MCI')}
    - TMI: {data_dict.get('TMI')}
    - VAMO: {data_dict.get('VAMO')}
    
    2. Support and Resistance Levels:
    - Current Price: {data_dict.get('current_price')}
    - Pivot: {data_dict.get('Pivot')}
    - Support_1: {data_dict.get('Support_1')}
    - Resistance_1: {data_dict.get('Resistance_1')}
    - MT_SR_Weighted_Level: {data_dict.get('MT_SR_Weighted_Level')}
    
    The report should look like this example:
    
    Comprehensive Analysis and Strategy
    Let's break down and interpret the new indexes and their values with a focus on actionable strategies for:
    - Short-term (4h trading)
    - Mid-term (1-7 days)
    - Long-term (7-30 days)
    
    Please provide:
    1. Detailed interpretation of each indicator
    2. Multiple timeframe analysis
    3. Specific entry, stop-loss, and take-profit levels
    4. Risk management recommendations
    5. Key levels to monitor

    Example of a your generated report should be like ------Comprehensive Analysis and Strategy
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
T
a
k
e
 
P
r
o
f
i
t
=
E
n
t
r
y
 
P
r
i
c
e
+
(
1.5
×
A
T
R
)
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
        response = openai.ChatCompletion.create(
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
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def main():
    st.set_page_config(page_title="LLM Trading Insights", layout="wide")
    st.title("LLM Trading Insights")
    st.markdown("LLM Trading Insights")
    
    # Load and process data
    latest_file = load_latest_file()
    if not latest_file:
        return
        
    df = read_csv_file(latest_file)
    if df is None:
        return
        
    # Calculate trading indexes
    df = calculate_trading_indexes(df)
    if df is None:
        return
    
    # Save processed data
    output_file = save_processed_data(df)
    if output_file:
        st.sidebar.success(f"Processed data saved to: {os.path.basename(output_file)}")
    
    # Display options
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["4h", "1d", "7d", "30d"],
        index=0
    )
    
    # Get and display latest data
    latest_data = df.tail(1).to_dict('records')[0]
    
    # Display data in organized sections
    st.subheader(f"Latest {timeframe.upper()} Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Price Levels")
        st.dataframe(pd.DataFrame({
            'Metric': ['Current Price', 'Pivot', 'Support_1', 'Resistance_1'],
            'Value': [
                latest_data['current_price'],
                latest_data['Pivot'],
                latest_data['Support_1'],
                latest_data['Resistance_1']
            ]
        }).set_index('Metric'))
    
    with col2:
        st.markdown("### Momentum Indicators")
        st.dataframe(pd.DataFrame({
            'Metric': ['MCI', 'TMI', 'VAMO', 'PEI'],
            'Value': [
                latest_data['MCI'],
                latest_data['TMI'],
                latest_data['VAMO'],
                latest_data['PEI']
            ]
        }).set_index('Metric'))
    
    with col3:
        st.markdown("### Technical Indicators")
        st.dataframe(pd.DataFrame({
            'Metric': ['RSI', 'ADX', 'VWDI'],
            'Value': [
                latest_data['rsi_14'],
                latest_data['ADX_swing'],
                latest_data['VWDI']
            ]
        }).set_index('Metric'))
    
    # Generate analysis
    if st.button("Generate Trading Analysis"):
        with st.spinner("Generating comprehensive analysis..."):
            analysis = generate_gpt_analysis(latest_data, timeframe)
            st.subheader("Trading Analysis Report")
            st.markdown(analysis)

if __name__ == "__main__":
    main()