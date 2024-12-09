import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from datetime import datetime

# Ensure Pandas displays small numbers correctly
pd.set_option('display.float_format', lambda x: '{:.12g}'.format(x))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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

def generate_gpt_analysis(last_candle, timeframe):
    """Generate analysis using OpenAI's GPT-4 model."""
    json_data = pd.DataFrame([last_candle]).to_json(orient='records')
    
    report_text = f"""
    This is the last {timeframe} candle data of PEPEUSDT: {json_data}
    Give me a clean, complete, comprehensive analysis and interpretation, and concrete examples for entry, take-profit, stop-loss points, 
    specially regarding these sections of my data:
    
    Current Market Conditions:
    - Current Price: {last_candle.get('current_price')}
    - EMA50: {last_candle.get('EMA50_swing')}
    - ATR: {last_candle.get('atr')}
    
    Technical Analysis:
    - RSI: {last_candle.get('rsi_14')}
    - ADX: {last_candle.get('ADX_swing')}
    - Stochastic: {last_candle.get('Stoch_Slow_K_14_3_3_swing')}
    
    Custom Indicators:
    - MCI: {last_candle.get('MCI')}
    - TMI: {last_candle.get('TMI')}
    - VAMO: {last_candle.get('VAMO')}
    - PEI: {last_candle.get('PEI')}
    
    Support/Resistance:
    - Pivot: {last_candle.get('Pivot')}
    - Support_1: {last_candle.get('Support_1')}
    - Resistance_1: {last_candle.get('Resistance_1')}
    - MT_SR_Weighted_Level: {last_candle.get('MT_SR_Weighted_Level')}

    The report should look like this example:
    Comprehensive Analysis and Strategy
    Let's break down and interpret the new indexes and their values with a focus on actionable strategies for:
    - Short-term (4h trading)
    - Mid-term (1-7 days)
    - Long-term (7-30 days)
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a professional trading assistant skilled in swing and scalping trading.
                Analyze the data and provide actionable insights with specific entry, stop-loss, and take-profit points."""},
                {"role": "user", "content": report_text}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def main():
    st.set_page_config(page_title="LLM Trading Insights", layout="wide")
    st.title("LLM Trading Insights")
    st.markdown("AI-Powered Technical Analysis and Trading Recommendations")
    
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
    
    # Display latest data in a single table
    latest_data = df.tail(1)
    st.subheader(f"Latest {timeframe.upper()} Data")
    st.dataframe(latest_data, use_container_width=True)
    
    # Generate analysis
    if st.button("Generate Trading Analysis"):
        with st.spinner("Generating comprehensive analysis..."):
            analysis = generate_gpt_analysis(latest_data.to_dict('records')[0], timeframe)
            st.subheader("Trading Analysis Report")
            st.markdown(analysis)

if __name__ == "__main__":
    main()