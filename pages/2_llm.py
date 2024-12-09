import streamlit as st
import pandas as pd
import os
import openai

# Ensure Pandas displays small numbers correctly
pd.options.display.float_format = '{:.12f}'.format

# Load OpenAI API Key from Streamlit Secrets
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

# Utility Functions
def load_latest_file():
    """Load the latest CSV file based on the last modified time."""
    if not os.path.exists(DATA_FOLDER):
        st.error("Data folder not found.")
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
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading file {file_path}: {str(e)}")
        return None

def ensure_columns_and_calculate_indexes(df):
    """Ensure necessary columns are present and calculate new indexes."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate custom trading indexes
    df['MCI'] = ((df['rsi_14'] - 50) / 50 + (df['Stoch_Slow_K_14_3_3_swing'] - 50) / 50 + df['histogram']) * df['ADX_swing']
    df['TMI'] = (df['rsi_14'] / 100) * df['ADX_swing'] * ((df['current_price'] - df['EMA50_swing']) / df['EMA50_swing'])
    df['VAMO'] = (df['rsi_14'] - 50) * (df['current_price'] / df['atr'])
    df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['Resistance_1'] = 2 * df['Pivot'] - df['low']
    df['Support_1'] = 2 * df['Pivot'] - df['high']
    df['Support_4h'] = df['low'] * 0.95
    df['Support_1d'] = df['low'] * 0.92
    df['Support_1w'] = df['low'] * 0.90
    df['MT_SR_Weighted_Level'] = (df['Support_4h'] + 2 * df['Support_1d'] + 3 * df['Support_1w']) / 6
    df['PEI'] = (df['Stoch_Slow_K_14_3_3_swing'] + df['rsi_14'] + (df['tenkan_sen'] - df['kijun_sen'])) / 3
    df['VWDI'] = df['VWAP_scalping'] - df['EMA21_scalping']
    df.fillna(0, inplace=True)
    return df

def save_updated_csv(df, file_name):
    """Save the updated DataFrame with new indexes."""
    output_file = os.path.join(DATA_FOLDER, file_name)
    df.to_csv(output_file, index=False)
    return output_file

def generate_gpt_analysis(report_text):
    """Generate an analysis using OpenAI's GPT model."""
    base_prompt = """
    You are a professional trading assistant skilled in swing and scalping trading and analyzing technical analysis reports for cryptocurrencies.
    Analyze the provided data, and give actionable insights, including entry, stop-loss, and take-profit recommendations.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": report_text}
            ],
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return f"Error generating GPT analysis: {str(e)}"

# Main Streamlit App
def main():
    st.set_page_config(page_title="LLM Trading Insights", layout="wide")
    st.title("LLM Trading Insights")
    st.markdown("Analyze trading data and generate actionable insights using GPT-powered AI.")

    # Load and preprocess data
    latest_file = load_latest_file()
    if not latest_file:
        return
    df = read_csv_file(latest_file)
    if df is None:
        return

    try:
        df = ensure_columns_and_calculate_indexes(df)
        updated_file = save_updated_csv(df, "PEPEUSDT_data_with_new_indexes.csv")
        st.sidebar.success(f"Updated data saved to: {os.path.basename(updated_file)}")
    except ValueError as e:
        st.error(f"Data processing error: {str(e)}")
        return

    # Select timeframe for analysis
    timeframe = st.sidebar.selectbox("Timeframe", ["4h", "1d", "7d", "30d"], index=0)

    # Extract last row for analysis
    try:
        last_row = df.tail(1)
        json_last_row = last_row.to_json(orient="records")
        st.subheader(f"Last {timeframe.upper()} Data")
        st.dataframe(last_row)
    except Exception as e:
        st.error(f"Error extracting last row for analysis: {str(e)}")
        return

    # Generate GPT Analysis
    if st.button("Generate GPT Analysis"):
        try:
            report_text = f"""
                This is the last {timeframe} candle data of PEPEUSDT: {json_last_row}.
                Provide a clean, comprehensive analysis, including entry, stop-loss, and take-profit points. Focus on key indexes like MCI, TMI, and VAMO.
            """
            analysis = generate_gpt_analysis(report_text)
            st.subheader("GPT Analysis Report")
            st.markdown(analysis)
        except Exception as e:
            st.error(f"Error generating GPT analysis: {str(e)}")

if __name__ == "__main__":
    main()
