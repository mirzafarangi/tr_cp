import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import openai

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
DATA_FOLDER = "data"
CONFIG_FILE = "config.json"

# Helper Functions
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
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error reading file {file_path}: {str(e)}")
        return None


def get_last_k_rows_with_json(df, k):
    """
    Returns the last k rows of a DataFrame as both a clean table and a JSON object.

    Parameters:
        df (pd.DataFrame): The DataFrame containing coin data.
        k (int): The number of last rows to retrieve.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The last k rows in table format.
            - str: The JSON representation of the last k rows.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    last_k_rows = df.tail(k)
    clean_table = last_k_rows.reset_index(drop=True)
    json_output = clean_table.to_json(orient="records", date_format="iso")
    return clean_table, json_output


def generate_gpt_analysis(report_text):
    """
    Generate an analysis using OpenAI's GPT model.

    Parameters:
        report_text (str): The prompt or data to send to GPT.

    Returns:
        str: The generated GPT response.
    """
    base_prompt = """
    You are a professional trading assistant skilled in analyzing technical analysis reports for cryptocurrencies.
    The user will provide data from their trading tools for specific coins and timeframes. Your role is to analyze this
    data, provide actionable insights, and make recommendations on entry, stop-loss, and take-profit points.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": report_text}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating GPT analysis: {str(e)}")
        return None


# Main Function
def main():
    st.set_page_config(page_title="Machine Learning Trading Insights", layout="wide")

    # Header
    st.title("Machine Learning Trading Insights")
    st.markdown("Analyze your trading data and generate actionable insights using advanced AI models.")

    # Load the latest data
    st.sidebar.header("Data Configuration")
    latest_file = load_latest_file()
    if latest_file:
        st.sidebar.success(f"Latest file: {os.path.basename(latest_file)}")
        df = read_csv_file(latest_file)
    else:
        st.sidebar.error("No valid data file found.")
        return

    # Select timeframe for analysis
    st.sidebar.header("Select Analysis Parameters")
    timeframe = st.sidebar.selectbox("Timeframe", ["4h", "1d", "7d", "30d"], index=0)

    # Process and display the last 4H data
    if df is not None:
        last_4h, json_last_4h = get_last_k_rows_with_json(df, 1)
        st.subheader(f"Last {timeframe.upper()} Data")
        st.write(last_4h)

        # Extract specific sections for detailed analysis
        last_4h_index = last_4h[
            [
                "MCI", "TMI", "VAMO", "Pivot", "Resistance_1", "Support_1",
                "Support_4h", "Support_1d", "Support_1w", "MT_SR_Weighted_Level", "PEI", "VWDI"
            ]
        ].to_dict(orient="records")[0]

        other_index = last_4h[
            [
                "current_signals", "overall_recommendation", "EMA9_scalping", "EMA21_scalping", "RSI7_scalping",
                "RSI9_scalping", "VWAP_scalping", "Stoch_Fast_K_scalping", "Stoch_Fast_D_scalping", "SMA20_swing",
                "EMA50_swing", "RSI14_swing", "ADX_swing", "Stoch_Slow_K_14_3_3_swing", "Stoch_Slow_D_14_3_3_swing",
                "Tenkan-sen_scalping", "Kijun-sen_scalping", "Senkou_Span_A_scalping", "Senkou_Span_B_scalping",
                "Cloud_Status_scalping", "Tenkan_Kijun_Signal_scalping", "Tenkan-sen_swing", "Kijun-sen_swing",
                "Senkou_Span_A_swing", "Senkou_Span_B_swing", "Cloud_Status_swing", "Tenkan_Kijun_Signal_swing",
                "Current_Support", "Current_Resistance", "Major_Resistance", "Weak_Resistance", "Strong_Support",
                "Major_Support"
            ]
        ].to_dict(orient="records")[0]

        # Generate GPT analysis
        if st.button("Generate GPT Analysis"):
            report_text = f"""
            This is the last {timeframe} candle data of PEPEUSDT: {json_last_4h}.
            Give me a clean, complete, comprehensive analysis and interpretation, and concrete examples for entry, take-profit, stop-loss points, 
            specially regarding these sections of my data: {last_4h_index}, {other_index}.
            """
            analysis = generate_gpt_analysis(report_text)
            if analysis:
                st.subheader("AI-Powered Trading Insights")
                st.markdown(analysis)
            else:
                st.error("Failed to generate analysis.")

    else:
        st.error("Failed to load data.")


if __name__ == "__main__":
    main()
