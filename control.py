import streamlit as st
import pandas as pd
import subprocess
import os
import sys
from datetime import datetime
import json
import requests
import time 

# Constants
CONFIG_FILE = 'config.json'
DATA_FOLDER = 'data'
TRADING_CONFIG = 'trading_config.json'

def configure_network():
    """Configure network settings for API access"""
    # Use a specific proxy from your Webshare dashboard
    proxy_username = "sqpwmtlu"
    proxy_password = "kl46de03cxib"
    # Replace these with an actual proxy from your dashboard
    proxy_host = "64.137.42.112"    # example - use one from your list
    proxy_port = "5157"              # use the port from your list
    
    proxy_url = f"http://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}"
    
    try:
        test = requests.get(
            'https://api.binance.com/api/v3/ping',
            proxies={
                'http': proxy_url,
                'https': proxy_url
            },
            timeout=15,
            verify=True
        )
        
        if test.status_code == 200:
            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            st.sidebar.success("Proxy Connected")
            return True
            
    except Exception as e:
        st.sidebar.error(f"Proxy Connection Failed: {str(e)}")
        
    return False

def load_initial_data():
    """Load initial configuration data into session state."""
    # Load config.json into session state
    if "config" not in st.session_state:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                st.session_state.config = json.load(f)
        else:
            st.session_state.config = {"data_path": ""}  # Default value

    # Load trading_config.json into session state
    if "trading_params" not in st.session_state:
        if os.path.exists(TRADING_CONFIG):
            with open(TRADING_CONFIG, "r") as f:
                st.session_state.trading_params = json.load(f)
        else:
            st.session_state.trading_params = {"symbol": "PEPEUSDT", "interval": "4h"}



# Replace the current load_trading_params function with:
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_trading_params():
    """Load trading parameters from config file"""
    try:
        if os.path.exists(TRADING_CONFIG):
            with open(TRADING_CONFIG, 'r') as f:
                params = json.load(f)
                if 'symbol' in params and 'interval' in params:
                    return params
        return {"symbol": "PEPEUSDT", "interval": "4h"}
    except Exception as e:
        st.sidebar.warning(f"Could not load trading parameters: {str(e)}")
        return {"symbol": "PEPEUSDT", "interval": "4h"}

# Add this new function for loading data:
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_latest_data() -> pd.DataFrame:
    """Load and cache the latest data"""
    try:
        config_path = ""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                config_path = config.get('data_path', "")
        
        if config_path and os.path.exists(config_path):
            return pd.read_csv(config_path, parse_dates=['timestamp'])
        else:
            st.error("No valid data path found")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def save_trading_params(symbol: str, interval: str):
    """Save trading parameters in session state."""
    st.session_state.trading_params = {
        'symbol': symbol.upper(),
        'interval': interval
    }

    # Optional: Write to file for debugging or local runs
    try:
        with open(TRADING_CONFIG, "w") as f:
            json.dump(st.session_state.trading_params, f, indent=4)
        st.sidebar.success("Trading parameters saved successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to save trading parameters: {str(e)}")



def save_latest_file_path():
    """Save the path of the most recent CSV file in session state."""
    if not os.path.exists(DATA_FOLDER):
        st.error("Data folder not found.")
        return None

    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not files:
        st.error("No CSV files found in the data folder.")
        return None

    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)))
    latest_path = os.path.join(DATA_FOLDER, latest_file)

    # Update session state
    st.session_state.config["data_path"] = latest_path

    # Optional: Write to file for debugging or local runs
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(st.session_state.config, f, indent=4)
        st.success(f"Data path updated: {latest_path}")
        return latest_path
    except Exception as e:
        st.error(f"Failed to update config.json: {str(e)}")
        return None




def run_script(script_name: str, status_placeholder) -> None:
    """Execute a Python script and display its output."""
    try:
        status_placeholder.info(f"Running {script_name}...")
        
        # Create environment with proxy settings
        env = os.environ.copy()
        if 'HTTP_PROXY' in os.environ:
            env['HTTP_PROXY'] = os.environ['HTTP_PROXY']
            env['HTTPS_PROXY'] = os.environ['HTTPS_PROXY']
        
        process = subprocess.Popen([sys.executable, script_name], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   env=env)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                status_placeholder.write(output.strip())
        
        if process.poll() == 0:
            status_placeholder.success(f"{script_name} completed successfully!")
            
            # Automatically update the latest file path in config.json
            latest_path = save_latest_file_path()
            if latest_path:
                status_placeholder.info(f"Updated data path in config.json: {latest_path}")
        else:
            errors = process.stderr.read()
            status_placeholder.error(f"Error in {script_name}: {errors}")
            
    except Exception as e:
        status_placeholder.error(f"Failed to run {script_name}: {str(e)}")


@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_data_status():
    """Check the status of data files and current path"""
    if not os.path.exists(DATA_FOLDER):
        return "No data folder found"
    
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not files:
        return "No data files found"
    
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)))
    mod_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(DATA_FOLDER, latest_file)))
    
    # Get current path from config
    current_path = "Not set"
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            current_path = config.get('data_path', "Not set")
    
    # Get trading parameters
    trading_params = load_trading_params()
    
    return (f"Trading Pair: {trading_params['symbol']}\n"
            f"Timeframe: {trading_params['interval']}\n"
            f"Latest data: {latest_file}\n"
            f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Current path: {current_path}")

def launch_dashboard(script_name: str):
    """Launch a Streamlit dashboard in a new process"""
    if not os.path.exists(CONFIG_FILE):
        st.error("No data path configured. Please fetch data first.")
        return
    
    subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', script_name])

def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Trading Control Panel", layout="wide")

    # Initialize session state for reloads
    if 'needs_rerun' not in st.session_state:
        st.session_state.needs_rerun = False

    # Load initial configuration data
    load_initial_data()

    # Configure the network (e.g., VPN or proxy) at the start
    configure_network()

    # Display the main header
    st.title("Trading Control Panel")

    # Sidebar: Trading Parameters Section
    st.sidebar.header("Currency-Interval")

    # Get cached trading parameters
    current_params = load_trading_params()  # Using cached version
    st.session_state.trading_params = current_params  # Update session state

    # Sidebar: Symbol Input
    symbol = st.sidebar.text_input(
        "Trading Pair",
        value=current_params["symbol"],
        help="Enter the trading pair (e.g., PEPEUSDT, BTCUSDT)"
    )

    # Sidebar: Interval Selection
    intervals = ["15m", "1h", "4h", "8h", "1w", "7d", "1M"]
    interval = st.sidebar.selectbox(
        "Timeframe",
        intervals,
        index=intervals.index(current_params["interval"]),
        help="Select the timeframe for analysis"
    )

    # Sidebar: Save Parameters Button
    if st.sidebar.button("Save Parameters"):
        save_trading_params(symbol, interval)
        st.session_state.needs_rerun = True
        st.experimental_rerun()

    # Main Layout: Two Columns
    col1, col2 = st.columns(2)

    # Column 1: Data Collection Section
    with col1:
        st.markdown("### Data Collection")

        # Display current data status using cached function
        st.code(check_data_status())

        # Buttons for Data Fetching and Path Updates
        if st.button("📥 Initial Data Fetch"):
            save_trading_params(symbol, interval)
            status = st.empty()
            run_script("fetch.py", status)
            st.session_state.needs_rerun = True
            st.experimental_rerun()

        if st.button("🔄 Update Data"):
            save_trading_params(symbol, interval)
            status = st.empty()
            run_script("fetch_update.py", status)
            st.session_state.needs_rerun = True
            st.experimental_rerun()

        if st.button("🔄 Update Path to Latest"):
            latest_path = save_latest_file_path()
            if latest_path:
                st.success(f"Data path updated to: {latest_path}")
                st.session_state.needs_rerun = True
                st.experimental_rerun()
            else:
                st.error("No data files found in the data folder.")

    

    # Handle rerun if needed
    if st.session_state.needs_rerun:
        st.session_state.needs_rerun = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()