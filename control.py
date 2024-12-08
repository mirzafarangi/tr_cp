import streamlit as st
import subprocess
import os
import sys
from datetime import datetime
import json


# Constants
CONFIG_FILE = 'config.json'
DATA_FOLDER = 'data'
TRADING_CONFIG = 'trading_config.json'

# Add this new function
def configure_network():
    """Configure network settings for API access"""
    session = requests.Session()
    
    # Your ExpressVPN credentials from environment variables
    vpn_user = os.getenv('VPN_USER', '')  # Will be set in Streamlit Cloud
    vpn_pass = os.getenv('VPN_PASS', '')  # Will be set in Streamlit Cloud
    
    # ExpressVPN's German server
    vpn_host = "germany-frankfurt-1.express-vpn-proxy.com"  # Example server, use your actual ExpressVPN server
    vpn_port = "443"
    
    # Configure the proxy settings
    if vpn_user and vpn_pass:
        proxy_url = f"https://{vpn_user}:{vpn_pass}@{vpn_host}:{vpn_port}"
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        
        # Test connection
        try:
            test = session.get('https://api.binance.com/api/v3/ping')
            if test.status_code == 200:
                st.sidebar.success("VPN Connection Successful")
                return True
        except Exception as e:
            st.sidebar.error(f"VPN Connection Failed: {str(e)}")
            return False
    return False

def save_trading_params(symbol: str, interval: str):
    """Save trading parameters to config file"""
    config = {
        'symbol': symbol.upper(),
        'interval': interval
    }
    with open(TRADING_CONFIG, 'w') as f:
        json.dump(config, f)

def load_trading_params():
    """Load trading parameters from config file"""
    if os.path.exists(TRADING_CONFIG):
        with open(TRADING_CONFIG, 'r') as f:
            return json.load(f)
    return {'symbol': 'PEPEUSDT', 'interval': '4h'}  # Default values

def save_latest_file_path():
    """Save the path of the most recent CSV file to config"""
    if not os.path.exists(DATA_FOLDER):
        return None
    
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not files:
        return None
    
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)))
    latest_path = os.path.join(DATA_FOLDER, latest_file)
    
    config = {'data_path': latest_path}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
    
    return latest_path

def run_script(script_name: str, status_placeholder) -> None:
    """Execute a Python script and display its output"""
    try:
        status_placeholder.info(f"Running {script_name}...")
        process = subprocess.Popen([sys.executable, script_name], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                text=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                status_placeholder.write(output.strip())
        
        if process.poll() == 0:
            status_placeholder.success(f"{script_name} completed successfully!")
            if script_name in ["fetch.py", "fetch_update.py"]:
                latest_path = save_latest_file_path()
                if latest_path:
                    status_placeholder.info(f"Data path updated: {latest_path}")
        else:
            errors = process.stderr.read()
            status_placeholder.error(f"Error in {script_name}: {errors}")
            
    except Exception as e:
        status_placeholder.error(f"Failed to run {script_name}: {str(e)}")

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
    st.set_page_config(page_title="Trading Control Panel", layout="wide")
    
    # Add VPN configuration at the start
    configure_network()
    
    # Simple header
    st.title("Control Panel")
    
    # Trading Parameters Section
    st.sidebar.header("Currency-Interval")
    
    # Load current parameters
    current_params = load_trading_params()
    
    # Symbol input
    symbol = st.sidebar.text_input("Trading Pair", 
                                value=current_params['symbol'],
                                help="Enter trading pair (e.g., BTCUSDT, ETHUSDT)")
    
    # Interval selection
    intervals = ['1h', '4h', '1d', '7d']
    interval = st.sidebar.selectbox("Timeframe",
                                intervals,
                                index=intervals.index(current_params['interval']),
                                help="Select timeframe for analysis")
    
    # Save parameters button
    if st.sidebar.button("Save Parameters"):
        save_trading_params(symbol, interval)
        st.sidebar.success("Parameters saved!")
    
    # Data collection section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Collection")
        # Show data status
        st.code(check_data_status())
        
        # Data collection buttons
        if st.button("📥 Initial Data Fetch"):
            save_trading_params(symbol, interval)  # Save parameters before fetching
            status = st.empty()
            run_script("fetch.py", status)
            
        if st.button("🔄 Update Data"):
            save_trading_params(symbol, interval)  # Save parameters before updating
            status = st.empty()
            run_script("fetch_update.py", status)
        
        # Manual path update button
        if st.button("🔄 Update Path to Latest"):
            latest_path = save_latest_file_path()
            if latest_path:
                st.success(f"Path updated to: {latest_path}")
            else:
                st.error("No data files found")
    
    with col2:
        st.markdown("### Analysis")
        st.markdown("Choose an analysis to launch:")
        
        # Dashboard launch buttons
        if st.button("📊 Risk 1"):
            launch_dashboard("risk.py")
            st.info("Risk 1 launched in new window")
            
        if st.button("📉 Risk 2"):
            launch_dashboard("risk_minimal.py")
            st.info("Risk 2 launched in new window")
            
        if st.button("📈 Entry Models"):
            launch_dashboard("entry_models.py")
            st.info("Entry Models launched in new window")

if __name__ == "__main__":
    main()