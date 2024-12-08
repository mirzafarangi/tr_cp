import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import json
import os
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trading_params():
    with open('trading_config.json', 'r') as f:
        return json.load(f)
    
class SignalType(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class PatternType(Enum):
    DOJI = "DOJI"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    ENGULFING_BULLISH = "ENGULFING_BULLISH"
    ENGULFING_BEARISH = "ENGULFING_BEARISH"
    NONE = "NONE"

@dataclass
class TechnicalMetrics:
    # Price Metrics
    current_price: float
    period_high: float
    period_low: float
    
    # Change Metrics
    change_24h: float
    period_change: float
    
    # Volume Metrics
    average_volume: float
    
    # Momentum Indicators
    rsi_14: float
    rsi_30: float
    rsi_50: float
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_overall_signal: SignalType
    
    # Moving Averages
    ma_10: float
    ma_50: float
    ma_200: float
    price_ma10_diff: float
    price_ma50_diff: float
    price_ma200_diff: float
    
    # Support/Resistance Levels
    current_support: float
    current_resistance: float
    major_resistance: float
    weak_resistance: float
    first_target: float
    middle_point: float
    strong_support: float
    major_support: float
    
    # Trend Analysis
    adx: float
    ichimoku_status: str
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    
    # Risk Metrics
    atr: float
    suggested_stop_loss: float
    risk_reward_target: float
    
    # Patterns and Signals
    patterns: List[PatternType]
    current_signals: List[str]
    overall_recommendation: str

class BinanceDataFetcher:
    """Handles all Binance API data fetching operations"""
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    def __init__(self):
        self.session = requests.Session()
        if 'HTTP_PROXY' in os.environ:
            self.session.proxies = {
                'http': os.environ['HTTP_PROXY'],
                'https': os.environ['HTTPS_PROXY']
            }
            self.session.verify = True
            self.session.timeout = 30
    
    def fetch_historical_data(
    self,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: Optional[str] = None,
    limit: int = 1000
) -> pd.DataFrame:
        """
        Fetch complete historical data from Binance
        
        Parameters:
        - symbol: Trading pair (e.g., 'BTCUSDT')
        - interval: Timeframe (e.g., '4h', '1d')
        - start_date: Start date in 'YYYY-MM-DD' format
        - end_date: Optional end date in 'YYYY-MM-DD HH:MM' format
        - limit: Number of candles per request (max 1000)
        """
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d %H:%M").timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)
        
        all_klines = []
        current_start = start_ts
        
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": limit
            }
            
            try:
                response = self.session.get(
                    self.BASE_URL, 
                    params=params,
                    timeout=30,
                    verify=True
                )
                response.raise_for_status()
                
                klines = response.json()
                if not klines:
                    break
                    
                all_klines.extend(klines)
                current_start = klines[-1][0] + 1
                time.sleep(0.1)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error: {str(e)}")
                time.sleep(1)  # Retry delay
                continue
                
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}")
                break
        
        return self._process_klines_data(all_klines)
    
    def get_listing_date(self, symbol: str) -> str:
        """Get listing date for a symbol from Binance API"""
        try:
            info_url = f"https://api.binance.com/api/v3/exchangeInfo?symbol={symbol}"
            response = self.session.get(info_url)
            response.raise_for_status()
            symbol_info = response.json()
            
            for s in symbol_info['symbols']:
                if s['symbol'] == symbol and 'listingDate' in s:
                    return datetime.fromtimestamp(s['listingDate']).strftime('%Y-%m-%d')
                    
            logger.warning(f"No listing date found for {symbol}, using default")
            return '2024-01-01'  # Default for newer coins
            
        except Exception as e:
            logger.error(f"Error getting listing date: {str(e)}")
            return '2024-01-01'
    
    def _process_klines_data(self, klines: List) -> pd.DataFrame:
        """Convert raw klines data to DataFrame with proper types"""
        if not klines:
            raise ValueError("No data received from Binance API")
        
        # Create DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'quote_volume', 'trades', 'taker_buy_base', 
                         'taker_buy_quote']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.sort_values('timestamp')

def validate_parameters(df: pd.DataFrame) -> bool:
    """Validate input data meets minimum requirements"""
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Required: {required_columns}")
    
    if len(df) < 200:  # Minimum required for most indicators
        raise ValueError("Insufficient data points. Minimum 200 required.")
    
    return True

def save_data(df: pd.DataFrame, symbol: str, folder: str = "data") -> str:
    """Save data with timestamp and return filename"""
    import os
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{folder}/{symbol}_data_{timestamp}.csv"
    
    # Save data
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    return filename

class TechnicalAnalysis:
    """Class containing all technical analysis calculations"""
    
    def __init__(self, df: pd.DataFrame):
        if len(df) < 200:
            raise ValueError("Insufficient data for technical analysis (minimum 200 periods required)")
        """Initialize with price data"""
        self.validate_data(df)
        self.df = df.copy()
        self.calculate_all_indicators()
    
    def validate_data(self, df: pd.DataFrame):
        """Validate input data and handle small decimal values"""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
            
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if df[col].min() < 1e-8:
                df[col] = df[col].astype(np.float64)
        
    def calculate_all_indicators(self):
        """Calculate complete set of technical indicators"""
        try:
            # Basic Price and Volume
            self.df['current_price'] = self.df['close']
            self.df['24h_change'] = self.df['close'].pct_change(6).fillna(0) * 100  # 6 periods = 24h for 4h data
            self.df['period_change'] = self.df['close'].pct_change().fillna(0) * 100
            self.df['period_high'] = self.df['high'].rolling(window=6).max()
            self.df['period_low'] = self.df['low'].rolling(window=6).min()
            self.df['average_volume'] = self.df['volume'].rolling(window=6).mean()

            # RSI Variants
            for period in [14, 30, 50]:
                delta = self.df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
            exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
            self.df['macd'] = exp1 - exp2
            self.df['signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
            self.df['histogram'] = self.df['macd'] - self.df['signal']
            self.df['macd_signal'] = np.where(self.df['macd'] > self.df['signal'], 'BULLISH', 'BEARISH')

            # Moving Averages
            for period in [10, 50, 200]:
                self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()
                self.df[f'price_ma{period}_diff'] = ((self.df['close'] - self.df[f'ma_{period}']) / 
                                                    self.df[f'ma_{period}'] * 100)

            # Support/Resistance and Fibonacci
            period = 20
            high = self.df['high'].rolling(window=period).max()
            low = self.df['low'].rolling(window=period).min()
            diff = high - low
            pivot = (high + low + self.df['close']) / 3

            self.df['current_support'] = pivot - diff
            self.df['current_resistance'] = pivot + diff
            self.df['major_resistance'] = high
            self.df['weak_resistance'] = high - (diff * 0.236)
            self.df['first_target'] = high - (diff * 0.382)
            self.df['middle_point'] = high - (diff * 0.5)
            self.df['strong_support'] = low + (diff * 0.236)
            self.df['major_support'] = low

            # Ichimoku Cloud
            high_9 = self.df['high'].rolling(window=9).max()
            low_9 = self.df['low'].rolling(window=9).min()
            self.df['tenkan_sen'] = (high_9 + low_9) / 2

            high_26 = self.df['high'].rolling(window=26).max()
            low_26 = self.df['low'].rolling(window=26).min()
            self.df['kijun_sen'] = (high_26 + low_26) / 2

            self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen']) / 2).shift(26)
            self.df['senkou_span_b'] = ((self.df['high'].rolling(window=52).max() + 
                                        self.df['low'].rolling(window=52).min()) / 2).shift(26)

            self.df['ichimoku_status'] = np.where(
                (self.df['close'] > self.df['senkou_span_a']) & 
                (self.df['close'] > self.df['senkou_span_b']),
                'BULLISH',
                np.where(
                    (self.df['close'] < self.df['senkou_span_a']) & 
                    (self.df['close'] < self.df['senkou_span_b']),
                    'BEARISH',
                    'NEUTRAL'
                )
            )

            # Volatility Metrics
            tr = pd.concat([
                self.df['high'] - self.df['low'],
                abs(self.df['high'] - self.df['close'].shift()),
                abs(self.df['low'] - self.df['close'].shift())
            ], axis=1).max(axis=1)
            
            self.df['atr'] = tr.rolling(window=14).mean()
            self.df['suggested_stop_loss'] = self.df['close'] - (2 * self.df['atr'])
            self.df['risk_reward_target'] = self.df['close'] + (4 * self.df['atr'])

            # Candlestick Patterns
            self.df['patterns'] = self.df.apply(self._identify_patterns, axis=1)

            # Generate Signals and Recommendations
            self.df['current_signals'] = self.df.apply(self._generate_signals, axis=1)
            self.df['overall_recommendation'] = self.df.apply(self._generate_recommendation, axis=1)

            return self.df

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
            
    

    def _identify_patterns(self, row) -> List[str]:
        """Identify candlestick patterns from OHLC data"""
        try:
            patterns = []
            body = abs(row['open'] - row['close'])
            wick_up = row['high'] - max(row['open'], row['close']) 
            wick_down = min(row['open'], row['close']) - row['low']
            total_range = row['high'] - row['low']

            if total_range == 0:  # Avoid division by zero
                return [PatternType.NONE.value]

            # Doji
            if body <= 0.1 * total_range:
                patterns.append(PatternType.DOJI.value)
            
            # Shooting Star 
            if (wick_up > 2 * body and wick_down < 0.2 * total_range):
                patterns.append(PatternType.SHOOTING_STAR.value)
            
            # Hammer
            if (wick_down > 2 * body and wick_up < 0.2 * total_range):
                patterns.append(PatternType.HAMMER.value)
            
            # Engulfing patterns
            prev_close = row.get('close_prev')  # Need to pass previous close
            if prev_close is not None:
                if row['open'] < row['close'] and row['open'] > prev_close:
                    patterns.append(PatternType.ENGULFING_BULLISH.value)
                elif row['open'] > row['close'] and row['open'] < prev_close:
                    patterns.append(PatternType.ENGULFING_BEARISH.value)
                    
            return patterns if patterns else [PatternType.NONE.value]

        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            return [PatternType.NONE.value]
    def _generate_signals(self, row) -> List[str]:
        """Generate comprehensive trading signals from all technical indicators"""
        signals = []
        
        # Price trends
        if all(row['close'] > row[f'ma_{period}'] for period in [10, 50, 200]):
            signals.append('STRONG_UPTREND')
        elif row['close'] < row['ma_200']:
            signals.append('LONG_TERM_DOWNTREND')
            
        # RSI conditions
        for period in [14, 30, 50]:
            if row[f'rsi_{period}'] > 70:
                signals.append(f'RSI{period}_OVERBOUGHT')
            elif row[f'rsi_{period}'] < 30:
                signals.append(f'RSI{period}_OVERSOLD')
                
        # MACD analysis
        if row['macd'] > row['signal']:
            # Get hist_diff by comparing current and previous rows
            if isinstance(row['histogram'], float):
                signals.append('MACD_BULLISH')
            else:
                if row['histogram'] > row['histogram'].shift(1):
                    signals.append('MACD_BULLISH_INCREASING')
                else:
                    signals.append('MACD_BULLISH_WEAKENING')
        else:
            if isinstance(row['histogram'], float):
                signals.append('MACD_BEARISH')
            else:
                if row['histogram'] < row['histogram'].shift(1):
                    signals.append('MACD_BEARISH_INCREASING')
                else:
                    signals.append('MACD_BEARISH_WEAKENING')
                
        # Support/Resistance
        if abs(row['close'] - row['current_support']) < row['atr']:
            signals.append('AT_SUPPORT')
        elif abs(row['close'] - row['current_resistance']) < row['atr']:
            signals.append('AT_RESISTANCE')
            
        # Ichimoku signals
        if row['close'] > row['senkou_span_a'] and row['close'] > row['senkou_span_b']:
            if row['close'] > row['tenkan_sen'] > row['kijun_sen']:
                signals.append('STRONG_ICHIMOKU_BULLISH')
            else:
                signals.append('ICHIMOKU_BULLISH')
        elif row['close'] < row['senkou_span_a'] and row['close'] < row['senkou_span_b']:
            if row['close'] < row['tenkan_sen'] < row['kijun_sen']:
                signals.append('STRONG_ICHIMOKU_BEARISH')
            else:
                signals.append('ICHIMOKU_BEARISH')
                
        # Pattern-based signals
        if PatternType.HAMMER.value in row['patterns'] and row['close'] < row['ma_50']:
            signals.append('POTENTIAL_REVERSAL_BULLISH')
        elif PatternType.SHOOTING_STAR.value in row['patterns'] and row['close'] > row['ma_50']:
            signals.append('POTENTIAL_REVERSAL_BEARISH')
        elif PatternType.ENGULFING_BULLISH.value in row['patterns']:
            signals.append('BULLISH_ENGULFING')
        elif PatternType.ENGULFING_BEARISH.value in row['patterns']:
            signals.append('BEARISH_ENGULFING')
            
        # Volume signals
        try:
            volume_ma = self.df['volume'].rolling(20).mean().iloc[row.name]
            if row['volume'] > volume_ma * 1.5:
                signals.append('HIGH_VOLUME')
        except:
            pass

        # Volatility
        try:
            atr_ma = self.df['atr'].rolling(20).mean().iloc[row.name]
            if row['atr'] > atr_ma * 1.2:
                signals.append('HIGH_VOLATILITY')
        except:
            pass
        
        return signals if signals else ['NO_CLEAR_SIGNALS']
    def _generate_recommendation(self, row) -> str:
        """Generate trading recommendation based on technical signals"""
        bullish = 0
        bearish = 0

        # Price action
        bullish += 1 if row['close'] > row['ma_50'] else -1
        bullish += 1 if row['close'] > row['ma_200'] else -1

        # Momentum 
        bullish += 1 if row['macd'] > row['signal'] else -1
        bullish += 1 if 30 < row['rsi_14'] < 50 else (-1 if row['rsi_14'] > 70 else 0)

        # Support/Resistance
        bullish += 1 if row['close'] > row['current_support'] else -1
        bullish += 1 if row['close'] < row['current_resistance'] else -1

        # Ichimoku
        bullish += 1 if row['ichimoku_status'] == 'BULLISH' else (-1 if row['ichimoku_status'] == 'BEARISH' else 0)

        if bullish >= 3:
            return SignalType.BULLISH.value
        elif bullish > 0:
            return 'BUY'
        elif bullish <= -3:
            return 'STRONG_SELL'  
        elif bullish < 0:
            return 'SELL'
        return SignalType.NEUTRAL.value
# Usage example:
def analyze_crypto_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to analyze cryptocurrency data"""
    analyzer = TechnicalAnalysis(df)
    return analyzer.df

def main():
    try:
        # Initialize fetcher
        fetcher = BinanceDataFetcher()
        params = load_trading_params()
        symbol = params['symbol']
        interval = params['interval']
        
        # Get data range
        start_date = fetcher.get_listing_date(symbol)
        end_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Initial validation
        if start_date >= end_date:
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")
        
        # Fetch data
        print(f"Fetching {symbol} data from {start_date} to {end_date}...")
        df = fetcher.fetch_historical_data(symbol, interval, start_date, end_date)
        
        # Validate data size
        if len(df) < 200:
            raise ValueError("Insufficient data points (minimum 200 required)")
        
        # Calculate indicators
        print("Calculating technical indicators...")
        analyzed_df = analyze_crypto_data(df)
        
        # Save results
        print("Saving results...")
        filename = save_data(analyzed_df, symbol)
        
        print(f"Analysis complete! Data saved to {filename}")
        print(f"Data range: {analyzed_df['timestamp'].min()} to {analyzed_df['timestamp'].max()}")
        
        return analyzed_df
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
if __name__ == "__main__":
    analyzed_df = main()