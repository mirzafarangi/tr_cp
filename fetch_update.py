import pandas as pd
import numpy as np
import ta
from pathlib import Path
import logging
from datetime import datetime
import json

def get_data_path():
    with open('config.json', 'r') as f:
        config = json.load(f)
        return config['data_path']

def enrich_features(input_file: str) -> pd.DataFrame:
    """
    Enrich trading data with additional technical indicators and scenarios
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def calculate_ichimoku(df, tenkan_period, kijun_period, senkou_span_b_period, suffix):
        """Calculate Ichimoku indicators with specified periods and suffix"""
        def get_period_levels(high, low, period):
            period_high = high.rolling(window=period).max()
            period_low = low.rolling(window=period).min()
            return (period_high + period_low) / 2
        
        tenkan = get_period_levels(df['high'], df['low'], tenkan_period)
        kijun = get_period_levels(df['high'], df['low'], kijun_period)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = get_period_levels(df['high'], df['low'], senkou_span_b_period)
        
        df[f'Tenkan-sen_{suffix}'] = tenkan
        df[f'Kijun-sen_{suffix}'] = kijun
        df[f'Senkou_Span_A_{suffix}'] = senkou_a
        df[f'Senkou_Span_B_{suffix}'] = senkou_b
        
        # Calculate cloud status and signals
        df[f'Cloud_Status_{suffix}'] = df.apply(
            lambda x: 'Price Above Cloud (Bullish)' if x['close'] > max(x[f'Senkou_Span_A_{suffix}'], x[f'Senkou_Span_B_{suffix}'])
            else 'Price Below Cloud (Bearish)' if x['close'] < min(x[f'Senkou_Span_A_{suffix}'], x[f'Senkou_Span_B_{suffix}'])
            else 'Price In Cloud (Neutral)', 
            axis=1
        )
        
        df[f'Tenkan_Kijun_Signal_{suffix}'] = np.where(
            df[f'Tenkan-sen_{suffix}'] > df[f'Kijun-sen_{suffix}'],
            'Bullish',
            'Bearish'
        )
        
        return df
    
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, parse_dates=['timestamp'])
        
        # Calculate Scalping Indicators
        logger.info("Calculating scalping indicators")
        df['EMA9_scalping'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['EMA21_scalping'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['RSI7_scalping'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        df['RSI9_scalping'] = ta.momentum.RSIIndicator(df['close'], window=9).rsi()
        df['VWAP_scalping'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
        ).volume_weighted_average_price()
        
        # Scalping Stochastic
        stoch_fast = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], 
            window=5, smooth_window=3
        )
        df['Stoch_Fast_K_scalping'] = stoch_fast.stoch()
        df['Stoch_Fast_D_scalping'] = stoch_fast.stoch_signal()
        
        # Calculate Swing Indicators
        logger.info("Calculating swing indicators")
        df['SMA20_swing'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['EMA50_swing'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['RSI14_swing'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['ADX_swing'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # Swing Stochastic
        stoch_slow = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], 
            window=14, smooth_window=3
        )
        df['Stoch_Slow_K_14_3_3_swing'] = stoch_slow.stoch()
        df['Stoch_Slow_D_14_3_3_swing'] = stoch_slow.stoch_signal()
        
        # Calculate Ichimoku for both scenarios
        logger.info("Calculating Ichimoku indicators")
        df = calculate_ichimoku(df, 6, 13, 9, 'scalping')
        df = calculate_ichimoku(df, 9, 26, 52, 'swing')
        
        # Price Levels
        logger.info("Calculating price levels")
        period = 20
        high_roll = df['high'].rolling(window=period).max()
        low_roll = df['low'].rolling(window=period).min()
        diff = high_roll - low_roll
        pivot = (high_roll + low_roll + df['close']) / 3
        
        df['Current_Support'] = pivot - diff
        df['Current_Resistance'] = pivot + diff
        df['Major_Resistance'] = high_roll
        df['Weak_Resistance'] = high_roll - (diff * 0.236)
        df['Strong_Support'] = low_roll + (diff * 0.236)
        df['Major_Support'] = low_roll
        
        # Calculate Entry, Stop Loss, and Take Profit levels
        logger.info("Calculating trading levels")
        risk_pct = 2.0  # Default 2%
        reward_pct = 6.0  # Default 6%
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Scalping levels
        df['Entry_Price_scalping'] = df['close'] + (0.5 * atr)
        df['Stop_Loss_scalping'] = df['Entry_Price_scalping'] * (1 - risk_pct/100)
        df['Take_Profit_scalping'] = df['Entry_Price_scalping'] * (1 + reward_pct/100)
        
        # Swing levels
        df['Entry_Price_swing'] = df['close']
        df['Stop_Loss_swing'] = df['Entry_Price_swing'] * (1 - risk_pct/100)
        df['Take_Profit_swing'] = df['Entry_Price_swing'] * (1 + reward_pct/100)
        
        # Save enriched dataset
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_updated.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Enriched data saved to {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = get_data_path()
    enriched_df = enrich_features(input_file)
    print("Data enrichment complete!")
    
    # Display column names for verification
    print("\nNew columns added:")
    print(enriched_df.columns.tolist())