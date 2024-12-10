import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
import json
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    # Basic Patterns
    ENGULFING_BULLISH = "Bullish Engulfing"
    ENGULFING_BEARISH = "Bearish Engulfing"
    PINBAR_BULLISH = "Bullish Pin Bar"
    PINBAR_BEARISH = "Bearish Pin Bar"
    INSIDE_BAR = "Inside Bar"
    OUTSIDE_BAR = "Outside Bar"
    
    # Advanced Reversal Patterns
    LIQUIDITY_SWEEP_HIGH = "Liquidity Sweep High"
    LIQUIDITY_SWEEP_LOW = "Liquidity Sweep Low"
    FAIR_VALUE_GAP = "Fair Value Gap"
    ORDER_BLOCK = "Order Block"
    BREAKER_BLOCK = "Breaker Block"
    
    # Volume Patterns
    INSTITUTIONAL_ACCUMULATION = "Institutional Accumulation"
    INSTITUTIONAL_DISTRIBUTION = "Institutional Distribution"
    VOLUME_PRICE_DIVERGENCE = "Volume Price Divergence"
    
    # Market Structure Patterns
    SWING_FAILURE = "Swing Failure Pattern"
    MARKET_STRUCTURE_BREAK = "Break of Market Structure"
    CHANGE_OF_CHARACTER = "Change of Character"
    
    # Volatility Patterns
    VOLATILITY_CONTRACTION = "Volatility Contraction"
    VOLATILITY_EXPANSION = "Volatility Expansion"
    RANGE_BREAKOUT = "Range Breakout"

@dataclass
class Pattern:
    type: PatternType
    start_idx: int
    end_idx: int
    confidence: float
    description: str
    timeframe: str
    risk_reward: float
    volume_confirmation: bool
    multi_timeframe_confluence: bool

class PatternAnalyzer:
    def __init__(self, df: pd.DataFrame, timeframe: str, lookback: int = 30):
        """
        Initialize pattern analyzer.
        df: DataFrame with OHLCV data.
        timeframe: Time interval.
        lookback: Number of candles to analyze (default 30).
        """
        buffer = 50  # Extra candles for calculations.
        total_needed = lookback + buffer
        self.df = df.tail(total_needed).copy()  # Select the most recent rows for analysis.
        self.analysis_df = self.df.tail(lookback).copy()  # Analyze the most recent `lookback` candles.

        # Validate `analysis_df` size
        if self.analysis_df.empty:
            logger.error("Analysis DataFrame is empty after slicing. Check lookback size.")
            raise ValueError("Analysis DataFrame is empty. Adjust the lookback parameter.")

        self.timeframe = timeframe
        self.lookback = lookback
        self.patterns = []
        self.preprocess_data()

    def preprocess_data(self):
        """Calculate necessary indicators and metrics."""
        # Calculate ATR without TA-Lib
        def calculate_atr(high, low, close, period=14):
            tr = pd.DataFrame()
            tr['h-l'] = high - low
            tr['h-pc'] = abs(high - close.shift())
            tr['l-pc'] = abs(low - close.shift())
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            return tr['tr'].rolling(period).mean()

        # Volatility metrics
        self.df['atr'] = calculate_atr(self.df['high'], self.df['low'], self.df['close'])
        self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()

        # Price action metrics
        self.df['body_size'] = abs(self.df['close'] - self.df['open'])
        self.df['upper_wick'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_wick'] = self.df[['open', 'close']].min(axis=1) - self.df['low']

        # Volume analysis
        self.df['volume_delta'] = np.where(
            self.df['close'] >= self.df['open'],
            self.df['volume'],
            -self.df['volume']
        )
        self.df['cvd'] = self.df['volume_delta'].cumsum()

        # Fill NaN values
        self.df = self.df.fillna(method='ffill').fillna(0)

        
    def find_all_patterns(self) -> List[Pattern]:
        """Detect all pattern types"""
        patterns = []
        logger.info("Starting pattern detection...")
        
        try:
            # Basic patterns
            logger.info("Finding basic patterns...")
            patterns.extend(self.find_engulfing_patterns())
            patterns.extend(self.find_pinbar_patterns())
            patterns.extend(self.find_inside_outside_bars())
            
            # Advanced patterns
            logger.info("Finding advanced patterns...")
            patterns.extend(self.find_liquidity_sweeps())
            patterns.extend(self.find_order_blocks())
            patterns.extend(self.find_fair_value_gaps())
            
            # Volume patterns
            logger.info("Finding volume patterns...")
            patterns.extend(self.find_institutional_accumulation())
            patterns.extend(self.find_volume_divergence())
            
            # Market structure patterns
            logger.info("Finding market structure patterns...")
            patterns.extend(self.find_market_structure_breaks())
            
            # Volatility patterns
            logger.info("Finding volatility patterns...")
            patterns.extend(self.find_volatility_patterns())
            
            # Sort patterns by end index
            self.patterns = sorted(patterns, key=lambda x: x.end_idx)
            logger.info(f"Found total of {len(patterns)} patterns")
        
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            raise
        
        return self.patterns

    
    def find_engulfing_patterns(self) -> List[Pattern]:
        """Detect engulfing patterns with volume confirmation"""
        patterns = []
        
        for i in range(1, len(self.df) - 1):
            current = self.df.iloc[i]
            previous = self.df.iloc[i - 1]
            
            # Bullish engulfing
            if (current['close'] > current['open'] and
                previous['close'] < previous['open'] and
                current['open'] < previous['close'] and
                current['close'] > previous['open'] and
                current['volume'] > current['volume_sma']):
                
                patterns.append(Pattern(
                    type=PatternType.ENGULFING_BULLISH,
                    start_idx=max(i - 1, 0),  # Prevent out-of-bounds
                    end_idx=min(i, len(self.df) - 1),  # Prevent out-of-bounds
                    confidence=0.8,
                    description="Bullish engulfing with volume confirmation",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Bearish engulfing
            elif (current['close'] < current['open'] and
                previous['close'] > previous['open'] and
                current['open'] > previous['close'] and
                current['close'] < previous['open'] and
                current['volume'] > current['volume_sma']):
                
                patterns.append(Pattern(
                    type=PatternType.ENGULFING_BEARISH,
                    start_idx=max(i - 1, 0),
                    end_idx=min(i, len(self.df) - 1),
                    confidence=0.8,
                    description="Bearish engulfing with volume confirmation",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
        
        return patterns

    
    def find_pinbar_patterns(self) -> List[Pattern]:
        """Detect pin bars with emphasis on wick ratios"""
        patterns = []
        
        for i in range(1, len(self.df) - 1):
            current = self.df.iloc[i]
            
            body_size = current['body_size']
            upper_wick = current['upper_wick']
            lower_wick = current['lower_wick']
            
            # Bullish pin bar
            if (lower_wick > 2 * body_size and
                lower_wick > 2 * upper_wick and
                current['volume'] > current['volume_sma']):
                
                patterns.append(Pattern(
                    type=PatternType.PINBAR_BULLISH,
                    start_idx=max(i - 1, 0),
                    end_idx=min(i, len(self.df) - 1),
                    confidence=0.75,
                    description="Bullish pin bar with strong rejection",
                    timeframe=self.timeframe,
                    risk_reward=2.5,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Bearish pin bar
            elif (upper_wick > 2 * body_size and
                upper_wick > 2 * lower_wick and
                current['volume'] > current['volume_sma']):
                
                patterns.append(Pattern(
                    type=PatternType.PINBAR_BEARISH,
                    start_idx=max(i - 1, 0),
                    end_idx=min(i, len(self.df) - 1),
                    confidence=0.75,
                    description="Bearish pin bar with strong rejection",
                    timeframe=self.timeframe,
                    risk_reward=2.5,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
        
        return patterns

    def find_inside_outside_bars(self) -> List[Pattern]:
        """Detect inside and outside bars with volume analysis"""
        patterns = []
        
        for i in range(1, len(self.df)-1):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            
            # Inside bar
            if (current['high'] < previous['high'] and
                current['low'] > previous['low']):
                
                # Higher confidence if volume is lower than average
                confidence = 0.85 if current['volume'] < current['volume_sma'] else 0.7
                
                patterns.append(Pattern(
                    type=PatternType.INSIDE_BAR,
                    start_idx=i-1,
                    end_idx=i,
                    confidence=confidence,
                    description="Inside bar consolidation pattern",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=current['volume'] < current['volume_sma'],
                    multi_timeframe_confluence=False
                ))
            
            # Outside bar
            elif (current['high'] > previous['high'] and
                  current['low'] < previous['low'] and
                  current['volume'] > current['volume_sma'] * 1.2):  # Strong volume confirmation
                
                # Determine if bullish or bearish outside bar
                pattern_type = (PatternType.OUTSIDE_BAR)
                desc = ("Bullish outside bar" if current['close'] > current['open']
                       else "Bearish outside bar")
                
                patterns.append(Pattern(
                    type=pattern_type,
                    start_idx=i-1,
                    end_idx=i,
                    confidence=0.8,
                    description=desc,
                    timeframe=self.timeframe,
                    risk_reward=2.5,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
        
        return patterns
    
    def find_liquidity_sweeps(self) -> List[Pattern]:
        """Detect liquidity sweeps and stop hunts"""
        patterns = []
        window = 10
        
        for i in range(window, len(self.df)-1):
            current = self.df.iloc[i]
            window_data = self.df.iloc[i-window:i]
            
            # High sweep
            if (current['high'] > window_data['high'].max() and
                current['close'] < window_data['close'].mean() and
                current['volume'] > current['volume_sma'] * 1.5):
                
                patterns.append(Pattern(
                    type=PatternType.LIQUIDITY_SWEEP_HIGH,
                    start_idx=i-1,
                    end_idx=i,
                    confidence=0.85,
                    description="Failed sweep of highs with volume spike",
                    timeframe=self.timeframe,
                    risk_reward=3.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
                
            # Low sweep
            elif (current['low'] < window_data['low'].min() and
                  current['close'] > window_data['close'].mean() and
                  current['volume'] > current['volume_sma'] * 1.5):
                
                patterns.append(Pattern(
                    type=PatternType.LIQUIDITY_SWEEP_LOW,
                    start_idx=i-1,
                    end_idx=i,
                    confidence=0.85,
                    description="Failed sweep of lows with volume spike",
                    timeframe=self.timeframe,
                    risk_reward=3.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
                
        return patterns
    def find_order_blocks(self) -> List[Pattern]:
        """Detect institutional order blocks and breaker blocks"""
        patterns = []
        window = 10
        
        for i in range(window, len(self.df)-2):
            current = self.df.iloc[i]
            next_candle = self.df.iloc[i+1]
            window_data = self.df.iloc[i-window:i]
            
            # Bullish Order Block
            if (current['close'] < current['open'] and           # Strong bearish candle
                next_candle['close'] > next_candle['open'] and  # Followed by bullish candle
                current['volume'] > window_data['volume'].mean() * 1.5 and
                current['body_size'] > current['atr']):
                
                patterns.append(Pattern(
                    type=PatternType.ORDER_BLOCK,
                    start_idx=i,
                    end_idx=i+1,
                    confidence=0.85,
                    description="Bullish Order Block - Institutional buying zone",
                    timeframe=self.timeframe,
                    risk_reward=3.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Bearish Order Block
            elif (current['close'] > current['open'] and         # Strong bullish candle
                  next_candle['close'] < next_candle['open'] and  # Followed by bearish candle
                  current['volume'] > window_data['volume'].mean() * 1.5 and
                  current['body_size'] > current['atr']):
                
                patterns.append(Pattern(
                    type=PatternType.ORDER_BLOCK,
                    start_idx=i,
                    end_idx=i+1,
                    confidence=0.85,
                    description="Bearish Order Block - Institutional selling zone",
                    timeframe=self.timeframe,
                    risk_reward=3.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Breaker Block
            if i > window + 5:
                previous_high = window_data['high'].max()
                previous_low = window_data['low'].min()
                
                if (current['close'] > previous_high and
                    next_candle['low'] < current['low'] and
                    current['volume'] > window_data['volume'].mean() * 2):
                    
                    patterns.append(Pattern(
                        type=PatternType.BREAKER_BLOCK,
                        start_idx=i,
                        end_idx=i+1,
                        confidence=0.9,
                        description="Breaker Block - Strong reversal zone",
                        timeframe=self.timeframe,
                        risk_reward=2.5,
                        volume_confirmation=True,
                        multi_timeframe_confluence=False
                    ))
                
        return patterns

    def find_fair_value_gaps(self) -> List[Pattern]:
        """Detect fair value gaps in price action"""
        patterns = []
        
        for i in range(2, len(self.df)-1):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            two_back = self.df.iloc[i-2]
            
            # Bullish FVG
            if (two_back['low'] > current['high'] and  # Gap up
                previous['body_size'] > previous['atr'] * 0.5 and
                current['volume'] > current['volume_sma']):
                
                patterns.append(Pattern(
                    type=PatternType.FAIR_VALUE_GAP,
                    start_idx=i-2,
                    end_idx=i,
                    confidence=0.8,
                    description="Bullish Fair Value Gap - Potential support",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Bearish FVG
            elif (two_back['high'] < current['low'] and  # Gap down
                  previous['body_size'] > previous['atr'] * 0.5 and
                  current['volume'] > current['volume_sma']):
                
                patterns.append(Pattern(
                    type=PatternType.FAIR_VALUE_GAP,
                    start_idx=i-2,
                    end_idx=i,
                    confidence=0.8,
                    description="Bearish Fair Value Gap - Potential resistance",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
                
        return patterns
    def find_institutional_accumulation(self) -> List[Pattern]:
        """Detect institutional accumulation and distribution patterns"""
        patterns = []
        window = 20  # Analysis window
        
        for i in range(window, len(self.df)-1):
            current = self.df.iloc[i]
            window_data = self.df.iloc[i-window:i]
            
            # Calculate volume and price metrics
            avg_volume = window_data['volume'].mean()
            price_trend = (current['close'] - window_data['close'].mean()) / window_data['close'].mean()
            volume_trend = (current['volume'] - avg_volume) / avg_volume
            cvd_change = self.df['cvd'].iloc[i] - self.df['cvd'].iloc[i-window]
            
            # Accumulation Pattern
            if (current['volume'] > avg_volume * 1.5 and
                abs(current['close'] - current['open']) < current['atr'] * 0.5 and  # Small body
                cvd_change > 0 and  # Positive cumulative delta
                current['close'] > window_data['close'].mean()):  # Overall uptrend
                
                patterns.append(Pattern(
                    type=PatternType.INSTITUTIONAL_ACCUMULATION,
                    start_idx=i-window,
                    end_idx=i,
                    confidence=0.85,
                    description="Institutional Accumulation - High volume with price stability",
                    timeframe=self.timeframe,
                    risk_reward=2.5,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
                
            # Distribution Pattern
            elif (current['volume'] > avg_volume * 1.5 and
                  abs(current['close'] - current['open']) < current['atr'] * 0.5 and
                  cvd_change < 0 and  # Negative cumulative delta
                  current['close'] < window_data['close'].mean()):  # Overall downtrend
                
                patterns.append(Pattern(
                    type=PatternType.INSTITUTIONAL_DISTRIBUTION,
                    start_idx=i-window,
                    end_idx=i,
                    confidence=0.85,
                    description="Institutional Distribution - High volume selling pressure",
                    timeframe=self.timeframe,
                    risk_reward=2.5,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
                
        return patterns
    def find_volume_divergence(self) -> List[Pattern]:
        """Detect volume-price divergence patterns"""
        patterns = []
        window = 10  # Analysis window
        
        for i in range(window, len(self.df)-1):
            current = self.df.iloc[i]
            window_data = self.df.iloc[i-window:i+1]
            
            # Calculate trends
            price_high_trend = window_data['high'].diff().mean()
            price_low_trend = window_data['low'].diff().mean()
            volume_trend = window_data['volume'].diff().mean()
            
            # Bullish Divergence
            # Price making lower lows but volume decreasing (accumulation)
            if (price_low_trend < 0 and  # Price trending down
                volume_trend < 0 and     # Volume trending down
                current['close'] > current['open'] and  # Current candle bullish
                current['volume'] > current['volume_sma']):  # Volume confirmation
                
                patterns.append(Pattern(
                    type=PatternType.VOLUME_PRICE_DIVERGENCE,
                    start_idx=i-window,
                    end_idx=i,
                    confidence=0.8,
                    description="Bullish Volume Divergence - Declining volume in downtrend",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Bearish Divergence
            # Price making higher highs but volume decreasing (distribution)
            elif (price_high_trend > 0 and  # Price trending up
                  volume_trend < 0 and      # Volume trending down
                  current['close'] < current['open'] and  # Current candle bearish
                  current['volume'] > current['volume_sma']):  # Volume confirmation
                
                patterns.append(Pattern(
                    type=PatternType.VOLUME_PRICE_DIVERGENCE,
                    start_idx=i-window,
                    end_idx=i,
                    confidence=0.8,
                    description="Bearish Volume Divergence - Declining volume in uptrend",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Additional check for climax volume
            if (current['volume'] > window_data['volume'].max() * 1.5 and
                abs(current['close'] - current['open']) > current['atr']):
                
                divergence_type = ("Bullish" if current['close'] > current['open'] 
                                 else "Bearish")
                
                patterns.append(Pattern(
                    type=PatternType.VOLUME_PRICE_DIVERGENCE,
                    start_idx=i-1,
                    end_idx=i,
                    confidence=0.9,
                    description=f"{divergence_type} Climax Volume - Potential reversal",
                    timeframe=self.timeframe,
                    risk_reward=2.5,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
                
        return patterns
    
    def find_market_structure_breaks(self) -> List[Pattern]:
        """Detect breaks in market structure and character changes"""
        patterns = []
        window = 20  # Analysis window
        
        for i in range(window, len(self.df)-1):
            current = self.df.iloc[i]
            window_data = self.df.iloc[i-window:i]
            
            # Calculate swing points
            swing_highs = []
            swing_lows = []
            
            for j in range(2, len(window_data)-2):
                if (window_data.iloc[j]['high'] > window_data.iloc[j-1]['high'] and 
                    window_data.iloc[j]['high'] > window_data.iloc[j-2]['high'] and
                    window_data.iloc[j]['high'] > window_data.iloc[j+1]['high'] and
                    window_data.iloc[j]['high'] > window_data.iloc[j+2]['high']):
                    swing_highs.append(window_data.iloc[j]['high'])
                
                if (window_data.iloc[j]['low'] < window_data.iloc[j-1]['low'] and 
                    window_data.iloc[j]['low'] < window_data.iloc[j-2]['low'] and
                    window_data.iloc[j]['low'] < window_data.iloc[j+1]['low'] and
                    window_data.iloc[j]['low'] < window_data.iloc[j+2]['low']):
                    swing_lows.append(window_data.iloc[j]['low'])
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Bullish Structure Break
                if (current['high'] > max(swing_highs[-2:]) and
                    current['volume'] > current['volume_sma'] * 1.5):
                    
                    patterns.append(Pattern(
                        type=PatternType.MARKET_STRUCTURE_BREAK,
                        start_idx=i-3,
                        end_idx=i,
                        confidence=0.85,
                        description="Bullish Market Structure Break - Higher High confirmed",
                        timeframe=self.timeframe,
                        risk_reward=2.5,
                        volume_confirmation=True,
                        multi_timeframe_confluence=False
                    ))
                
                # Bearish Structure Break
                elif (current['low'] < min(swing_lows[-2:]) and
                      current['volume'] > current['volume_sma'] * 1.5):
                    
                    patterns.append(Pattern(
                        type=PatternType.MARKET_STRUCTURE_BREAK,
                        start_idx=i-3,
                        end_idx=i,
                        confidence=0.85,
                        description="Bearish Market Structure Break - Lower Low confirmed",
                        timeframe=self.timeframe,
                        risk_reward=2.5,
                        volume_confirmation=True,
                        multi_timeframe_confluence=False
                    ))
            
            # Change of Character (CHoCH)
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                trend_change = False
                
                # Bullish Change
                if (swing_lows[-1] > swing_lows[-2] and
                    swing_highs[-1] > swing_highs[-2] and
                    current['close'] > current['open']):
                    trend_change = True
                    desc = "Bullish Change of Character"
                
                # Bearish Change
                elif (swing_highs[-1] < swing_highs[-2] and
                      swing_lows[-1] < swing_lows[-2] and
                      current['close'] < current['open']):
                    trend_change = True
                    desc = "Bearish Change of Character"
                
                if trend_change:
                    patterns.append(Pattern(
                        type=PatternType.CHANGE_OF_CHARACTER,
                        start_idx=i-5,
                        end_idx=i,
                        confidence=0.9,
                        description=desc,
                        timeframe=self.timeframe,
                        risk_reward=3.0,
                        volume_confirmation=True,
                        multi_timeframe_confluence=False
                    ))
        
        return patterns
    
    def find_volatility_patterns(self) -> List[Pattern]:
        """Detect volatility-based patterns including contractions and expansions"""
        patterns = []
        window = 20  # Analysis window
        
        for i in range(window, len(self.df)-1):
            current = self.df.iloc[i]
            window_data = self.df.iloc[i-window:i]
            
            # Calculate volatility metrics
            atr_series = window_data['atr']
            current_atr = current['atr']
            avg_atr = atr_series.mean()
            atr_std = atr_series.std()
            
            # Volatility Contraction (VCP)
            if (current_atr < avg_atr - (atr_std * 1.5) and  # Significant contraction
                current['body_size'] < current['atr'] * 0.5 and  # Small bodies
                window_data['body_size'].mean() < window_data['atr'].mean() * 0.7):  # Sustained contraction
                
                patterns.append(Pattern(
                    type=PatternType.VOLATILITY_CONTRACTION,
                    start_idx=i-5,
                    end_idx=i,
                    confidence=0.8,
                    description="Volatility Contraction Pattern - Potential breakout setup",
                    timeframe=self.timeframe,
                    risk_reward=3.0,
                    volume_confirmation=current['volume'] < current['volume_sma'],
                    multi_timeframe_confluence=False
                ))
            
            # Volatility Expansion
            elif (current_atr > avg_atr + (atr_std * 2) and  # Significant expansion
                  current['volume'] > current['volume_sma'] * 1.5):  # Volume confirmation
                
                patterns.append(Pattern(
                    type=PatternType.VOLATILITY_EXPANSION,
                    start_idx=i-1,
                    end_idx=i,
                    confidence=0.85,
                    description="Volatility Expansion - Strong directional move",
                    timeframe=self.timeframe,
                    risk_reward=2.0,
                    volume_confirmation=True,
                    multi_timeframe_confluence=False
                ))
            
            # Range Breakout
            if i > 5:  # Need some prior data for range
                recent_data = self.df.iloc[i-5:i]
                price_range = recent_data['high'].max() - recent_data['low'].min()
                range_avg = price_range / 5
                
                if (abs(current['close'] - current['open']) > range_avg * 2 and  # Strong breakout
                    current['volume'] > current['volume_sma'] * 1.3):  # Volume confirmation
                    
                    breakout_type = "Bullish" if current['close'] > current['open'] else "Bearish"
                    
                    patterns.append(Pattern(
                        type=PatternType.RANGE_BREAKOUT,
                        start_idx=i-5,
                        end_idx=i,
                        confidence=0.85,
                        description=f"{breakout_type} Range Breakout - Strong momentum move",
                        timeframe=self.timeframe,
                        risk_reward=2.5,
                        volume_confirmation=True,
                        multi_timeframe_confluence=False
                    ))
        
        return patterns
    
    def create_pattern_visualization(self, patterns: List[Pattern]) -> go.Figure:
        """Create visualization with pattern annotations."""
        display_df = self.analysis_df.copy()
        global_to_local_index = {idx: i for i, idx in enumerate(display_df.index)}

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=display_df['timestamp'],
            open=display_df['open'],
            high=display_df['high'],
            low=display_df['low'],
            close=display_df['close'],
            name="Price"
        ), row=1, col=1)

        # Add volume bars
        colors = ['red' if close < open else 'green'
                for close, open in zip(display_df['close'], display_df['open'])]
        fig.add_trace(go.Bar(
            x=display_df['timestamp'],
            y=display_df['volume'],
            name="Volume",
            marker_color=colors
        ), row=2, col=1)

        # Add pattern annotations and shapes
        for pattern in patterns:
            logger.info(f"Pattern indices: start_idx={pattern.start_idx}, end_idx={pattern.end_idx}")
            
            if pattern.start_idx not in global_to_local_index or pattern.end_idx not in global_to_local_index:
                logger.warning(f"Pattern indices out of range for display_df. Skipping pattern.")
                continue
            
            start_idx_local = global_to_local_index[pattern.start_idx]
            end_idx_local = global_to_local_index[pattern.end_idx]
            pattern_color = 'green' if 'BULLISH' in pattern.type.value else 'red'

            # Add rectangle highlighting the pattern
            fig.add_shape(
                type="rect",
                x0=display_df['timestamp'].iloc[start_idx_local],
                x1=display_df['timestamp'].iloc[end_idx_local],
                y0=display_df['low'].iloc[start_idx_local:end_idx_local + 1].min(),
                y1=display_df['high'].iloc[start_idx_local:end_idx_local + 1].max(),
                line=dict(color=pattern_color, width=1),
                fillcolor=pattern_color,
                opacity=0.2,
                row=1, col=1
            )

            # Add annotation for the pattern
            fig.add_annotation(
                x=display_df['timestamp'].iloc[end_idx_local],
                y=display_df['high'].iloc[end_idx_local],
                text=pattern.type.value,
                showarrow=True,
                arrowhead=1,
                ax=0, ay=-20,
                row=1, col=1
            )

        # Layout adjustments
        fig.update_layout(
            title=f"Pattern Analysis ({self.timeframe})",
            height=800,
            showlegend=True
        )

        return fig




class PatternDashboard:
    def __init__(self):
        self.data_path = self._get_data_path()
        
    def _get_data_path(self) -> str:
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config['data_path']
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, parse_dates=['timestamp'])
    
    def run_dashboard(self):
        st.title("Advanced Pattern Recognition")

        try:
            # Step 1: Load data
            with st.spinner('Loading data...'):
                logger.info("Starting to load data...")
                df = self.load_data()
                if df.empty:
                    st.error("The data is empty. Please check your data source.")
                    return
                logger.info(f"Loaded data with shape: {df.shape}")

            # Step 2: Timeframe and candle count selection
            timeframe = st.selectbox(
                "Select Timeframe for Analysis",
                ["15m", "1h", "4h", "8h", "1w", "1M"]
            )
            logger.info(f"Selected timeframe: {timeframe}")

            lookback = st.slider(
                "Number of candles to analyze",
                min_value=10,
                max_value=100,
                value=30,
                help="Select how many recent candles to analyze"
            )
            logger.info(f"User selected {lookback} candles for analysis.")

            # Step 3: Analyze patterns
            with st.spinner('Analyzing patterns...'):
                logger.info("Initializing PatternAnalyzer...")
                analyzer = PatternAnalyzer(df, timeframe, lookback)

                logger.info("Finding patterns...")
                patterns = analyzer.find_all_patterns()
                logger.info(f"Found {len(patterns)} patterns.")

            # Step 4: Visualization
            st.subheader("Pattern Analysis Chart")
            with st.spinner('Creating visualization...'):
                fig = analyzer.create_pattern_visualization(patterns)
                st.plotly_chart(fig, use_container_width=True)
                logger.info("Visualization created.")

            # Step 5: Display pattern details
            st.subheader("Detected Patterns")
            if patterns:
                for pattern in patterns:
                    with st.expander(f"{pattern.type.value} at {analyzer.df['timestamp'].iloc[pattern.end_idx]}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("Pattern Details:")
                            st.write(f"Confidence: {pattern.confidence:.2f}")
                            st.write(f"Risk/Reward: {pattern.risk_reward:.2f}")
                            st.write(f"Volume Confirmation: {'Yes' if pattern.volume_confirmation else 'No'}")

                        with col2:
                            st.write("Trade Setup:")
                            if 'BULLISH' in pattern.type.value:
                                st.write("Entry: Market structure high")
                                st.write("Stop: Below pattern low")
                                st.write("Target: Based on R:R ratio")
                            elif 'BEARISH' in pattern.type.value:
                                st.write("Entry: Market structure low")
                                st.write("Stop: Above pattern high")
                                st.write("Target: Based on R:R ratio")
            else:
                st.info("No patterns detected for the selected timeframe and candles.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Exception in run_dashboard: {str(e)}", exc_info=True)


if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_page():
    if not st.session_state.initialized:
        # Do heavy initialization here
        st.session_state.initialized = True

def main():
    initialize_page()

    dashboard = PatternDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()