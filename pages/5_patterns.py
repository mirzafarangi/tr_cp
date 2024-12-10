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
    def __init__(self, df: pd.DataFrame, timeframe: str):
        self.df = df.copy()
        self.timeframe = timeframe
        self.patterns = []
        self.preprocess_data()
        
    def preprocess_data(self):
        """Calculate necessary indicators and metrics"""
        # Volatility metrics
        self.df['atr'] = ta.volatility.average_true_range(
            self.df['high'], self.df['low'], self.df['close']
        )
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
        
    def find_all_patterns(self) -> List[Pattern]:
        """Detect all pattern types"""
        patterns = []
        
        # Basic patterns
        patterns.extend(self.find_engulfing_patterns())
        patterns.extend(self.find_pinbar_patterns())
        patterns.extend(self.find_inside_outside_bars())
        
        # Advanced patterns
        patterns.extend(self.find_liquidity_sweeps())
        patterns.extend(self.find_order_blocks())
        patterns.extend(self.find_fair_value_gaps())
        
        # Volume patterns
        patterns.extend(self.find_institutional_accumulation())
        patterns.extend(self.find_volume_divergence())
        
        # Market structure patterns
        patterns.extend(self.find_market_structure_breaks())
        
        # Volatility patterns
        patterns.extend(self.find_volatility_patterns())
        
        self.patterns = sorted(patterns, key=lambda x: x.end_idx)
        return self.patterns
    
    def find_engulfing_patterns(self) -> List[Pattern]:
        """Detect engulfing patterns with volume confirmation"""
        patterns = []
        
        for i in range(1, len(self.df)-1):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            
            # Bullish engulfing
            if (current['close'] > current['open'] and
                previous['close'] < previous['open'] and
                current['open'] < previous['close'] and
                current['close'] > previous['open'] and
                current['volume'] > current['volume_sma']):
                
                patterns.append(Pattern(
                    type=PatternType.ENGULFING_BULLISH,
                    start_idx=i-1,
                    end_idx=i,
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
                    start_idx=i-1,
                    end_idx=i,
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
        
        for i in range(1, len(self.df)-1):
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
                    start_idx=i,
                    end_idx=i,
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
                    start_idx=i,
                    end_idx=i,
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
    
    def create_pattern_visualization(self, patterns: List[Pattern]) -> go.Figure:
        """Create visualization with pattern annotations"""
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=self.df['timestamp'],
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name="Price"
        ), row=1, col=1)
        
        # Add volume
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(self.df['close'], self.df['open'])]
        
        fig.add_trace(go.Bar(
            x=self.df['timestamp'],
            y=self.df['volume'],
            name="Volume",
            marker_color=colors
        ), row=2, col=1)
        
        # Add pattern annotations
        for pattern in patterns:
            pattern_color = 'green' if 'BULLISH' in pattern.type.value else 'red'
            
            # Add shape for pattern
            fig.add_shape(
                type="rect",
                x0=self.df['timestamp'].iloc[pattern.start_idx],
                x1=self.df['timestamp'].iloc[pattern.end_idx],
                y0=self.df['low'].iloc[pattern.start_idx:pattern.end_idx+1].min(),
                y1=self.df['high'].iloc[pattern.start_idx:pattern.end_idx+1].max(),
                line=dict(color=pattern_color, width=1),
                fillcolor=pattern_color,
                opacity=0.2,
                row=1, col=1
            )
            
            # Add annotation
            fig.add_annotation(
                x=self.df['timestamp'].iloc[pattern.end_idx],
                y=self.df['high'].iloc[pattern.end_idx],
                text=pattern.type.value,
                showarrow=True,
                arrowhead=1,
                row=1, col=1
            )
        
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
            # Load data
            df = self.load_data()
            
            # Timeframe selection
            timeframe = st.selectbox(
                "Select Timeframe for Analysis",
                ["15m", "1h", "4h", "8h", "1w", "1M"]
            )
            
            # Initialize analyzer
            analyzer = PatternAnalyzer(df, timeframe)
            patterns = analyzer.find_all_patterns()
            
            # Create and display visualization
            st.subheader("Pattern Analysis Chart")
            fig = analyzer.create_pattern_visualization(patterns)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display pattern details
            st.subheader("Detected Patterns")
            
            for pattern in patterns:
                with st.expander(f"{pattern.type.value} at {df['timestamp'].iloc[pattern.end_idx]}"):
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
            
        except Exception as e:
            st.error(f"Error in dashboard: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def main():
    dashboard = PatternDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()