import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import ta

@dataclass
class PricePattern:
    name: str
    start_idx: int
    end_idx: int
    pattern_type: str  # 'bullish' or 'bearish'
    confidence: float
    description: str

@dataclass
class LiquidityZone:
    price_level: float
    zone_type: str  # 'sweep' or 'block'
    strength: float
    description: str

class CandlestickAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.patterns = []
        self.liquidity_zones = []
        
    def _create_engulfing_pattern(self, idx: int, window: pd.DataFrame) -> PricePattern:
        """Create engulfing pattern object"""
        current = window.iloc[-1]
        pattern_type = 'bullish' if current['close'] > current['open'] else 'bearish'
        
        return PricePattern(
            name="Engulfing",
            start_idx=idx-1,
            end_idx=idx,
            pattern_type=pattern_type,
            confidence=0.85,
            description=f"{pattern_type.capitalize()} engulfing pattern detected"
        )
    
    def _create_pin_pattern(self, idx: int, candle: pd.Series) -> PricePattern:
        """Create pin bar pattern object"""
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        pattern_type = 'bullish' if lower_wick > upper_wick else 'bearish'
        
        return PricePattern(
            name="Pin Bar",
            start_idx=idx,
            end_idx=idx,
            pattern_type=pattern_type,
            confidence=0.75,
            description=f"{pattern_type.capitalize()} pin bar detected"
        )
    
    def _create_liquidity_sweep(self, idx: int, window: pd.DataFrame) -> LiquidityZone:
        """Create liquidity sweep zone object"""
        current = window.iloc[-1]
        
        return LiquidityZone(
            price_level=current['high'] if current['close'] > current['open'] else current['low'],
            zone_type='sweep',
            strength=0.8,
            description="Potential liquidity sweep zone"
        )
    
    def _create_institutional_block(self, idx: int, window: pd.DataFrame) -> LiquidityZone:
        """Create institutional block zone object"""
        current = window.iloc[-1]
        
        return LiquidityZone(
            price_level=current['close'],
            zone_type='block',
            strength=0.9,
            description="Potential institutional order block"
        )
    
    def _is_inside_bar(self, window: pd.DataFrame) -> bool:
        current = window.iloc[-1]
        previous = window.iloc[-2]
        
        return (current['high'] < previous['high'] and 
                current['low'] > previous['low'])

    def identify_patterns(self) -> List[PricePattern]:
        """Identify candlestick patterns in the data"""
        patterns = []
        
        for i in range(3, len(self.df)):
            window = self.df.iloc[i-3:i+1]
            
            # Engulfing Pattern
            if self._is_engulfing_pattern(window):
                pattern = self._create_engulfing_pattern(i, window)
                patterns.append(pattern)
            
            # Pin Bar Pattern
            if self._is_pin_bar(window.iloc[-1]):
                pattern = self._create_pin_pattern(i, window.iloc[-1])
                patterns.append(pattern)
                
            # Inside Bar Pattern
            if self._is_inside_bar(window.iloc[-2:]):
                pattern = PricePattern(
                    name="Inside Bar",
                    start_idx=i-1,
                    end_idx=i,
                    pattern_type="neutral",
                    confidence=0.8,
                    description="Inside bar formation - potential breakout setup"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _add_patterns_to_chart(self, fig: go.Figure):
        """Add pattern annotations to chart"""
        for pattern in self.patterns:
            pattern_color = 'green' if pattern.pattern_type == 'bullish' else 'red'
            
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
    
    def _add_liquidity_zones_to_chart(self, fig: go.Figure):
        """Add liquidity zone annotations to chart"""
        for zone in self.liquidity_zones:
            zone_color = 'blue' if zone.zone_type == 'sweep' else 'purple'
            
            fig.add_hline(
                y=zone.price_level,
                line_dash="dash",
                line_color=zone_color,
                opacity=0.5,
                row=1, col=1
            )
    
    def create_candlestick_chart(self) -> go.Figure:
        """Create main candlestick chart with patterns and zones"""
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3])
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=self.df['timestamp'],
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name="OHLC"
        ), row=1, col=1)
        
        # Add volume bars
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(self.df['close'], self.df['open'])]
        
        fig.add_trace(go.Bar(
            x=self.df['timestamp'],
            y=self.df['volume'],
            name="Volume",
            marker_color=colors
        ), row=2, col=1)
        
        # Add patterns
        self.patterns = self.identify_patterns()
        self._add_patterns_to_chart(fig)
        
        # Add liquidity zones
        self.liquidity_zones = self.identify_liquidity_zones()
        self._add_liquidity_zones_to_chart(fig)
        
        # Update layout
        fig.update_layout(
            title="Advanced Price Action Analysis",
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
    
    def create_technical_indicators_chart(self) -> go.Figure:
        """Create chart with technical indicators"""
        fig = make_subplots(rows=4, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.4, 0.2, 0.2, 0.2])
        
        # Add RSIs
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['rsi_7'],
            name="RSI(7)"
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['rsi_14'],
            name="RSI(14)"
        ), row=2, col=1)
        
        # Add MACD
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['macd'],
            name="MACD"
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['signal'],
            name="Signal"
        ), row=3, col=1)
        
        # Add Stochastic
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['Stoch_Fast_K_scalping'],
            name="Stoch Fast K"
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['Stoch_Fast_D_scalping'],
            name="Stoch Fast D"
        ), row=4, col=1)
        
        # Add horizontal lines for RSI and Stochastic
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title="Technical Indicators",
            height=800,
            showlegend=True,
            yaxis2_title="RSI",
            yaxis3_title="MACD",
            yaxis4_title="Stochastic"
        )
        
        return fig

class TradingDashboard:
    def __init__(self):
        self.data_path = self._get_data_path()
        
    def _get_data_path(self) -> str:
        """Get data path from config file"""
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config['data_path']
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare last 30 candles of data"""
        df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        return df.tail(30).copy()
    
    def run_dashboard(self):
        """Run the main dashboard"""
        st.title("Advanced Candlestick Analysis")
        
        try:
            # Load data
            df = self.load_data()
            
            # Initialize analyzer
            analyzer = CandlestickAnalyzer(df)
            
            # Create main candlestick chart
            st.subheader("Price Action Analysis")
            candlestick_fig = analyzer.create_candlestick_chart()
            st.plotly_chart(candlestick_fig, use_container_width=True)
            
            # Create technical indicators chart
            st.subheader("Technical Analysis")
            indicators_fig = analyzer.create_technical_indicators_chart()
            st.plotly_chart(indicators_fig, use_container_width=True)
            
            # Pattern Analysis Section
            st.subheader("Detected Patterns")
            patterns = analyzer.identify_patterns()
            
            if patterns:
                for pattern in patterns:
                    with st.expander(f"{pattern.name} Pattern"):
                        st.write(f"Type: {pattern.pattern_type}")
                        st.write(f"Confidence: {pattern.confidence:.2f}")
                        st.write(f"Description: {pattern.description}")
            else:
                st.write("No significant patterns detected in current timeframe")
            
            # Liquidity Analysis Section
            st.subheader("Liquidity Analysis")
            zones = analyzer.identify_liquidity_zones()
            
            if zones:
                for zone in zones:
                    with st.expander(f"Liquidity Zone at {zone.price_level:.8f}"):
                        st.write(f"Type: {zone.zone_type}")
                        st.write(f"Strength: {zone.strength:.2f}")
                        st.write(f"Description: {zone.description}")
            else:
                st.write("No significant liquidity zones detected")
            
            # Trading Recommendations
            st.subheader("Trading Analysis")
            self._display_trading_analysis(df)
            
        except Exception as e:
            st.error(f"Error in dashboard: {str(e)}")
            st.error(f"Error details: {type(e).__name__}")
            import traceback
            st.error(traceback.format_exc())
    
    def _display_trading_analysis(self, df: pd.DataFrame):
        """Display trading analysis and recommendations"""
        current_data = df.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Key Levels")
            st.write(f"Current Price: {current_data['close']:.8f}")
            st.write(f"Key Support: {current_data['strong_support']:.8f}")
            st.write(f"Key Resistance: {current_data['major_resistance']:.8f}")
        
        with col2:
            st.markdown("### Technical Status")
            st.write(f"Trend Status: {current_data['current_signals']}")
            st.write(f"Overall Recommendation: {current_data['overall_recommendation']}")

def main():
    dashboard = TradingDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()