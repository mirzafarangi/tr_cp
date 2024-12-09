import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from enum import Enum
import ta

class TimeFrame(Enum):
    SCALPING = "Scalping"
    SWING = "Swing"
    LONGTERM = "Long-term"

@dataclass
class IchimokuSettings:
    tenkan_period: int
    kijun_period: int
    senkou_span_b_period: int
    timeframe: str
    trading_style: TimeFrame

class IchimokuAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.settings = {
            "15m": IchimokuSettings(6, 13, 26, "15m", TimeFrame.SCALPING),
            "1h": IchimokuSettings(6, 13, 26, "1h", TimeFrame.SCALPING),
            "4h": IchimokuSettings(9, 26, 52, "4h", TimeFrame.SWING),
            "8h": IchimokuSettings(9, 26, 52, "8h", TimeFrame.SWING),
            "1w": IchimokuSettings(13, 37, 68, "1w", TimeFrame.LONGTERM),
            "1M": IchimokuSettings(13, 37, 68, "1M", TimeFrame.LONGTERM)
        }
        
    def calculate_ichimoku(self, timeframe: str) -> pd.DataFrame:
        """Calculate Ichimoku indicators for given timeframe"""
        df = self.df.copy()
        settings = self.settings[timeframe]
        
        # Calculate Tenkan-sen
        high_tenkan = df['high'].rolling(window=settings.tenkan_period).max()
        low_tenkan = df['low'].rolling(window=settings.tenkan_period).min()
        df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen
        high_kijun = df['high'].rolling(window=settings.kijun_period).max()
        low_kijun = df['low'].rolling(window=settings.kijun_period).min()
        df['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(settings.kijun_period)
        
        # Calculate Senkou Span B
        high_senkou = df['high'].rolling(window=settings.senkou_span_b_period).max()
        low_senkou = df['low'].rolling(window=settings.senkou_span_b_period).min()
        df['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(settings.kijun_period)
        
        # Calculate Chikou Span
        df['chikou_span'] = df['close'].shift(-settings.kijun_period)
        
        # Calculate cloud metrics
        df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b']).fillna(0)
        df['price_cloud_distance'] = (df['close'] - ((df['senkou_span_a'] + df['senkou_span_b']) / 2)).fillna(0)
        
        # Identify flat Kumo areas (Span B)
        span_b_std = df['senkou_span_b'].rolling(window=5).std()
        df['flat_kumo'] = (span_b_std < span_b_std.quantile(0.2)).fillna(False)
        
        # Forward fill NaN values in Ichimoku components
        ichimoku_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        df[ichimoku_columns] = df[ichimoku_columns].fillna(method='ffill')
        
        return df
    
    def analyze_trend_strength(self, df: pd.DataFrame) -> str:
        """Analyze trend strength using price position relative to cloud"""
        latest = df.iloc[-1]
        
        price = latest['close']
        span_a = latest['senkou_span_a']
        span_b = latest['senkou_span_b']
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if price > cloud_top:
            distance_ratio = (price - cloud_top) / latest['cloud_thickness']
            if distance_ratio > 2:
                return "Strong Uptrend (Overextended)"
            return "Strong Uptrend"
        elif price < cloud_bottom:
            distance_ratio = (cloud_bottom - price) / latest['cloud_thickness']
            if distance_ratio > 2:
                return "Strong Downtrend (Overextended)"
            return "Strong Downtrend"
        else:
            return "In Cloud (Neutral)"
    
    def check_timeframe_confluence(self, short_term_df: pd.DataFrame, long_term_df: pd.DataFrame) -> str:
        """Check for confluence between timeframes"""
        short_trend = self.analyze_trend_strength(short_term_df)
        long_trend = self.analyze_trend_strength(long_term_df)
        
        if "Strong Uptrend" in short_trend and "Strong Uptrend" in long_trend:
            return "Strong Bullish Confluence"
        elif "Strong Downtrend" in short_trend and "Strong Downtrend" in long_trend:
            return "Strong Bearish Confluence"
        elif "Strong" in short_trend or "Strong" in long_trend:
            return "Mixed Signals (Weak Confluence)"
        return "No Clear Confluence"
    
    def create_ichimoku_chart(self, timeframe: str) -> go.Figure:
        """Create Ichimoku chart with all components"""
        df = self.calculate_ichimoku(timeframe)
        
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ), row=1, col=1)
        
        # Add Ichimoku components
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['tenkan_sen'],
            name="Tenkan-sen",
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['kijun_sen'],
            name="Kijun-sen",
            line=dict(color='red')
        ), row=1, col=1)
        
        # Add cloud
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['senkou_span_a'],
            name="Senkou Span A",
            line=dict(color='green', width=0),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['senkou_span_b'],
            name="Senkou Span B",
            line=dict(color='red', width=0),
            fill='tonexty',
            fillcolor='rgba(0,250,0,0.1)',
            showlegend=False
        ), row=1, col=1)
        
        # Add Chikou Span
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['chikou_span'],
            name="Chikou Span",
            line=dict(color='purple', dash='dot')
        ), row=1, col=1)
        
        # Add volume
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name="Volume",
            marker_color=colors
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Ichimoku Cloud Analysis ({timeframe})",
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig

class IchimokuDashboard:
    def __init__(self):
        self.data_path = self._get_data_path()
        
    def _get_data_path(self) -> str:
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config['data_path']
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, parse_dates=['timestamp'])
    
    def run_dashboard(self):
        st.title("Advanced Ichimoku Analysis")
        
        try:
            # Load data
            df = self.load_data()
            analyzer = IchimokuAnalyzer(df)
            
            # Timeframe selection
            timeframe = st.selectbox(
                "Select Timeframe for Visualization",
                ["15m", "1h", "4h", "8h", "1w", "1M"]
            )
            
            # Create and display main chart
            st.subheader("Ichimoku Cloud Chart")
            fig = analyzer.create_ichimoku_chart(timeframe)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display analysis tables for all timeframes
            st.subheader("Multi-Timeframe Analysis")
            
            for tf in ["15m", "1h", "4h", "8h", "1w", "1M"]:
                with st.expander(f"{tf} Analysis"):
                    df_tf = analyzer.calculate_ichimoku(tf)
                    latest = df_tf.iloc[-1]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Key Levels")
                        st.write(f"Tenkan-sen: {latest['tenkan_sen']:.8f}")
                        st.write(f"Kijun-sen: {latest['kijun_sen']:.8f}")
                        st.write(f"Cloud Top: {max(latest['senkou_span_a'], latest['senkou_span_b']):.8f}")
                        st.write(f"Cloud Bottom: {min(latest['senkou_span_a'], latest['senkou_span_b']):.8f}")
                    
                    with col2:
                        st.markdown("### Cloud Analysis")
                        st.write(f"Cloud Thickness: {latest['cloud_thickness']:.8f}")
                        st.write(f"Price-Cloud Distance: {latest['price_cloud_distance']:.8f}")
                        st.write(f"Flat Kumo: {'Yes' if latest['flat_kumo'] else 'No'}")
                        st.write(f"Trend Strength: {analyzer.analyze_trend_strength(df_tf)}")
            
            # Display confluence analysis
            st.subheader("Timeframe Confluence Analysis")
            
            # Short-term confluence (15m & 1h)
            df_15m = analyzer.calculate_ichimoku("15m")
            df_1h = analyzer.calculate_ichimoku("1h")
            st.write("Short-term Confluence (15m & 1h):")
            st.write(analyzer.check_timeframe_confluence(df_15m, df_1h))
            
            # Medium-term confluence (4h & 8h)
            df_4h = analyzer.calculate_ichimoku("4h")
            df_8h = analyzer.calculate_ichimoku("8h")
            st.write("Medium-term Confluence (4h & 8h):")
            st.write(analyzer.check_timeframe_confluence(df_4h, df_8h))
            
            # Long-term confluence (1w & 1M)
            df_1w = analyzer.calculate_ichimoku("1w")
            df_1m = analyzer.calculate_ichimoku("1M")
            st.write("Long-term Confluence (1w & 1M):")
            st.write(analyzer.check_timeframe_confluence(df_1w, df_1m))


            # Add this right after the confluence analysis section in the run_dashboard method:

            # Educational Section
            st.markdown("---")
            st.header("Ichimoku Components Explained")
            
            latest = df.iloc[-1]  # Get latest values for dynamic content
            
            with st.expander("### 1. Tenkan-sen (Conversion Line)"):
                st.markdown(f"""
                **Current Value**: {latest['tenkan_sen']:.8f}
                
                **Definition**:
                - Average of highest high and lowest low over last 9 periods
                - Formula: (Highest High (9) + Lowest Low (9)) / 2
                
                **Purpose**:
                - Reflects short-term trend direction
                - Acts as a signal line for momentum and triggers potential entries
                
                **Interpretation**:
                - If price is above Tenkan-sen → short-term trend is bullish
                - If price is below Tenkan-sen → short-term trend is bearish
                
                **Use Cases**:
                1. **Momentum Indicator**:
                   - Sharp turns show acceleration/deceleration in trend
                2. **Entry Signal**:
                   - Bullish entry when crossing above Kijun-sen
                """)
            
            with st.expander("### 2. Kijun-sen (Base Line)"):
                st.markdown(f"""
                **Current Value**: {latest['kijun_sen']:.8f}
                
                **Definition**:
                - Average of highest high and lowest low over last 26 periods
                - Formula: (Highest High (26) + Lowest Low (26)) / 2
                
                **Purpose**:
                - Represents medium-term trend direction
                - Serves as key support/resistance level and trailing stop
                
                **Interpretation**:
                - Price above Kijun-sen → bullish momentum
                - Price below Kijun-sen → bearish momentum
                
                **Use Cases**:
                1. **Support/Resistance**:
                   - Bounce from Kijun-sen confirms trend support
                2. **Trailing Stop**:
                   - Dynamic stop-loss adjustment in trends
                """)
            
            with st.expander("### 3. Cloud Analysis"):
                cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
                cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
                
                st.markdown(f"""
                **Cloud Top (Senkou Span A)**: {latest['senkou_span_a']:.8f}
                **Cloud Bottom (Senkou Span B)**: {latest['senkou_span_b']:.8f}
                
                **Cloud Thickness**: {latest['cloud_thickness']:.8f}
                **Price-Cloud Distance**: {latest['price_cloud_distance']:.8f}
                **Flat Kumo**: {'Yes' if latest['flat_kumo'] else 'No'}
                
                **Purpose**:
                - Forms trading cloud (Kumo)
                - Indicates dynamic support/resistance zones
                - Projects future trend direction
                
                **Interpretation**:
                - Thick cloud ({latest['cloud_thickness']:.8f}) → Strong support/resistance
                - Current price vs cloud: {latest['price_cloud_distance']:.8f} → 
                  {"Bullish momentum" if latest['price_cloud_distance'] > 0 else "Bearish pressure"}
                """)
            
            with st.expander("### Trading Scenarios"):
                # Determine current scenario
                price = latest['close']
                tenkan = latest['tenkan_sen']
                kijun = latest['kijun_sen']
                
                if price > cloud_top and tenkan > kijun:
                    scenario = """
                    **Current Scenario: Strong Bullish**
                    - Price above cloud with momentum
                    - Tenkan-sen above Kijun-sen
                    
                    **Suggested Actions**:
                    - Look for pullbacks to Kijun-sen
                    - Trail stops below Kijun-sen
                    - Target next resistance levels
                    """
                elif price < cloud_bottom and tenkan < kijun:
                    scenario = """
                    **Current Scenario: Strong Bearish**
                    - Price below cloud with downward momentum
                    - Tenkan-sen below Kijun-sen
                    
                    **Suggested Actions**:
                    - Watch for bounces to cloud bottom
                    - Keep stops above cloud
                    - Target previous support levels
                    """
                else:
                    scenario = """
                    **Current Scenario: Neutral/Consolidation**
                    - Price near or inside cloud
                    - Mixed indicator signals
                    
                    **Suggested Actions**:
                    - Wait for clear breakout
                    - Reduce position sizes
                    - Focus on shorter timeframes
                    """
                
                st.markdown(scenario)
                
                st.markdown("""
                **Key Scenario Patterns**:
                
                1. **Reversal Setup**:
                   - Thin cloud
                   - Price testing cloud edge
                   - Tenkan-sen momentum shift
                
                2. **Trend Continuation**:
                   - Thick cloud
                   - Price well outside cloud
                   - Strong Tenkan-Kijun alignment
                
                3. **Consolidation Pattern**:
                   - Price inside cloud
                   - Flat Tenkan/Kijun
                   - Reduced cloud thickness
                """)
                
            
        except Exception as e:
            st.error(f"Error in dashboard: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def main():
    dashboard = IchimokuDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()