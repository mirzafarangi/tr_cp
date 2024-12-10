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
        
        # Calculate cloud thickness
        df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
        
        # Calculate distance from cloud
        df['price_cloud_distance'] = df['close'] - df[['senkou_span_a', 'senkou_span_b']].mean(axis=1)
        
        # Identify flat Kumo areas (Span B)
        span_b_std = df['senkou_span_b'].rolling(window=5).std()
        df['flat_kumo'] = span_b_std < span_b_std.quantile(0.2)
        
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

            st.markdown("---")
            
            with st.expander("Understanding Ichimoku Components", expanded=False):
                st.markdown("""### **1. Tenkan-sen (Conversion Line): `0.00002487`**

- **Definition**:Tenkan-sen=2Highest High (9)+Lowest Low (9)
    
    The average of the highest high and lowest low over the last **9 periods** (or a custom period).
    
    Tenkan-sen=Highest High (9)+Lowest Low (9)2
    
- **Purpose**:
    - Reflects **short-term trend direction**.
    - Acts as a **signal line** for momentum and triggers potential entries.
- **Interpretation**:
    - If the price is **above Tenkan-sen**, the short-term trend is bullish.
    - If the price is **below Tenkan-sen**, the short-term trend is bearish.
- **Use Cases**:
    1. **Momentum Indicator**:
        - When Tenkan-sen turns upwards or downwards sharply, it shows acceleration or deceleration in the trend.
    2. **Entry Signal**:
        - If Tenkan-sen crosses above Kijun-sen, it signals a bullish entry.

### **2. Kijun-sen (Base Line): `0.00002530`**

- **Definition**:Kijun-sen=2Highest High (26)+Lowest Low (26)
    
    The average of the highest high and lowest low over the last **26 periods** (or a custom period).
    
    Kijun-sen=Highest High (26)+Lowest Low (26)2
    
- **Purpose**:
    - Represents **medium-term trend direction**.
    - Serves as a **key support/resistance level** and trailing stop.
- **Interpretation**:
    - Price **above Kijun-sen** suggests bullish momentum.
    - Price **below Kijun-sen** suggests bearish momentum.
- **Use Cases**:
    1. **Support/Resistance**:
        - If price retraces to Kijun-sen and bounces, it’s a confirmation of support in a bullish trend.
    2. **Trailing Stop**:
        - Use Kijun-sen to dynamically adjust stop-loss levels in trending markets.

---

### **3. Cloud Top (Senkou Span A): `0.00002719`**

- **Definition**:Senkou Span A=2Tenkan-sen+Kijun-sen
    
    The average of Tenkan-sen and Kijun-sen, plotted **26 periods ahead**.
    
    Senkou Span A=Tenkan-sen+Kijun-sen2
    
- **Purpose**:
    - Forms the **upper boundary of the Kumo (cloud)**.
    - Indicates **dynamic resistance** in bearish trends and **dynamic support** in bullish trends.
- **Interpretation**:
    - If price is **above the Cloud Top**, the trend is bullish.
    - If price is **below the Cloud Top**, it may encounter resistance.
- **Use Cases**:
    1. **Resistance Zone**:
        - Price approaching the Cloud Top from below often stalls or reverses due to resistance.
    2. **Reversal Signal**:
        - A break above the Cloud Top can signify a shift from bearish to bullish momentum.

---

### **4. Cloud Bottom (Senkou Span B): `0.00002704`**

- **Definition**:Senkou Span B=2Highest High (52)+Lowest Low (52)
    
    The average of the highest high and lowest low over the last **52 periods**, plotted **26 periods ahead**.
    
    Senkou Span B=Highest High (52)+Lowest Low (52)2
    
- **Purpose**:
    - Forms the **lower boundary of the Kumo (cloud)**.
    - Represents stronger and longer-term support/resistance.
- **Interpretation**:
    - Price **above Cloud Bottom** indicates support.
    - Price **below Cloud Bottom** suggests further bearish pressure.
- **Use Cases**:
    1. **Support Zone**:
        - When price bounces off the Cloud Bottom, it confirms strong support in bullish trends.
    2. **Continuation Signal**:
        - Price staying between the Cloud Bottom and Cloud Top indicates indecision but often trends persist.

---

### **5. Cloud Thickness: `0.00000015`**

- **Definition**:Cloud Thickness=Senkou Span A−Senkou Span B
    
    The difference between Senkou Span A and Senkou Span B.
    
    Cloud Thickness=Senkou Span A−Senkou Span B
    
- **Purpose**:
    - Measures **strength of the Kumo** as support or resistance.
    - Thick clouds = Strong barriers. Thin clouds = Easier breakouts.
- **Interpretation**:
    - **Thin Cloud**: Price can easily break through, signaling weaker trends.
    - **Thick Cloud**: Acts as a strong support or resistance zone.
- **Use Cases**:
    1. **Breakout Scenarios**:
        - Price breaking a thin cloud is more reliable for reversals.
    2. **Trend Strength**:
        - In trending markets, a thick cloud reinforces the prevailing trend.

---

### **6. Price-Cloud Distance: `0.00000092`**

- **Definition**:Price-Cloud Distance=Current Price−Nearest Edge of the Cloud
    
    The distance between the current price and the nearest edge of the Kumo.
    
    Price-Cloud Distance=Current Price−Nearest Edge of the Cloud
    
- **Purpose**:
    - Indicates how overextended or close price is to key support/resistance zones.
- **Interpretation**:
    - **Positive Distance**: Price is above the cloud → Strong bullish momentum.
    - **Negative Distance**: Price is below the cloud → Bearish momentum.
- **Use Cases**:
    1. **Overextension Detection**:
        - If price is far from the cloud, expect a pullback or consolidation.
    2. **Entry Confirmation**:
        - Enter trades when price closes near the cloud edge and bounces in the trend’s direction.

---

### **7. Flat Kumo: `No`**

- **Definition**:
    
    Flat Kumo occurs when Senkou Span B is flat (unchanging for several periods).
    
- **Purpose**:
    - Represents **price equilibrium** and strong support/resistance.
- **Interpretation**:
    - Flat Kumo acts as a magnet for price; it tends to revisit these levels.
- **Use Cases**:
    1. **Reversal Signals**:
        - Price approaching a flat Kumo from above may reverse (support).
    2. **Pullback Opportunities**:
        - Look for price retracements to flat Kumo zones for better entry points.

---

### **Scenarios and Interpretations**

### **Scenario 1: Reversal Opportunity**

- **Variables**:
    - Tenkan-sen: Rising.
    - Kijun-sen: Flat.
    - Cloud Thickness: Thin.
    - Price-Cloud Distance: Slightly negative.
- **Interpretation**:
    
    Price is testing resistance (cloud edge), but the thin Kumo suggests a potential breakout. A rising Tenkan-sen confirms bullish momentum.
    
- **Action**:
    - **Enter long** if price closes above the cloud.
    - **Stop-loss** below Kijun-sen.

---

### **Scenario 2: Trending Market**

- **Variables**:
    - Tenkan-sen > Kijun-sen.
    - Cloud Top > Cloud Bottom (bullish cloud).
    - Price far above cloud (positive Price-Cloud Distance).
    - Cloud Thickness: Thick.
- **Interpretation**:
    
    Strong bullish trend, with the cloud acting as robust support.
    
- **Action**:
    - Use Kijun-sen for pullback entries.
    - Trail stop below Kijun-sen to lock profits.

---

### **Scenario 3: Bearish Rejection**

- **Variables**:
    - Tenkan-sen < Kijun-sen.
    - Price inside the cloud.
    - Cloud Thickness: Moderate.
    - Price-Cloud Distance: Neutral.
- **Interpretation**:
    
    Bearish rejection likely as price struggles to exit the cloud.
    
- **Action**:
    - **Enter short** on confirmation of a bearish candlestick close below the cloud.
    - Target the next support zone.""")

            
                
            
        except Exception as e:
            st.error(f"Error in dashboard: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_page():
    if not st.session_state.initialized:
        # Do heavy initialization here
        st.session_state.initialized = True

def main():
    initialize_page()
    dashboard = IchimokuDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()