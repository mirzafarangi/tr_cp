import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
from ichimoko_class import AdvancedIchimokuSystem, IchimokuParameters, CloudColor, TrendStrength

def get_data_path():
    """Get data path from config file"""
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get('data_path', '')
    return ''

def load_data():
    """Load and prepare data for Ichimoku analysis"""
    try:
        df = pd.read_csv(get_data_path())
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_ichimoku_plot(df: pd.DataFrame, timeframe: str) -> go.Figure:
    """Create an interactive Ichimoku plot"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'Ichimoku Cloud - {timeframe}', 'Volume')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Ichimoku components
    colors = {
        'tenkan': '#0496ff',      # Blue
        'kijun': '#991515',       # Dark Red
        'span_a': '#4caf50',      # Green
        'span_b': '#ff5252',      # Red
        'chikou': '#9c27b0'       # Purple
    }

    # Add Ichimoku lines
    for component in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']:
        if component in df.columns:
            color = colors.get(component.split('_')[0], '#000000')
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[component],
                    name=component.replace('_', ' ').title(),
                    line=dict(color=color),
                    opacity=0.7
                ),
                row=1, col=1
            )

    # Add cloud
    if 'senkou_span_a' in df.columns and 'senkou_span_b' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index.tolist() + df.index.tolist()[::-1],
                y=df['senkou_span_a'].tolist() + df['senkou_span_b'].tolist()[::-1],
                fill='tonexty',
                fillcolor='rgba(0,250,0,0.1)',
                line=dict(width=0),
                showlegend=False,
                name='Cloud'
            ),
            row=1, col=1
        )

    # Volume bars with colors based on price movement
    colors = ['red' if close < open else 'green' 
             for open, close in zip(df['open'], df['close'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title_text=f"Ichimoku Analysis - {timeframe}",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def display_metrics(df: pd.DataFrame, timeframe: str):
    """Display current market metrics"""
    latest = df.iloc[-1]
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Price Analysis")
        st.metric("Current Price", f"{latest['close']:.8f}")
        st.metric("Trend Strength", latest['trend_strength'])
        st.metric("Signal", latest['signal'])
        
    with col2:
        st.markdown("### Cloud Metrics")
        st.metric("Cloud Color", latest['cloud_color'])
        st.metric("Cloud Thickness", f"{latest['cloud_thickness']:.8f}")
        st.metric("Time in Cloud %", f"{(latest['time_in_cloud']/20*100):.1f}%")
        
    with col3:
        st.markdown("### Technical Levels")
        st.metric("Tenkan-sen", f"{latest['tenkan_sen']:.8f}")
        st.metric("Kijun-sen", f"{latest['kijun_sen']:.8f}")
        st.metric("Signal Confidence", latest['signal_confidence'])

def main():
    st.set_page_config(layout="wide", page_title="Ichimoku Analysis")
    
    st.title("Advanced Ichimoku Cloud Analysis")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Initialize Ichimoku system
    ichimoku = AdvancedIchimokuSystem()
    
    # Add sidebar controls
    st.sidebar.title("Analysis Settings")
    lookback = st.sidebar.slider("Lookback Period (days)", 1, 30, 7)
    
    # Calculate date range
    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=lookback)
    df_period = df[df.index >= start_date]
    
    # Analyze multiple timeframes
    try:
        results = ichimoku.analyze_multiple_timeframes(df_period)
        
        # Create tabs for different timeframes
        tabs = st.tabs(['15m', '4h', '1d', 'Cross-Timeframe Analysis'])
        
        # Display analysis for each timeframe
        for timeframe, tab in zip(['15m', '4h', '1d'], tabs[:3]):
            with tab:
                st.subheader(f"{timeframe} Timeframe Analysis")
                
                # Display metrics
                display_metrics(results[timeframe]['data'], timeframe)
                
                # Plot chart
                fig = create_ichimoku_plot(results[timeframe]['data'], timeframe)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed report in expander
                with st.expander("Detailed Analysis Report"):
                    st.text(results[timeframe]['report'])
        
        # Cross-timeframe analysis
        with tabs[3]:
            st.subheader("Cross-Timeframe Analysis")
            
            for comparison, data in results['cross_timeframe_analysis'].items():
                st.write(f"### {comparison}")
                
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Trend Agreement", "✓" if data['trend_agreement'] else "✗")
                with cols[1]:
                    st.metric("Signal Agreement", "✓" if data['signal_agreement'] else "✗")
                with cols[2]:
                    st.metric("Confidence Level", data['confidence_level'])
                
                st.markdown("---")
        
        # Additional analysis in expandable sections
        with st.expander("Trading Recommendations"):
            for timeframe in ['15m', '4h', '1d']:
                df_tf = results[timeframe]['data']
                latest = df_tf.iloc[-1]
                
                st.write(f"### {timeframe} Trading Signals")
                st.write(f"Primary Signal: **{latest['signal']}**")
                st.write(f"Confidence: **{latest['signal_confidence']}**")
                st.write(f"Trend Strength: **{latest['trend_strength']}**")
                st.markdown("---")
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.error("Please check if the data format matches the expected structure")

if __name__ == "__main__":
    main()