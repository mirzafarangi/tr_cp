import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class CloudColor(Enum):
    BULLISH = "GREEN"
    BEARISH = "RED"
    NEUTRAL = "GREY"

class TrendStrength(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    MODERATE_BULLISH = "MODERATE_BULLISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    NEUTRAL = "NEUTRAL"
    WEAK_BEARISH = "WEAK_BEARISH"
    MODERATE_BEARISH = "MODERATE_BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

@dataclass
class IchimokuParameters:
    tenkan_period: int = 9        # Conversion line period
    kijun_period: int = 26       # Base line period
    senkou_b_period: int = 52    # Leading Span B period
    displacement: int = 26       # Displacement period
    chikou_period: int = 26      # Lagging Span period
    
    # Additional advanced parameters
    kumo_breakout_threshold: float = 0.02  # % threshold for cloud breakout
    trend_strength_threshold: float = 0.05  # % threshold for trend strength
    flat_kumo_threshold: float = 0.001     # % threshold for flat kumo detection
    
    # Time-based parameters
    time_weight: float = 0.7     # Weight for time-based analysis
    volume_weight: float = 0.3   # Weight for volume-based analysis

class AdvancedIchimokuSystem:
    def __init__(self, params: Optional[IchimokuParameters] = None):
        self.params = params or IchimokuParameters()
        self.signals_history: List[Dict] = []
        
    def calculate_ichimoku_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all Ichimoku components with advanced metrics"""
        df = df.copy()
        
        # Basic Ichimoku components
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 1. Tenkan-sen (Conversion Line)
        period_high = high.rolling(window=self.params.tenkan_period).max()
        period_low = low.rolling(window=self.params.tenkan_period).min()
        df['tenkan_sen'] = (period_high + period_low) / 2
        
        # 2. Kijun-sen (Base Line)
        period_high = high.rolling(window=self.params.kijun_period).max()
        period_low = low.rolling(window=self.params.kijun_period).min()
        df['kijun_sen'] = (period_high + period_low) / 2
        
        # 3. Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.params.displacement)
        
        # 4. Senkou Span B (Leading Span B)
        period_high = high.rolling(window=self.params.senkou_b_period).max()
        period_low = low.rolling(window=self.params.senkou_b_period).min()
        df['senkou_span_b'] = ((period_high + period_low) / 2).shift(self.params.displacement)
        
        # 5. Chikou Span (Lagging Span)
        df['chikou_span'] = close.shift(-self.params.chikou_period)
        
        # Advanced Components
        
        # 6. Cloud Thickness
        df['cloud_thickness'] = df['senkou_span_a'] - df['senkou_span_b']
        
        # 7. Cloud Color
        df['cloud_color'] = np.where(
            df['senkou_span_a'] > df['senkou_span_b'],
            CloudColor.BULLISH.value,
            np.where(
                df['senkou_span_a'] < df['senkou_span_b'],
                CloudColor.BEARISH.value,
                CloudColor.NEUTRAL.value
            )
        )
        
        # 8. Price-Cloud Distance
        df['price_cloud_distance'] = np.where(
            close > df['senkou_span_a'],
            close - df['senkou_span_a'],
            np.where(
                close < df['senkou_span_b'],
                close - df['senkou_span_b'],
                0
            )
        )
        
        # 9. Flat Kumo Detection
        df['is_flat_kumo'] = (
            df['cloud_thickness'].rolling(window=5).std() < self.params.flat_kumo_threshold
        )
        
        # 10. Advanced Momentum Indicators
        df['tenkan_momentum'] = df['tenkan_sen'].diff()
        df['kijun_momentum'] = df['kijun_sen'].diff()
        
        # 11. Time-in-Cloud Analysis
        df['time_in_cloud'] = np.where(
            (close >= df['senkou_span_b']) & (close <= df['senkou_span_a']) |
            (close >= df['senkou_span_a']) & (close <= df['senkou_span_b']),
            1, 0
        ).rolling(window=20).sum()
        
        return df
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive trend strength indicators"""
        df = df.copy()
        close = df['close']
        
        # Price position relative to cloud
        cloud_position = np.where(
            close > df['senkou_span_a'],
            1,
            np.where(
                close < df['senkou_span_b'],
                -1,
                0
            )
        )
        
        # Trend line alignments
        trend_alignment = np.where(
            (df['tenkan_sen'] > df['kijun_sen']) & 
            (df['kijun_sen'] > df['senkou_span_a']) & 
            (df['senkou_span_a'] > df['senkou_span_b']),
            1,
            np.where(
                (df['tenkan_sen'] < df['kijun_sen']) & 
                (df['kijun_sen'] < df['senkou_span_a']) & 
                (df['senkou_span_a'] < df['senkou_span_b']),
                -1,
                0
            )
        )
        
        # Chikou span position
        chikou_position = np.where(
            df['chikou_span'] > close.shift(self.params.chikou_period),
            1,
            np.where(
                df['chikou_span'] < close.shift(self.params.chikou_period),
                -1,
                0
            )
        )
        
        # Combined strength calculation
        strength = (
            cloud_position * 0.4 +
            trend_alignment * 0.4 +
            chikou_position * 0.2
        )
        
        df['trend_strength'] = np.where(
            strength > 0.6, TrendStrength.STRONG_BULLISH.value,
            np.where(
                strength > 0.2, TrendStrength.MODERATE_BULLISH.value,
                np.where(
                    strength > 0, TrendStrength.WEAK_BULLISH.value,
                    np.where(
                        strength == 0, TrendStrength.NEUTRAL.value,
                        np.where(
                            strength > -0.2, TrendStrength.WEAK_BEARISH.value,
                            np.where(
                                strength > -0.6, TrendStrength.MODERATE_BEARISH.value,
                                TrendStrength.STRONG_BEARISH.value
                            )
                        )
                    )
                )
            )
        )
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive trading signals"""
        df = df.copy()
        
        # Basic signal conditions
        bullish_conditions = (
            (df['close'] > df['senkou_span_a']) &
            (df['close'] > df['senkou_span_b']) &
            (df['tenkan_sen'] > df['kijun_sen']) &
            (df['chikou_span'] > df['close'].shift(self.params.chikou_period))
        )
        
        bearish_conditions = (
            (df['close'] < df['senkou_span_a']) &
            (df['close'] < df['senkou_span_b']) &
            (df['tenkan_sen'] < df['kijun_sen']) &
            (df['chikou_span'] < df['close'].shift(self.params.chikou_period))
        )
        
        # Signal generation with confidence levels
        df['signal'] = np.where(
            bullish_conditions & (df['trend_strength'] == TrendStrength.STRONG_BULLISH.value),
            'STRONG_BUY',
            np.where(
                bullish_conditions,
                'BUY',
                np.where(
                    bearish_conditions & (df['trend_strength'] == TrendStrength.STRONG_BEARISH.value),
                    'STRONG_SELL',
                    np.where(
                        bearish_conditions,
                        'SELL',
                        'NEUTRAL'
                    )
                )
            )
        )
        
        # Add signal confidence based on multiple factors
        df['signal_confidence'] = np.where(
            df['signal'].isin(['STRONG_BUY', 'STRONG_SELL']),
            'HIGH',
            np.where(
                df['signal'].isin(['BUY', 'SELL']),
                'MEDIUM',
                'LOW'
            )
        )
        
        return df
    
    def analyze_timeframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Complete analysis for a single timeframe"""
        # Calculate all components
        df = self.calculate_ichimoku_components(df)
        df = self.calculate_trend_strength(df)
        df = self.generate_signals(df)
        
        # Generate analysis summary
        summary = {
            'trend_distribution': df['trend_strength'].value_counts().to_dict(),
            'signal_distribution': df['signal'].value_counts().to_dict(),
            'avg_cloud_thickness': df['cloud_thickness'].mean(),
            'flat_kumo_percentage': df['is_flat_kumo'].mean() * 100,
            'time_in_cloud_percentage': (df['time_in_cloud'] / 20).mean() * 100,
            'current_trend': df['trend_strength'].iloc[-1],
            'current_signal': df['signal'].iloc[-1],
            'signal_confidence': df['signal_confidence'].iloc[-1]
        }
        
        return df, summary
    
    def generate_report(self, df: pd.DataFrame, timeframe: str) -> str:
        """Generate detailed analysis report"""
        _, summary = self.analyze_timeframe(df)
        
        report = f"""
Ichimoku Analysis Report - {timeframe} Timeframe
================================================

Current Market Status:
---------------------
Trend: {summary['current_trend']}
Signal: {summary['current_signal']}
Confidence: {summary['signal_confidence']}

Cloud Analysis:
--------------
Average Cloud Thickness: {summary['avg_cloud_thickness']:.4f}
Flat Kumo Percentage: {summary['flat_kumo_percentage']:.2f}%
Time in Cloud: {summary['time_in_cloud_percentage']:.2f}%

Trend Distribution:
------------------
"""
        
        for trend, count in summary['trend_distribution'].items():
            percentage = (count / len(df)) * 100
            report += f"{trend}: {percentage:.2f}%\n"
            
        report += f"""
Signal Distribution:
-------------------
"""
        
        for signal, count in summary['signal_distribution'].items():
            percentage = (count / len(df)) * 100
            report += f"{signal}: {percentage:.2f}%\n"
            
        return report

    def resample_timeframe(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe with volume-weighted calculations"""
        resampled = df.resample(target_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate volume-weighted price for more accurate resampling
        df['vw_price'] = df['close'] * df['volume']
        vwap = df['vw_price'].resample(target_timeframe).sum() / resampled['volume']
        resampled['vwap'] = vwap
        
        return resampled.dropna()

    def analyze_multiple_timeframes(self, df_15m: pd.DataFrame) -> Dict:
        """Analyze multiple timeframes with correlation analysis"""
        timeframes = {
            '15m': df_15m,
            '4h': self.resample_timeframe(df_15m, '4H'),
            '1d': self.resample_timeframe(df_15m, '1D')
        }
        
        results = {}
        for timeframe, data in timeframes.items():
            analyzed_df, summary = self.analyze_timeframe(data)
            results[timeframe] = {
                'data': analyzed_df,
                'summary': summary,
                'report': self.generate_report(analyzed_df, timeframe)
            }
            
        # Add cross-timeframe analysis
        results['cross_timeframe_analysis'] = self._analyze_timeframe_correlation(results)
        
        return results
    
    def _analyze_timeframe_correlation(self, results: Dict) -> Dict:
        """Analyze correlation and confluence between timeframes"""
        timeframes = ['15m', '4h', '1d']
        correlation = {}
        
        for i in range(len(timeframes)):
            for j in range(i + 1, len(timeframes)):
                tf1, tf2 = timeframes[i], timeframes[j]
                
                # Compare trend directions
                trend_agreement = (
                    results[tf1]['data']['trend_strength'].iloc[-1] ==
                    results[tf2]['data']['trend_strength'].iloc[-1]
                )
                
                # Compare signals
                signal_agreement = (
                    results[tf1]['data']['signal'].iloc[-1] ==
                    results[tf2]['data']['signal'].iloc[-1]
                )
                
                correlation[f'{tf1}_vs_{tf2}'] = {
                    'trend_agreement': trend_agreement,
                    'signal_agreement': signal_agreement,
                    'confidence_level': 'HIGH' if (trend_agreement and signal_agreement) else 'MEDIUM' if (trend_agreement or signal_agreement) else 'LOW'
                }
                
        return correlation

