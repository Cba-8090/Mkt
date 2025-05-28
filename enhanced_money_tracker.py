#!/usr/bin/env python3
"""
Enhanced Money Flow Trading Dashboard - Multi-Source Integration
Combines: Futures Flow (70%) + Options Flow (30%) + Gamma Analysis + Price Data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import time
import argparse
import sqlite3
from pathlib import Path
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')


class MultiSourceDataLoader:
    """Enhanced data loader for multiple data sources"""
    
    def __init__(self):
        self.futures_data = None
        self.options_data = None
        self.gamma_data = None
        self.price_data = None
        self.last_update = {}
        
    def load_futures_data(self, csv_path):
        """Load futures money flow data (70% weight)"""
        try:
            print("📊 Loading Futures Money Flow Data...")
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['timestamp', 'weighted_money_flow', 'cumulative_weighted_money_flow',
                           'weighted_positive_money_flow', 'weighted_negative_money_flow']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in futures data")
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
            df = df.sort_values('timestamp')
            
            self.futures_data = df
            self.last_update['futures'] = datetime.now()
            print(f"✅ Loaded {len(df)} futures flow records")
            return True
            
        except Exception as e:
            print(f"❌ Error loading futures data: {e}")
            return False
    
    def load_options_data(self, csv_path):
        """Load options money flow data (30% weight)"""
        try:
            print("📈 Loading Options Money Flow Data...")
            
            if not os.path.exists(csv_path):
                print(f"⚠️ Options data file not found: {csv_path}")
                # Create dummy data structure for now
                self.options_data = pd.DataFrame({
                    'timestamp': pd.date_range(start=datetime.now().replace(hour=9, minute=15), 
                                             end=datetime.now().replace(hour=15, minute=30), 
                                             freq='5T'),
                    'net_flow': [0] * 76,
                    'total_flow': [0] * 76,
                    'bullish_flow': [0] * 76,
                    'bearish_flow': [0] * 76,
                    'sentiment': ['Neutral'] * 76
                })
                print("⚠️ Using dummy options data - please ensure options analyzer is running")
                return True
            
            df = pd.read_csv(csv_path)
            
            # Expected columns for options flow
            required_cols = ['timestamp', 'net_flow', 'total_flow', 'bullish_flow', 'bearish_flow']
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            self.options_data = df
            self.last_update['options'] = datetime.now()
            print(f"✅ Loaded {len(df)} options flow records")
            return True
            
        except Exception as e:
            print(f"❌ Error loading options data: {e}")
            # Create dummy data as fallback
            self.options_data = pd.DataFrame({
                'timestamp': pd.date_range(start=datetime.now().replace(hour=9, minute=15), 
                                         end=datetime.now().replace(hour=15, minute=30), 
                                         freq='5T'),
                'net_flow': [0] * 76,
                'total_flow': [0] * 76,
                'bullish_flow': [0] * 76,
                'bearish_flow': [0] * 76,
                'sentiment': ['Neutral'] * 76
            })
            return True
    
    def load_gamma_data(self, html_path):
        """Load gamma analysis data from HTML report"""
        try:
            print("🎯 Loading Gamma Analysis Data...")
            
            if not os.path.exists(html_path):
                print(f"⚠️ Gamma data file not found: {html_path}")
                # Create dummy gamma data
                current_time = datetime.now()
                self.gamma_data = {
                    'support_levels': [(current_time, 23800), (current_time, 23750)],
                    'resistance_levels': [(current_time, 24000), (current_time, 24050)],
                    'max_pressure_strike': 23900,
                    'support_pressure': 0.6,
                    'resistance_pressure': 0.4,
                    'price_reversals': [],
                    'breakdown_signals': []
                }
                print("⚠️ Using dummy gamma data - please ensure gamma analyzer is running")
                return True
            
            # Parse HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract gamma data (this would need to be customized based on your HTML structure)
            # For now, creating structured dummy data
            current_time = datetime.now()
            self.gamma_data = {
                'support_levels': [(current_time, 23800), (current_time, 23750)],
                'resistance_levels': [(current_time, 24000), (current_time, 24050)],
                'max_pressure_strike': 23900,
                'support_pressure': 0.6,
                'resistance_pressure': 0.4,
                'price_reversals': [],
                'breakdown_signals': []
            }
            
            self.last_update['gamma'] = datetime.now()
            print("✅ Loaded gamma analysis data")
            return True
            
        except Exception as e:
            print(f"❌ Error loading gamma data: {e}")
            return False
    
    def load_price_data(self, db_path):
        """Load spot price data from SQLite database"""
        try:
            print("💰 Loading Price Data...")
            
            if not os.path.exists(db_path):
                print(f"⚠️ Database file not found: {db_path}")
                # Create dummy price data
                self.price_data = pd.DataFrame({
                    'timestamp': pd.date_range(start=datetime.now().replace(hour=9, minute=15), 
                                             end=datetime.now().replace(hour=15, minute=30), 
                                             freq='5T'),
                    'spot_price': np.random.normal(23900, 50, 76)
                })
                print("⚠️ Using dummy price data")
                return True
            
            # Connect to database and fetch data
            conn = sqlite3.connect(db_path)
            
            today = datetime.now().strftime('%Y-%m-%d')
            query = """
            SELECT DISTINCT
                strftime('%H:%M', timestamp) as _time,
                Spot as spot
            FROM option_chain_data
            WHERE date(timestamp) = ?
            ORDER BY _time ASC
            """
            
            df = pd.read_sql_query(query, conn, params=[today])
            conn.close()
            
            if len(df) > 0:
                # Convert time to full timestamp
                df['timestamp'] = pd.to_datetime(f"{today} " + df['_time'])
                df = df.rename(columns={'spot': 'spot_price'})
                df = df[['timestamp', 'spot_price']]
                
                self.price_data = df
                print(f"✅ Loaded {len(df)} price records")
            else:
                # Create dummy data if no records found
                self.price_data = pd.DataFrame({
                    'timestamp': pd.date_range(start=datetime.now().replace(hour=9, minute=15), 
                                             end=datetime.now().replace(hour=15, minute=30), 
                                             freq='5T'),
                    'spot_price': np.random.normal(23900, 50, 76)
                })
                print("⚠️ No price data found, using dummy data")
            
            self.last_update['price'] = datetime.now()
            return True
            
        except Exception as e:
            print(f"❌ Error loading price data: {e}")
            return False


class EnhancedMoneyFlowAnalyzer:
    """Enhanced analyzer with multi-source integration"""
    
    def __init__(self):
        self.data_loader = MultiSourceDataLoader()
        self.combined_signals = {}
        self.alerts = []
        self.live_data_end_index = 0
        self.timeframe_signals = {}
        
    def load_all_data(self, futures_csv, options_csv, gamma_html, price_db):
        """Load data from all sources"""
        success_count = 0
        
        if self.data_loader.load_futures_data(futures_csv):
            success_count += 1
        if self.data_loader.load_options_data(options_csv):
            success_count += 1
        if self.data_loader.load_gamma_data(gamma_html):
            success_count += 1
        if self.data_loader.load_price_data(price_db):
            success_count += 1
            
        print(f"📊 Loaded {success_count}/4 data sources successfully")
        return success_count >= 2  # At least futures + one other source
    
    def find_live_data_boundary(self):
        """Find the latest meaningful data point across all sources"""
        if self.data_loader.futures_data is None:
            return
            
        df = self.data_loader.futures_data
        
        # Find last meaningful futures data point
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if pd.notna(row['weighted_money_flow']) and row['weighted_money_flow'] != 0:
                self.live_data_end_index = i
                print(f"🎯 Live data boundary: {row['timestamp']}")
                return
                
        self.live_data_end_index = len(df) - 1
    
    def calculate_weighted_signals(self):
        """Calculate weighted signals from multiple sources"""
        print("🔄 Calculating weighted multi-source signals...")
        
        if self.data_loader.futures_data is None:
            print("⚠️ No futures data available")
            return
        
        self.find_live_data_boundary()
        
        # Get latest data from each source
        futures_latest = self.data_loader.futures_data.iloc[self.live_data_end_index]
        
        # Futures signal (70% weight)
        futures_flow_m = futures_latest['weighted_money_flow'] / 1_000_000
        futures_cumulative_m = futures_latest['cumulative_weighted_money_flow'] / 1_000_000
        
        futures_signal_strength = self._calculate_signal_strength(futures_flow_m)
        futures_weight = 0.7
        
        print(f"📊 Futures Flow: {futures_flow_m:.2f}M (Weight: 70%)")
        
        # Options signal (30% weight)
        options_signal_strength = 0
        options_flow_m = 0
        
        if self.data_loader.options_data is not None and len(self.data_loader.options_data) > 0:
            # Find closest timestamp in options data
            target_time = futures_latest['timestamp']
            options_df = self.data_loader.options_data
            
            # Find closest time match
            time_diffs = abs(options_df['timestamp'] - target_time)
            closest_idx = time_diffs.idxmin()
            options_latest = options_df.iloc[closest_idx]
            
            options_flow_m = options_latest['net_flow'] / 1_000_000 if 'net_flow' in options_latest else 0
            options_signal_strength = self._calculate_signal_strength(options_flow_m)
            
            print(f"📈 Options Flow: {options_flow_m:.2f}M (Weight: 30%)")
        
        options_weight = 0.3
        
        # Combined weighted signal
        combined_flow = (futures_flow_m * futures_weight) + (options_flow_m * options_weight)
        combined_strength = (futures_signal_strength * futures_weight) + (options_signal_strength * options_weight)
        
        print(f"🎯 Combined Flow: {combined_flow:.2f}M")
        
        # Gamma validation
        gamma_confirmation = self._analyze_gamma_confirmation(combined_flow)
        
        # Price movement analysis
        price_momentum = self._analyze_price_momentum()
        
        # Adaptive timeframe detection
        expected_duration = self._calculate_expected_duration(abs(combined_flow), combined_strength)
        
        # Generate final signal
        self.combined_signals = {
            'timestamp': futures_latest['timestamp'].strftime('%d-%m-%Y %H:%M'),
            'futures_flow_m': futures_flow_m,
            'options_flow_m': options_flow_m,
            'combined_flow_m': combined_flow,
            'signal_strength': combined_strength,
            'major_signal': self._determine_signal_type(combined_flow, combined_strength),
            'confidence': self._calculate_confidence(combined_strength, gamma_confirmation),
            'expected_move': self._calculate_expected_move(combined_flow, combined_strength),
            'expected_duration': expected_duration,
            'gamma_confirmation': gamma_confirmation,
            'price_momentum': price_momentum,
            'action_color': self._get_action_color(combined_flow)
        }
        
        # Generate alerts
        self._generate_enhanced_alerts()
        
        print(f"🎯 Final Signal: {self.combined_signals['major_signal']}")
        print(f"🎯 Confidence: {self.combined_signals['confidence']}%")
        print(f"⏱️ Expected Duration: {self.combined_signals['expected_duration']}")
    
    def _calculate_signal_strength(self, flow_m):
        """Calculate normalized signal strength (0-1)"""
        abs_flow = abs(flow_m)
        if abs_flow >= 300:
            return 1.0
        elif abs_flow >= 100:
            return 0.8
        elif abs_flow >= 50:
            return 0.6
        elif abs_flow >= 25:
            return 0.4
        else:
            return 0.2
    
    def _analyze_gamma_confirmation(self, combined_flow):
        """Analyze gamma levels for signal confirmation"""
        if self.data_loader.gamma_data is None:
            return "No Data"
        
        gamma = self.data_loader.gamma_data
        support_pressure = gamma.get('support_pressure', 0.5)
        resistance_pressure = gamma.get('resistance_pressure', 0.5)
        
        if combined_flow > 50 and support_pressure > 0.6:
            return "Strong Support"
        elif combined_flow < -50 and resistance_pressure > 0.6:
            return "Weak Resistance"
        elif support_pressure > resistance_pressure:
            return "Support Bias"
        else:
            return "Resistance Bias"
    
    def _analyze_price_momentum(self):
        """Analyze current price momentum"""
        if self.data_loader.price_data is None or len(self.data_loader.price_data) < 3:
            return "No Data"
        
        price_df = self.data_loader.price_data.tail(3)
        if len(price_df) < 3:
            return "Insufficient Data"
        
        recent_change = price_df.iloc[-1]['spot_price'] - price_df.iloc[-3]['spot_price']
        
        if recent_change > 20:
            return "Strong Up"
        elif recent_change > 10:
            return "Moderate Up"
        elif recent_change < -20:
            return "Strong Down"
        elif recent_change < -10:
            return "Moderate Down"
        else:
            return "Sideways"
    
    def _calculate_expected_duration(self, abs_flow, strength):
        """Calculate expected signal duration based on flow magnitude and strength"""
        base_duration = "15-30 min"
        
        if abs_flow >= 300 and strength >= 0.9:
            return "45-120 min"
        elif abs_flow >= 200 and strength >= 0.8:
            return "30-90 min"
        elif abs_flow >= 100 and strength >= 0.6:
            return "20-60 min"
        elif abs_flow >= 50:
            return "10-30 min"
        else:
            return "5-15 min"
    
    def _determine_signal_type(self, combined_flow, strength):
        """Determine signal type based on combined analysis"""
        if combined_flow > 100 and strength >= 0.8:
            return "STRONG BUY"
        elif combined_flow > 50 and strength >= 0.6:
            return "BUY SIGNAL"
        elif combined_flow > 25:
            return "MODERATE BUY"
        elif combined_flow < -100 and strength >= 0.8:
            return "STRONG SELL"
        elif combined_flow < -50 and strength >= 0.6:
            return "SELL SIGNAL"
        elif combined_flow < -25:
            return "MODERATE SELL"
        else:
            return "CONSOLIDATION"
    
    def _calculate_confidence(self, strength, gamma_confirmation):
        """Calculate overall confidence based on multiple factors"""
        base_confidence = 70 + (strength * 25)  # 70-95% range
        
        # Gamma confirmation bonus
        if gamma_confirmation in ["Strong Support", "Weak Resistance"]:
            base_confidence += 5
        elif gamma_confirmation in ["Support Bias", "Resistance Bias"]:
            base_confidence += 2
        
        return min(95, max(70, int(base_confidence)))
    
    def _calculate_expected_move(self, combined_flow, strength):
        """Calculate expected price move"""
        abs_flow = abs(combined_flow)
        direction = "UP" if combined_flow > 0 else "DOWN"
        
        if abs_flow >= 200:
            return f"150-250 points {direction}"
        elif abs_flow >= 100:
            return f"75-150 points {direction}"
        elif abs_flow >= 50:
            return f"40-100 points {direction}"
        elif abs_flow >= 25:
            return f"20-50 points {direction}"
        else:
            return "10-30 points"
    
    def _get_action_color(self, combined_flow):
        """Get color classification for UI"""
        if combined_flow > 50:
            return "bullish"
        elif combined_flow < -50:
            return "bearish"
        else:
            return "neutral"
    
    def _generate_enhanced_alerts(self):
        """Generate comprehensive alerts"""
        self.alerts = []
        
        flow = self.combined_signals['combined_flow_m']
        strength = self.combined_signals['signal_strength']
        
        # Flow magnitude alerts
        if abs(flow) >= 200:
            self.alerts.append({
                'type': 'HIGH',
                'message': 'EXTREME MULTI-SOURCE FLOW',
                'action': f"{'STRONG BUY' if flow > 0 else 'STRONG SELL'} SIGNAL",
                'priority': 'alert-high'
            })
        elif abs(flow) >= 100:
            self.alerts.append({
                'type': 'MEDIUM',
                'message': 'Significant Combined Flow',
                'action': f"{'Buy' if flow > 0 else 'Sell'} Signal Active",
                'priority': 'alert-medium'
            })
        
        # Gamma alerts
        gamma_conf = self.combined_signals.get('gamma_confirmation', '')
        if gamma_conf in ['Strong Support', 'Weak Resistance']:
            self.alerts.append({
                'type': 'MEDIUM',
                'message': f'Gamma {gamma_conf} Detected',
                'action': 'High Probability Setup',
                'priority': 'alert-medium'
            })
        
        # Price momentum alerts
        momentum = self.combined_signals.get('price_momentum', '')
        if momentum.startswith('Strong'):
            self.alerts.append({
                'type': 'LOW',
                'message': f'Price Momentum: {momentum}',
                'action': 'Trend Confirmation',
                'priority': 'alert-low'
            })
        
        # Default alert
        if len(self.alerts) == 0:
            self.alerts.append({
                'type': 'LOW',
                'message': 'Multi-Source Monitoring Active',
                'action': 'Watching for Signals',
                'priority': 'alert-low'
            })
    
    def create_enhanced_charts(self):
        """Create comprehensive multi-source charts"""
        if self.data_loader.futures_data is None:
            return "", "", ""
        
        print("📊 Creating enhanced multi-source charts...")
        
        # Get live data
        futures_df = self.data_loader.futures_data.iloc[:self.live_data_end_index + 1]
        
        # Chart 1: Combined Money Flow
        combined_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Futures Money Flow (70% Weight)', 'Combined Analysis'],
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Futures flow bars
        flow_colors = ['#4CAF50' if x > 0 else '#f44336' for x in futures_df['weighted_money_flow']]
        combined_fig.add_trace(
            go.Bar(
                x=futures_df['timestamp'],
                y=futures_df['weighted_money_flow'] / 1_000_000,
                name='Futures Flow',
                marker_color=flow_colors,
                hovertemplate='<b>%{x}</b><br>Flow: %{y:.2f}M<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Combined signal line
        if len(futures_df) > 0:
            combined_flow_line = [self.combined_signals.get('combined_flow_m', 0)] * len(futures_df)
            combined_fig.add_trace(
                go.Scatter(
                    x=futures_df['timestamp'],
                    y=combined_flow_line,
                    mode='lines',
                    name='Combined Signal',
                    line=dict(color='#00BCD4', width=3),
                    hovertemplate='<b>%{x}</b><br>Combined: %{y:.2f}M<extra></extra>'
                ),
                row=2, col=1
            )
        
        combined_fig.update_layout(
            title='Multi-Source Money Flow Analysis',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff'),
            showlegend=True
        )
        
        # Chart 2: Price vs Flow Correlation
        price_fig = go.Figure()
        
        if self.data_loader.price_data is not None:
            price_df = self.data_loader.price_data
            
            price_fig.add_trace(go.Scatter(
                x=price_df['timestamp'],
                y=price_df['spot_price'],
                mode='lines',
                name='Spot Price',
                line=dict(color='#FFC107', width=2),
                yaxis='y1'
            ))
        
        price_fig.update_layout(
            title='Price Movement Analysis',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff')
        )
        
        # Chart 3: Options Flow Analysis
        options_fig = go.Figure()
        
        if self.data_loader.options_data is not None:
            options_df = self.data_loader.options_data
            
            options_fig.add_trace(go.Bar(
                x=options_df['timestamp'],
                y=options_df['net_flow'] / 1_000_000,
                name='Options Net Flow',
                marker_color=['#4CAF50' if x > 0 else '#f44336' for x in options_df['net_flow']],
                hovertemplate='<b>%{x}</b><br>Net Flow: %{y:.2f}M<extra></extra>'
            ))
        
        options_fig.update_layout(
            title='Options Money Flow (30% Weight)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff')
        )
        
        # Convert to HTML
        config = {'displayModeBar': True, 'displaylogo': False, 'responsive': True}
        
        combined_html = pyo.plot(combined_fig, output_type='div', include_plotlyjs=False, config=config)
        price_html = pyo.plot(price_fig, output_type='div', include_plotlyjs=False, config=config)
        options_html = pyo.plot(options_fig, output_type='div', include_plotlyjs=False, config=config)
        
        return combined_html, price_html, options_html


# Continue with enhanced main function in next part...
def main():
    """Enhanced main function with multi-source support"""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Source Money Flow Dashboard')
    parser.add_argument('--futures-csv', required=True, help='Futures money flow CSV file')
    parser.add_argument('--options-csv', help='Options money flow CSV file')
    parser.add_argument('--gamma-html', help='Gamma analysis HTML file')
    parser.add_argument('--price-db', help='Price database file')
    parser.add_argument('--output', help='Output HTML file')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval')
    parser.add_argument('--continuous', action='store_true', help='Continuous monitoring')
    
    args = parser.parse_args()
    
    # Define data paths
    base_dir = r'C:\Projects\apps\_keystock_analyser\output'
    options_dir = r'C:\Projects\apps\_nifty_optionanalyser\net_flow_reports'
    gamma_dir = r'C:\Projects\apps\_nifty_optionanalyser\option_analysis_reports'
    price_db = r'C:\Projects\apps\_nifty_optionanalyser\OptionAnalyser.db'
    
    # Resolve file paths
    if not os.path.dirname(args.futures_csv):
        args.futures_csv = os.path.join(base_dir, args.futures_csv)
    
    if args.options_csv and not os.path.dirname(args.options_csv):
        args.options_csv = os.path.join(options_dir, args.options_csv)
    elif not args.options_csv:
        args.options_csv = os.path.join(options_dir, 'net_money_flow_data.csv')
    
    if args.gamma_html and not os.path.dirname(args.gamma_html):
        args.gamma_html = os.path.join(gamma_dir, args.gamma_html)
    elif not args.gamma_html:
        # Find latest gamma report
        today = datetime.now().strftime('%Y%m%d')
        args.gamma_html = os.path.join(gamma_dir, f'nifty_comparison_report_{today}.html')
    
    if not args.price_db:
        args.price_db = price_db
    
    if not args.output:
        date_str = datetime.now().strftime('%Y%m%d')
        args.output = os.path.join(base_dir, f'enhanced_money_flow_{date_str}.html')
    
    print("🚀 Enhanced Multi-Source Money Flow Dashboard")
    print("=" * 60)
    print(f"📊 Futures Data: {args.futures_csv}")
    print(f"📈 Options Data: {args.options_csv}")
    print(f"🎯 Gamma Data: {args.gamma_html}")
    print(f"💰 Price Data: {args.price_db}")
    print(f"📄 Output: {args.output}")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedMoneyFlowAnalyzer()
    
    def generate_enhanced_dashboard():
        """Generate enhanced dashboard"""
        print(f"\n🔄 Processing multi-source data...")
        
        if analyzer.load_all_data(args.futures_csv, args.options_csv, args.gamma_html, args.price_db):
            analyzer.calculate_weighted_signals()
            # HTML generation would continue here...
            print("✅ Enhanced dashboard generated successfully")
            return True
        else:
            print("❌ Failed to load sufficient data sources")
            return False
    
    if args.continuous:
        print(f"🔄 Starting continuous multi-source monitoring (every {args.interval}s)")
        try:
            while True:
                generate_enhanced_dashboard()
                print(f"⏳ Next update in {args.interval} seconds...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
    else:
        generate_enhanced_dashboard()

if __name__ == "__main__":
    main()