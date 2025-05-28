#!/usr/bin/env python3
"""
Money Flow Trading Dashboard Generator - Part 1
Core Classes and Data Loading
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from datetime import datetime, timedelta
import json
import os
import time
import argparse
from pathlib import Path


class MoneyFlowAnalyzer:
    """Money Flow Analysis Engine"""

    def __init__(self):
        self.df = None
        self.live_data_end_index = 0
        self.signals = {}
        self.alerts = []

    def load_data(self, csv_file_path):
        """Load and validate CSV data"""
        try:
            print(f"📂 Loading data from: {csv_file_path}")

            # Read CSV with proper data types
            self.df = pd.read_csv(csv_file_path)

            # Validate required columns
            required_columns = [
                'timestamp', 'weighted_money_flow', 'cumulative_weighted_money_flow',
                'weighted_positive_money_flow', 'weighted_negative_money_flow'
            ]

            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")

            # Convert timestamp to datetime if it's not already
            if self.df['timestamp'].dtype == 'object':
                try:
                    self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%d-%m-%Y %H:%M')
                except:
                    # Try alternative format
                    self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

            # Find live data boundary
            self.find_live_data_end()

            print(f"✅ Loaded {len(self.df)} rows")
            print(f"🎯 Live data until: {self.get_latest_timestamp()}")
            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def find_live_data_end(self):
        """Find the last meaningful trading data point"""
        print("🔍 Detecting live data boundary...")

        for i in range(len(self.df) - 1, -1, -1):
            row = self.df.iloc[i]

            # Primary check: weighted_money_flow (most important for signals)
            if pd.notna(row['weighted_money_flow']) and row['weighted_money_flow'] != 0:
                self.live_data_end_index = i
                print(f"🎯 Live data detected at index {i}: {row['timestamp']}")
                print(f"💰 Flow: {row['weighted_money_flow'] / 1_000_000:.2f}M")
                return

            # Secondary check: positive or negative flows
            if (pd.notna(row['weighted_positive_money_flow']) and row['weighted_positive_money_flow'] != 0) or \
                    (pd.notna(row['weighted_negative_money_flow']) and row['weighted_negative_money_flow'] != 0):
                self.live_data_end_index = i
                print(f"🎯 Live data detected at index {i}: {row['timestamp']} (pos/neg flows)")
                return

        # Fallback: use all data if no meaningful data found
        self.live_data_end_index = len(self.df) - 1
        print(f"⚠️ No meaningful data found, using last row: {self.df.iloc[self.live_data_end_index]['timestamp']}")

    def get_latest_timestamp(self):
        """Get the timestamp of the latest data"""
        if self.df is not None and len(self.df) > 0:
            latest_row = self.df.iloc[self.live_data_end_index]
            return latest_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        return "No data"

    def get_latest_display_timestamp(self):
        """Get formatted timestamp for display"""
        if self.df is not None and len(self.df) > 0:
            latest_row = self.df.iloc[self.live_data_end_index]
            return latest_row['timestamp'].strftime('%d-%m-%Y %H:%M')
        return "No data"

    def analyze_signals(self):
        """Generate trading signals from live data"""
        if self.df is None or len(self.df) == 0:
            print("⚠️ No data available for signal analysis")
            return

        print("📊 Analyzing trading signals...")

        # Get live data only
        live_df = self.df.iloc[:self.live_data_end_index + 1]
        latest = live_df.iloc[-1]

        # Convert to millions
        flow_m = latest['weighted_money_flow'] / 1_000_000
        cumulative_m = latest['cumulative_weighted_money_flow'] / 1_000_000

        print(f"💰 Current Flow: {flow_m:.2f}M")
        print(f"📈 Cumulative Flow: {cumulative_m:.2f}M")

        # Determine signal strength based on documentation thresholds
        if abs(flow_m) >= 300:
            strength = 'EXTREME'
            confidence = 95
            expected_move = f"{'150-200+ points UP' if flow_m > 0 else '200+ points DOWN'}"
            lead_time = f"{'15-45 min' if flow_m > 0 else '30-60 min'}"
        elif abs(flow_m) >= 100:
            strength = 'HIGH'
            confidence = 90
            expected_move = f"{'75-150 points UP' if flow_m > 0 else '100-200 points DOWN'}"
            lead_time = '10-30 min'
        elif abs(flow_m) >= 50:
            strength = 'MEDIUM'
            confidence = 85
            expected_move = f"{'30-75 points UP' if flow_m > 0 else '40-100 points DOWN'}"
            lead_time = f"{'5-20 min' if flow_m > 0 else '10-30 min'}"
        else:
            strength = 'LOW'
            confidence = 80
            expected_move = '15-40 points'
            lead_time = '30-120 min'

        # Determine signal type and action color
        if flow_m > 100:
            signal_type = 'STRONG BUY'
            action_color = 'bullish'
        elif flow_m > 50:
            signal_type = 'BUY SIGNAL'
            action_color = 'bullish'
        elif flow_m < -100:
            signal_type = 'STRONG SELL'
            action_color = 'bearish'
        elif flow_m < -50:
            signal_type = 'SELL SIGNAL'
            action_color = 'bearish'
        else:
            signal_type = 'CONSOLIDATION'
            action_color = 'neutral'

        # Store signals
        self.signals = {
            'timestamp': latest['timestamp'].strftime('%d-%m-%Y %H:%M'),
            'major_signal': signal_type,
            'strength': strength,
            'confidence': confidence,
            'expected_move': expected_move,
            'lead_time': lead_time,
            'flow_m': flow_m,
            'cumulative_m': cumulative_m,
            'action_color': action_color
        }

        print(f"🎯 Signal Generated: {signal_type} ({confidence}% confidence)")

        # Generate alerts
        self.generate_alerts(flow_m, cumulative_m)

    def generate_alerts(self, flow_m, cumulative_m):
        """Generate trading alerts based on flow data"""
        print("🚨 Generating alerts...")
        self.alerts = []

        # Flow magnitude alerts
        if abs(flow_m) >= 300:
            self.alerts.append({
                'type': 'HIGH',
                'message': 'EXTREME FLOW DETECTED',
                'action': f"{'Strong Buy Signal' if flow_m > 0 else 'Major Sell Alert'}",
                'priority': 'alert-high'
            })
            print(f"🔴 HIGH ALERT: EXTREME FLOW ({flow_m:.2f}M)")

        elif abs(flow_m) >= 100:
            self.alerts.append({
                'type': 'MEDIUM',
                'message': 'Significant Money Flow',
                'action': f"{'Buy Signal' if flow_m > 0 else 'Sell Signal'}",
                'priority': 'alert-medium'
            })
            print(f"🟡 MEDIUM ALERT: Significant Flow ({flow_m:.2f}M)")

        # Cumulative flow alerts
        if cumulative_m > 600:
            self.alerts.append({
                'type': 'HIGH',
                'message': 'Potential Peak Formation',
                'action': 'Consider Profit Booking',
                'priority': 'alert-high'
            })
            print(f"🔴 HIGH ALERT: Peak Formation ({cumulative_m:.2f}M)")

        elif cumulative_m < -500:
            self.alerts.append({
                'type': 'HIGH',
                'message': 'Major Distribution Detected',
                'action': 'Avoid New Positions',
                'priority': 'alert-high'
            })
            print(f"🔴 HIGH ALERT: Major Distribution ({cumulative_m:.2f}M)")

        # Session timing alerts
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 10:
            self.alerts.append({
                'type': 'LOW',
                'message': 'Opening Hour - High Volatility',
                'action': 'Follow Momentum',
                'priority': 'alert-low'
            })
            print("🟢 INFO: Opening Hour Volatility")

        # Default alert if no specific conditions met
        if len(self.alerts) == 0:
            self.alerts.append({
                'type': 'LOW',
                'message': 'Normal Market Conditions',
                'action': 'Monitor for Changes',
                'priority': 'alert-low'
            })
            print("🟢 INFO: Normal Market Conditions")

        print(f"✅ Generated {len(self.alerts)} alerts")

    def create_charts(self):
        """Generate interactive Plotly charts"""
        if self.df is None:
            print("⚠️ No data available for chart generation")
            return "", ""

        print("📊 Creating interactive charts...")

        # Get live data only
        live_df = self.df.iloc[:self.live_data_end_index + 1]
        print(f"📈 Charting {len(live_df)} data points")

        # Money Flow Chart with color coding
        flow_colors = []
        for _, row in live_df.iterrows():
            flow = row['weighted_money_flow'] / 1_000_000
            if flow > 100:
                flow_colors.append('#4CAF50')  # Strong positive (green)
            elif flow > 50:
                flow_colors.append('#8BC34A')  # Moderate positive (light green)
            elif flow < -100:
                flow_colors.append('#f44336')  # Strong negative (red)
            elif flow < -50:
                flow_colors.append('#FF5722')  # Moderate negative (orange-red)
            else:
                flow_colors.append('#ff9800')  # Neutral (orange)

        # Create Money Flow Bar Chart
        flow_fig = go.Figure()
        flow_fig.add_trace(go.Bar(
            x=live_df['timestamp'],
            y=live_df['weighted_money_flow'] / 1_000_000,
            name='Money Flow (M)',
            marker=dict(
                color=flow_colors,
                line=dict(width=1, color='#333')
            ),
            hovertemplate='<b>%{x}</b><br>Flow: %{y:.2f}M<extra></extra>'
        ))

        flow_fig.update_layout(
            title=dict(
                text='Money Flow Analysis (5-min intervals)',
                font=dict(color='#e6f1ff', size=16)
            ),
            xaxis=dict(
                title='Time',
                gridcolor='#3b4472',
                tickangle=-45,
                color='#e6f1ff'
            ),
            yaxis=dict(
                title='Flow (Millions)',
                gridcolor='#3b4472',
                zeroline=True,
                zerolinecolor='#666',
                color='#e6f1ff'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff'),
            margin=dict(t=50, b=80, l=60, r=30),
            showlegend=False
        )

        # Create Cumulative Flow Line Chart
        cumulative_fig = go.Figure()
        cumulative_fig.add_trace(go.Scatter(
            x=live_df['timestamp'],
            y=live_df['cumulative_weighted_money_flow'] / 1_000_000,
            mode='lines',
            name='Cumulative Flow (M)',
            line=dict(color='#00BCD4', width=3),
            fill='tonexty',
            fillcolor='rgba(0, 188, 212, 0.1)',
            hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.2f}M<extra></extra>'
        ))

        cumulative_fig.update_layout(
            title=dict(
                text='Cumulative Money Flow Trend',
                font=dict(color='#e6f1ff', size=16)
            ),
            xaxis=dict(
                title='Time',
                gridcolor='#3b4472',
                tickangle=-45,
                color='#e6f1ff'
            ),
            yaxis=dict(
                title='Cumulative Flow (Millions)',
                gridcolor='#3b4472',
                zeroline=True,
                zerolinecolor='#666',
                color='#e6f1ff'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff'),
            margin=dict(t=50, b=80, l=60, r=30),
            showlegend=False
        )

        # Convert charts to HTML divs
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True
        }

        flow_html = pyo.plot(flow_fig, output_type='div', include_plotlyjs=False, config=config)
        cumulative_html = pyo.plot(cumulative_fig, output_type='div', include_plotlyjs=False, config=config)

        print("✅ Charts generated successfully")
        return flow_html, cumulative_html

    def get_statistics(self):
        """Get dashboard statistics"""
        if self.df is None:
            return {
                'total_data_points': 0,
                'live_data_points': 0,
                'significant_flows': 0,
                'accuracy': 0
            }

        live_df = self.df.iloc[:self.live_data_end_index + 1]
        significant_flows = len(live_df[abs(live_df['weighted_money_flow'] / 1_000_000) >= 50])

        return {
            'total_data_points': len(self.df),
            'live_data_points': len(live_df),
            'significant_flows': significant_flows,
            'accuracy': self.signals.get('confidence', 80)
        }


class HTMLGenerator:
    """HTML Dashboard Generator"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def generate_alerts_html(self, alerts):
        """Generate HTML for alerts section"""
        if not alerts:
            return '<div class="alert-item alert-low"><i class="fas fa-info-circle"></i><span>No alerts</span></div>'

        html = ""
        for alert in alerts:
            icon = 'fa-exclamation-triangle' if alert['type'] == 'HIGH' else \
                'fa-info-circle' if alert['type'] == 'MEDIUM' else 'fa-check-circle'

            html += f'''
                <div class="alert-item {alert['priority']}">
                    <i class="fas {icon}"></i>
                    <div>
                        <div style="font-weight: 600;">{alert['message']}</div>
                        <div style="font-size: 12px; color: #8892b0;">{alert['action']}</div>
                    </div>
                </div>
            '''

        return html

    def get_css_styles(self):
        """Get CSS styles for the dashboard"""
        return '''
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0c1022 0%, #1a1f3a 100%);
            color: #ffffff;
            min-height: 100vh;
        }

        .dashboard-header {
            background: linear-gradient(135deg, #1e2749 0%, #2d3561 100%);
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border-bottom: 2px solid #3b4472;
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo-icon {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            padding: 12px;
            border-radius: 50%;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }

        .title-section h1 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(45deg, #4CAF50, #81C784);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .status-section {
            display: flex;
            gap: 20px;
            align-items: center;
        }

        .live-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(76, 175, 80, 0.2);
            border-radius: 20px;
            border: 1px solid #4CAF50;
        }

        .pulse {
            width: 8px;
            height: 8px;
            background: #4CAF50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .main-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }

        .signals-panel {
            background: linear-gradient(135deg, #1e2749 0%, #252b4f 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid #3b4472;
            height: fit-content;
        }

        .charts-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .chart-container {
            background: linear-gradient(135deg, #1e2749 0%, #252b4f 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid #3b4472;
        }

        .signal-card {
            background: linear-gradient(135deg, #2a3266 0%, #323870 100%);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border-left: 4px solid;
        }

        .signal-card.bullish {
            border-left-color: #4CAF50;
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%);
        }

        .signal-card.bearish {
            border-left-color: #f44336;
            background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(244, 67, 54, 0.05) 100%);
        }

        .signal-card.neutral {
            border-left-color: #ff9800;
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 152, 0, 0.05) 100%);
        }

        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .signal-type {
            font-size: 18px;
            font-weight: 600;
        }

        .confidence-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }

        .confidence-high {
            background: rgba(76, 175, 80, 0.2);
            color: #4CAF50;
            border: 1px solid #4CAF50;
        }

        .confidence-medium {
            background: rgba(255, 152, 0, 0.2);
            color: #ff9800;
            border: 1px solid #ff9800;
        }

        .signal-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .detail-item {
            display: flex;
            flex-direction: column;
        }

        .detail-label {
            font-size: 12px;
            color: #8892b0;
            margin-bottom: 4px;
        }

        .detail-value {
            font-size: 16px;
            font-weight: 600;
        }

        .alert-section {
            margin-top: 25px;
        }

        .alert-item {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .alert-high {
            background: rgba(244, 67, 54, 0.1);
            border: 1px solid rgba(244, 67, 54, 0.3);
        }

        .alert-medium {
            background: rgba(255, 152, 0, 0.1);
            border: 1px solid rgba(255, 152, 0, 0.3);
        }

        .alert-low {
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid rgba(76, 175, 80, 0.3);
        }

        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #e6f1ff;
        }

        .chart-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #e6f1ff;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .stat-card {
            background: linear-gradient(135deg, #2a3266 0%, #323870 100%);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid #3b4472;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 12px;
            color: #8892b0;
        }

        .auto-refresh {
            background: linear-gradient(45deg, #2196F3, #1976D2);
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }

        .footer-info {
            text-align: center;
            padding: 20px;
            color: #8892b0;
            font-size: 12px;
            border-top: 1px solid #3b4472;
            margin-top: 20px;
        }

        @media (max-width: 1200px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        '''

    def generate_dashboard(self, output_file):
        """Generate complete HTML dashboard"""
        print(f"🔨 Generating HTML dashboard: {output_file}")

        # Generate charts
        flow_chart, cumulative_chart = self.analyzer.create_charts()

        # Get current data
        signals = self.analyzer.signals
        alerts = self.analyzer.alerts
        stats = self.analyzer.get_statistics()
        latest_timestamp = self.analyzer.get_latest_display_timestamp()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Generate HTML template
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Money Flow Trading Dashboard - {datetime.now().strftime('%Y-%m-%d')}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        {self.get_css_styles()}
    </style>
    <script>
        // Auto-refresh page every 30 seconds
        setTimeout(function() {{
            window.location.reload();
        }}, 30000);

        console.log("Money Flow Dashboard loaded at {current_time}");
        console.log("Auto-refresh in 30 seconds");
    </script>
</head>
<body>
    <div class="dashboard-header">
        <div class="header-content">
            <div class="logo-section">
                <div class="logo-icon">
                    <i class="fas fa-chart-line fa-lg" style="color: white;"></i>
                </div>
                <div class="title-section">
                    <h1>Money Flow Trading Dashboard</h1>
                    <div style="font-size: 14px; color: #8892b0;">Real-time Institutional Money Flow Analysis</div>
                </div>
            </div>
            <div class="status-section">
                <div class="live-status">
                    <div class="pulse"></div>
                    <span>Live - {latest_timestamp}</span>
                </div>
                <div class="auto-refresh">
                    <i class="fas fa-sync-alt"></i> Auto-Refresh
                </div>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="signals-panel">
            <div class="section-title">
                <i class="fas fa-signal"></i> Trading Signals
            </div>

            <div class="signal-card {signals.get('action_color', 'neutral')}">
                <div class="signal-header">
                    <div class="signal-type">{signals.get('major_signal', 'NO SIGNAL')}</div>
                    <div class="confidence-badge {'confidence-high' if signals.get('confidence', 0) >= 90 else 'confidence-medium'}">
                        {signals.get('confidence', 0)}%
                    </div>
                </div>
                <div class="signal-details">
                    <div class="detail-item">
                        <div class="detail-label">Expected Move</div>
                        <div class="detail-value">{signals.get('expected_move', '--')}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Lead Time</div>
                        <div class="detail-value">{signals.get('lead_time', '--')}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Flow (M)</div>
                        <div class="detail-value" style="color: {'#4CAF50' if signals.get('flow_m', 0) > 0 else '#f44336'}">
                            {'+' if signals.get('flow_m', 0) > 0 else ''}{signals.get('flow_m', 0):.2f}
                        </div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Cumulative (M)</div>
                        <div class="detail-value" style="color: {'#4CAF50' if signals.get('cumulative_m', 0) > 0 else '#f44336'}">
                            {'+' if signals.get('cumulative_m', 0) > 0 else ''}{signals.get('cumulative_m', 0):.2f}
                        </div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #8892b0;">
                    Data as of: {signals.get('timestamp', 'No data')}
                </div>
            </div>

            <div class="alert-section">
                <div class="section-title">
                    <i class="fas fa-exclamation-triangle"></i> Alerts
                </div>
                {self.generate_alerts_html(alerts)}
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats['total_data_points']}</div>
                    <div class="stat-label">Total Signals</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['accuracy']}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['significant_flows']}</div>
                    <div class="stat-label">Significant Flows</div>
                </div>
            </div>
        </div>

        <div class="charts-panel">
            <div class="chart-container">
                <div class="chart-title">
                    <i class="fas fa-chart-bar"></i> Money Flow Analysis
                </div>
                {flow_chart}
            </div>

            <div class="chart-container">
                <div class="chart-title">
                    <i class="fas fa-chart-line"></i> Cumulative Flow Trend
                </div>
                {cumulative_chart}
            </div>
        </div>
    </div>

    <div class="footer-info">
        <div>Generated at: {current_time} | Auto-refresh: 30 seconds | 
        Data Source: Money Flow Analysis | 
        Accuracy: 85-95% | Lead Time: 15-30 minutes</div>
    </div>
</body>
</html>'''

        # Write to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Dashboard generated: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error writing HTML file: {e}")
            return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Money Flow Dashboard Generator')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--output', help='Output HTML file (default: keymoney_YYYYMMDD.html)')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output:
        date_str = datetime.now().strftime('%Y%m%d')
        args.output = f'keymoney_{date_str}.html'

    print("🚀 Money Flow Dashboard Generator")
    print("=" * 50)

    # Initialize analyzer
    analyzer = MoneyFlowAnalyzer()
    html_generator = HTMLGenerator(analyzer)

    def generate_dashboard():
        """Generate dashboard once"""
        print(f"\n🔄 Processing {args.csv}...")

        if analyzer.load_data(args.csv):
            analyzer.analyze_signals()
            if html_generator.generate_dashboard(args.output):
                print(f"📊 Dashboard updated: {args.output}")
                print(f"🎯 Latest signal: {analyzer.signals.get('major_signal', 'No signal')}")
                print(f"⏰ Data timestamp: {analyzer.get_latest_timestamp()}")
                print(f"💰 Flow: {analyzer.signals.get('flow_m', 0):.2f}M")
                return True
        else:
            print("❌ Failed to load data")
            return False

    if args.continuous:
        print(f"🔄 Starting continuous mode (refresh every {args.interval} seconds)")
        print(f"📁 CSV file: {args.csv}")
        print(f"📄 Output file: {args.output}")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                success = generate_dashboard()
                if success:
                    print(f"⏳ Waiting {args.interval} seconds...\n")
                else:
                    print(f"⚠️ Error occurred, retrying in {args.interval} seconds...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
    else:
        generate_dashboard()
        print(f"\n✅ Single generation complete")
        print(f"💡 Use --continuous flag for live updates")

if __name__ == "__main__":
    main()



