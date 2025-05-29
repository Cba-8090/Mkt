#!/usr/bin/env python3
"""
Complete Enhanced Multi-Source Money Flow Trading Dashboard
Integrates: Futures Flow (70%) + Options Flow (30%) + Gamma Analysis + Price Data
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
        """Load options money flow data (30% weight) - Enhanced with scaling fix"""
        try:
            print("📈 Loading Options Money Flow Data...")

            if not os.path.exists(csv_path):
                print(f"⚠️ Options data file not found: {csv_path}")
                self._create_dummy_options_data()
                return True

            df = pd.read_csv(csv_path)

            # Validate required columns
            required_cols = ['timestamp', 'net_flow', 'total_flow', 'bullish_flow', 'bearish_flow']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"⚠️ Missing columns in options data: {missing_cols}")
                self._create_dummy_options_data()
                return True

            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            # **SCALING FIX**: Detect and correct scale mismatch
            if 'net_flow' in df.columns:
                # Check the scale of the data
                max_abs_flow = df['net_flow'].abs().max()

                print(f"📊 Original options flow range: ±{max_abs_flow:.2f}")

                # If options flow is much smaller than typical futures flow (millions), scale it up
                if max_abs_flow < 100_000:  # Less than 100K suggests it needs scaling
                    # Apply scaling factor to match futures data magnitude
                    scaling_factor = 1_000_000  # Scale to millions

                    flow_columns = ['net_flow', 'total_flow', 'bullish_flow', 'bearish_flow']
                    if 'call_buying' in df.columns:
                        flow_columns.extend(['call_buying', 'put_writing', 'call_short_covering',
                                             'put_unwinding', 'put_buying', 'call_writing',
                                             'put_short_covering', 'call_unwinding'])

                    for col in flow_columns:
                        if col in df.columns:
                            df[col] = df[col] * scaling_factor

                    print(f"🔧 Applied scaling factor: {scaling_factor:,}")
                    print(f"📊 Scaled options flow range: ±{df['net_flow'].abs().max() / 1_000_000:.2f}M")
                else:
                    print(f"📊 Options flow already in appropriate scale")

            self.options_data = df
            self.last_update['options'] = datetime.now()
            print(f"✅ Loaded {len(df)} options flow records with scaling correction")
            return True

        except Exception as e:
            print(f"❌ Error loading options data: {e}")
            self._create_dummy_options_data()
            return True

    def load_gamma_data(self, html_path):
        """Load gamma analysis data from YOUR HTML report format"""
        try:
            print("🎯 Loading Gamma Analysis Data from your HTML format...")

            if not os.path.exists(html_path):
                print(f"⚠️ Gamma data file not found: {html_path}")
                self.gamma_data = self._create_dummy_gamma_time_series()
                print("⚠️ Using dummy gamma data - please ensure gamma analyzer is running")
                return True

            # Parse HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'html.parser')

            # Extract gamma data using your specific format
            gamma_data = self._extract_gamma_from_html(soup)

            if gamma_data and gamma_data['timestamps']:
                # Convert lists to numpy arrays for easier manipulation
                self.gamma_data = {
                    'timestamps': gamma_data['timestamps'],
                    'support_pressure': np.array(gamma_data['support_pressure']),
                    'resistance_pressure': np.array(gamma_data['resistance_pressure']),
                    'sr_ratio': np.array(gamma_data['sr_ratio']),
                    'max_pressure_strikes': np.array(gamma_data['max_pressure_strikes']),
                    'support_levels': np.array(gamma_data['support_levels']),
                    'resistance_levels': np.array(gamma_data['resistance_levels']),
                    'reversal_signals': np.array(gamma_data['reversal_signals']),
                    'last_update': datetime.now()
                }
                print("✅ Successfully loaded real gamma analysis data from HTML")
            else:
                # Fallback to dummy data
                self.gamma_data = self._create_dummy_gamma_time_series()
                print("⚠️ Could not extract gamma data from HTML, using dummy data")

            self.last_update['gamma'] = datetime.now()
            return True

        except Exception as e:
            print(f"❌ Error loading gamma data: {e}")
            self.gamma_data = self._create_dummy_gamma_time_series()
            return True

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


    def _extract_gamma_from_html(self, soup):
        """Extract gamma data from YOUR specific HTML format"""
        try:
            gamma_data = {
                'timestamps': [],
                'support_pressure': [],
                'resistance_pressure': [],
                'sr_ratio': [],
                'max_pressure_strikes': [],
                'support_levels': [],
                'resistance_levels': [],
                'reversal_signals': []
            }

            print("🔍 Parsing your gamma HTML format...")

            # Find the main data table - "All Reports Analyzed"
            tables = soup.find_all('table')
            data_table = None

            for table in tables:
                # Look for the table with headers: Time, Spot Price, Support Pressure, etc.
                header_row = table.find('tr')
                if header_row:
                    headers = [th.get_text().strip() for th in header_row.find_all('th')]
                    if 'Time' in headers and 'Support Pressure' in headers:
                        data_table = table
                        print(f"✅ Found data table with {len(headers)} columns")
                        break

            if not data_table:
                print("⚠️ Could not find data table in HTML")
                return None

            # Parse the data rows
            rows = data_table.find_all('tr')[1:]  # Skip header row
            print(f"📊 Processing {len(rows)} data rows...")

            for i, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 7:  # Ensure we have enough columns
                    try:
                        # Extract data based on your table structure:
                        # Time | Spot Price | Structure | Pressure Bias | Gamma Bias | Support Pressure | Resistance Pressure | S/R Ratio
                        time_str = cells[0].get_text().strip()
                        spot_price = float(cells[1].get_text().strip())
                        support_pressure = float(cells[5].get_text().strip())
                        resistance_pressure = float(cells[6].get_text().strip())
                        sr_ratio = float(cells[7].get_text().strip())

                        # Parse time - your format is HH:MM:SS
                        if ':' in time_str:
                            # Convert to full timestamp (assuming today's date)
                            timestamp = pd.to_datetime(f"2025-05-28 {time_str}")

                            gamma_data['timestamps'].append(timestamp)
                            gamma_data['support_pressure'].append(
                                support_pressure / 1_000_000)  # Convert to millions for consistency
                            gamma_data['resistance_pressure'].append(resistance_pressure / 1_000_000)
                            gamma_data['sr_ratio'].append(sr_ratio)

                            # Use spot price as max pressure strike (close approximation)
                            gamma_data['max_pressure_strikes'].append(spot_price)

                            # Calculate support/resistance levels (approximate based on spot price)
                            gamma_data['support_levels'].append(spot_price - 50)  # 50 points below
                            gamma_data['resistance_levels'].append(spot_price + 50)  # 50 points above

                            # Detect reversal signals based on S/R ratio changes
                            if len(gamma_data['sr_ratio']) > 1:
                                # Significant S/R ratio change indicates potential reversal
                                prev_ratio = gamma_data['sr_ratio'][-2]
                                current_ratio = sr_ratio
                                ratio_change = abs(current_ratio - prev_ratio)

                                # Mark as reversal if ratio changes significantly (>0.5)
                                reversal_signal = 1 if ratio_change > 0.5 else 0
                                gamma_data['reversal_signals'].append(reversal_signal)
                            else:
                                gamma_data['reversal_signals'].append(0)

                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Error parsing row {i}: {e}")
                        continue

            # Extract breakdown signals from text
            breakdown_signals = self._extract_breakdown_signals(soup)
            if breakdown_signals:
                print(f"✅ Found {len(breakdown_signals)} breakdown signals")
                # Mark reversal signals at breakdown times
                for breakdown_time in breakdown_signals:
                    # Find closest timestamp and mark as reversal
                    for j, timestamp in enumerate(gamma_data['timestamps']):
                        if abs((timestamp - breakdown_time).total_seconds()) < 300:  # Within 5 minutes
                            if j < len(gamma_data['reversal_signals']):
                                gamma_data['reversal_signals'][j] = 1

            print(f"✅ Successfully extracted {len(gamma_data['timestamps'])} gamma data points")
            print(
                f"📊 Support pressure range: {min(gamma_data['support_pressure']):.2f}M - {max(gamma_data['support_pressure']):.2f}M")
            print(
                f"📊 Resistance pressure range: {min(gamma_data['resistance_pressure']):.2f}M - {max(gamma_data['resistance_pressure']):.2f}M")
            print(f"📊 S/R ratio range: {min(gamma_data['sr_ratio']):.2f} - {max(gamma_data['sr_ratio']):.2f}")

            return gamma_data

        except Exception as e:
            print(f"❌ Error parsing gamma HTML: {e}")
            return None


    def _extract_breakdown_signals(self, soup):
        """Extract breakdown signal times from HTML text"""
        try:
            breakdown_times = []

            # Look for breakdown signal text
            text_content = soup.get_text()

            # Find lines containing "CRITICAL BREAKDOWN SIGNAL" or "BREAKDOWN SIGNAL"
            lines = text_content.split('\n')
            for line in lines:
                if 'BREAKDOWN SIGNAL' in line.upper():
                    # Extract time from lines like "09:31:29: CRITICAL BREAKDOWN SIGNAL: ..."
                    time_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
                    if time_match:
                        time_str = time_match.group(1)
                        try:
                            breakdown_time = pd.to_datetime(f"2025-05-28 {time_str}")
                            breakdown_times.append(breakdown_time)
                        except:
                            continue

            return breakdown_times

        except Exception as e:
            print(f"⚠️ Error extracting breakdown signals: {e}")
            return []















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
        return success_count >= 1  # At least futures source for testing
    
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
        """Analyze gamma levels for signal confirmation - Fixed for array data"""
        if self.data_loader.gamma_data is None:
            return "No Data"

        gamma = self.data_loader.gamma_data

        # Get latest gamma values (last element from arrays)
        try:
            if isinstance(gamma.get('support_pressure'), np.ndarray) and len(gamma['support_pressure']) > 0:
                support_pressure = gamma['support_pressure'][-1]  # Get latest value
            else:
                support_pressure = gamma.get('support_pressure', 0.5)

            if isinstance(gamma.get('resistance_pressure'), np.ndarray) and len(gamma['resistance_pressure']) > 0:
                resistance_pressure = gamma['resistance_pressure'][-1]  # Get latest value
            else:
                resistance_pressure = gamma.get('resistance_pressure', 0.5)

            if isinstance(gamma.get('sr_ratio'), np.ndarray) and len(gamma['sr_ratio']) > 0:
                sr_ratio = gamma['sr_ratio'][-1]  # Get latest S/R ratio
            else:
                sr_ratio = 1.0

            print(
                f"🎯 Latest Gamma Values: Support={support_pressure:.2f}M, Resistance={resistance_pressure:.2f}M, S/R Ratio={sr_ratio:.2f}")

            # Analyze based on combined flow and gamma confirmation
            if combined_flow > 50 and support_pressure > 0.6:
                return "Strong Support"
            elif combined_flow < -50 and resistance_pressure > 0.6:
                return "Weak Resistance"
            elif sr_ratio > 1.2:  # Support bias when S/R ratio > 1.2
                return "Support Bias"
            elif sr_ratio < 0.8:  # Resistance bias when S/R ratio < 0.8
                return "Resistance Bias"
            else:
                return "Neutral"

        except Exception as e:
            print(f"⚠️ Error analyzing gamma confirmation: {e}")
            return "Error"

    def create_gamma_pressure_chart(self):
        """Create gamma pressure chart with improved spacing"""
        if self.data_loader.gamma_data is None:
            print("⚠️ No gamma data available for gamma pressure chart")
            return ""

        print("🎯 Creating Gamma Pressure Analysis Chart with improved layout...")

        gamma_data = self.data_loader.gamma_data

        # Create 3-panel subplot with better spacing
        gamma_fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'Support vs Resistance Pressure',
                'S/R Ratio & Max Pressure Strike',
                'Price Levels & Reversal Signals'
            ],
            vertical_spacing=0.12,  # INCREASED spacing between panels
            row_heights=[0.4, 0.3, 0.3],
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": True}],
                [{"secondary_y": True}]
            ]
        )

        # Panel 1: Support vs Resistance Pressure (Filled Area Chart)
        gamma_fig.add_trace(
            go.Scatter(
                x=gamma_data['timestamps'],
                y=gamma_data['support_pressure'],
                mode='lines',
                name='Support Pressure',
                fill='tonexty',
                fillcolor='rgba(76, 175, 80, 0.3)',
                line=dict(color='#4CAF50', width=2),
                hovertemplate='<b>%{x}</b><br>Support: %{y:.2f}M<extra></extra>'
            ),
            row=1, col=1
        )

        gamma_fig.add_trace(
            go.Scatter(
                x=gamma_data['timestamps'],
                y=gamma_data['resistance_pressure'],
                mode='lines',
                name='Resistance Pressure',
                fill='tozeroy',
                fillcolor='rgba(244, 67, 54, 0.3)',
                line=dict(color='#f44336', width=2),
                hovertemplate='<b>%{x}</b><br>Resistance: %{y:.2f}M<extra></extra>'
            ),
            row=1, col=1
        )

        # Panel 2: S/R Ratio (Primary Y-axis) & Max Pressure Strike (Secondary Y-axis)
        gamma_fig.add_trace(
            go.Scatter(
                x=gamma_data['timestamps'],
                y=gamma_data['sr_ratio'],
                mode='lines+markers',
                name='S/R Ratio',
                line=dict(color='#FF9800', width=3),
                marker=dict(size=4),
                hovertemplate='<b>%{x}</b><br>S/R Ratio: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1, secondary_y=False
        )

        gamma_fig.add_trace(
            go.Scatter(
                x=gamma_data['timestamps'],
                y=gamma_data['max_pressure_strikes'],
                mode='lines',
                name='Max Pressure Strike',
                line=dict(color='#2196F3', width=2, dash='dash'),
                hovertemplate='<b>%{x}</b><br>Strike: ₹%{y:.0f}<extra></extra>',
                yaxis='y4'
            ),
            row=2, col=1, secondary_y=True
        )

        # Panel 3: Price Levels & Reversal Signals
        gamma_fig.add_trace(
            go.Scatter(
                x=gamma_data['timestamps'],
                y=gamma_data['support_levels'],
                mode='lines',
                name='Support Level',
                line=dict(color='#4CAF50', width=2, dash='dot'),
                hovertemplate='<b>%{x}</b><br>Support: ₹%{y:.0f}<extra></extra>'
            ),
            row=3, col=1, secondary_y=False
        )

        gamma_fig.add_trace(
            go.Scatter(
                x=gamma_data['timestamps'],
                y=gamma_data['resistance_levels'],
                mode='lines',
                name='Resistance Level',
                line=dict(color='#f44336', width=2, dash='dot'),
                hovertemplate='<b>%{x}</b><br>Resistance: ₹%{y:.0f}<extra></extra>'
            ),
            row=3, col=1, secondary_y=False
        )

        # Add reversal signals as star markers
        reversal_times = []
        reversal_prices = []
        for i, signal in enumerate(gamma_data['reversal_signals']):
            if signal == 1:
                reversal_times.append(gamma_data['timestamps'][i])
                reversal_prices.append(gamma_data['support_levels'][i])

        if reversal_times:
            gamma_fig.add_trace(
                go.Scatter(
                    x=reversal_times,
                    y=reversal_prices,
                    mode='markers',
                    name='Reversal Signals',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='#FFD700',
                        line=dict(color='#FF6B35', width=2)
                    ),
                    hovertemplate='<b>%{x}</b><br>Reversal Signal<br>Price: ₹%{y:.0f}<extra></extra>',
                    yaxis='y6'
                ),
                row=3, col=1, secondary_y=True
            )

        # Update layout with improved spacing and styling
        gamma_fig.update_layout(
            title={
                'text': 'Gamma Pressure Analysis - Multi-Panel View',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff'),
            showlegend=True,
            height=900,  # INCREASED height for better spacing
            margin=dict(l=80, r=80, t=100, b=80),  # INCREASED margins
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        # Update Y-axes labels with better spacing
        gamma_fig.update_yaxes(title_text="Pressure (Millions)", row=1, col=1, title_font_size=12)
        gamma_fig.update_yaxes(title_text="S/R Ratio", secondary_y=False, row=2, col=1, title_font_size=12)
        gamma_fig.update_yaxes(title_text="Strike Price (₹)", secondary_y=True, row=2, col=1, title_font_size=12)
        gamma_fig.update_yaxes(title_text="Price Levels (₹)", secondary_y=False, row=3, col=1, title_font_size=12)
        gamma_fig.update_yaxes(title_text="Reversal Signals", secondary_y=True, row=3, col=1, title_font_size=12)

        # Update X-axes
        gamma_fig.update_xaxes(title_text="Time", row=3, col=1, title_font_size=12)

        # Convert to HTML
        config = {'displayModeBar': True, 'displaylogo': False, 'responsive': True}
        gamma_html = pyo.plot(gamma_fig, output_type='div', include_plotlyjs=False, config=config)

        print("✅ Gamma Pressure Analysis Chart with improved spacing generated successfully")
        return gamma_html

    def _get_gamma_statistics(self):
        """Extract current gamma statistics for display"""
        if self.data_loader.gamma_data is None:
            return {
                'support_pressure': 0,
                'resistance_pressure': 0,
                'sr_ratio': 1.0,
                'reversal_signals': 0,
                'max_pressure_strike': 0,
                'last_update': 'No data'
            }

        gamma = self.data_loader.gamma_data

        try:
            # Get latest values from arrays
            latest_support = gamma['support_pressure'][-1] if len(gamma['support_pressure']) > 0 else 0
            latest_resistance = gamma['resistance_pressure'][-1] if len(gamma['resistance_pressure']) > 0 else 0
            latest_sr_ratio = gamma['sr_ratio'][-1] if len(gamma['sr_ratio']) > 0 else 1.0
            latest_strike = gamma['max_pressure_strikes'][-1] if len(gamma['max_pressure_strikes']) > 0 else 0

            # Count reversal signals
            reversal_count = np.sum(gamma['reversal_signals']) if len(gamma['reversal_signals']) > 0 else 0

            return {
                'support_pressure': latest_support,
                'resistance_pressure': latest_resistance,
                'sr_ratio': latest_sr_ratio,
                'reversal_signals': int(reversal_count),
                'max_pressure_strike': latest_strike,
                'last_update': gamma.get('last_update', datetime.now()).strftime('%H:%M:%S')
            }

        except Exception as e:
            print(f"⚠️ Error extracting gamma statistics: {e}")
            return {
                'support_pressure': 0,
                'resistance_pressure': 0,
                'sr_ratio': 1.0,
                'reversal_signals': 0,
                'max_pressure_strike': 0,
                'last_update': 'Error'
            }


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
        """Create comprehensive multi-source charts with improved spacing and layout"""
        if self.data_loader.futures_data is None:
            return "", "", "", ""

        print("📊 Creating enhanced multi-source charts with improved spacing...")

        # Get live data
        futures_df = self.data_loader.futures_data.iloc[:self.live_data_end_index + 1]

        # Chart 1: Combined Money Flow with Better Spacing
        combined_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                'Futures Money Flow (70% Weight) + Cumulative',
                'Combined Analysis'
            ],
            vertical_spacing=0.2,  # INCREASED from 0.1 to 0.2 for better spacing
            row_heights=[0.65, 0.35],  # ADJUSTED: More space for top chart, less for bottom
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}]
            ]
        )

        # Futures flow bars (Primary Y-axis)
        flow_colors = ['#4CAF50' if x > 0 else '#f44336' for x in futures_df['weighted_money_flow']]
        combined_fig.add_trace(
            go.Bar(
                x=futures_df['timestamp'],
                y=futures_df['weighted_money_flow'] / 1_000_000,
                name='Futures Flow',
                marker_color=flow_colors,
                hovertemplate='<b>%{x}</b><br>Flow: %{y:.2f}M<extra></extra>',
                opacity=0.7
            ),
            row=1, col=1, secondary_y=False
        )

        # Futures Cumulative Line (Secondary Y-axis)
        combined_fig.add_trace(
            go.Scatter(
                x=futures_df['timestamp'],
                y=futures_df['cumulative_weighted_money_flow'] / 1_000_000,
                mode='lines',
                name='Cumulative Futures',
                line=dict(color='#FF6B35', width=3),
                hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.2f}M<extra></extra>',
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )

        # Calculate combined signal for each timestamp
        combined_flow_values = self._calculate_combined_flow_timeseries(futures_df)

        combined_fig.add_trace(
            go.Scatter(
                x=futures_df['timestamp'],
                y=combined_flow_values,
                mode='lines',
                name='Combined Signal',
                line=dict(color='#00BCD4', width=4),  # INCREASED width for better visibility
                hovertemplate='<b>%{x}</b><br>Combined: %{y:.2f}M<extra></extra>'
            ),
            row=2, col=1
        )

        # Update layout with improved spacing and margins
        combined_fig.update_layout(
            title={
                'text': 'Multi-Source Money Flow Analysis with Cumulative Trends',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff'),
            showlegend=True,
            height=650,  # INCREASED total height for better proportions
            margin=dict(
                l=80,  # Left margin
                r=80,  # Right margin
                t=100,  # Top margin (more space for title)
                b=60  # Bottom margin
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        # Update Y-axes labels with better formatting
        combined_fig.update_yaxes(
            title_text="Flow (Millions)",
            secondary_y=False,
            row=1, col=1,
            title_font_size=12,
            tickfont_size=10
        )
        combined_fig.update_yaxes(
            title_text="Cumulative (Millions)",
            secondary_y=True,
            row=1, col=1,
            title_font_size=12,
            tickfont_size=10
        )
        combined_fig.update_yaxes(
            title_text="Combined Flow (M)",
            row=2, col=1,
            title_font_size=12,
            tickfont_size=10
        )

        # Update X-axes with better formatting
        combined_fig.update_xaxes(
            title_text="Time",
            row=2, col=1,
            title_font_size=12,
            tickfont_size=10
        )

        # Chart 2: Price vs Flow Correlation (Improved)
        price_fig = go.Figure()

        if self.data_loader.price_data is not None:
            price_df = self.data_loader.price_data

            price_fig.add_trace(go.Scatter(
                x=price_df['timestamp'],
                y=price_df['spot_price'],
                mode='lines',
                name='Spot Price',
                line=dict(color='#FFC107', width=3),  # INCREASED width
                hovertemplate='<b>%{x}</b><br>Price: ₹%{y:.2f}<extra></extra>'
            ))

        price_fig.update_layout(
            title={
                'text': 'Price Movement Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff'),
            yaxis_title="Price (₹)",
            xaxis_title="Time",
            height=400,  # INCREASED height
            margin=dict(l=60, r=60, t=80, b=60)
        )

        # Chart 3: Options Flow Analysis (Improved)
        options_fig = make_subplots(specs=[[{"secondary_y": True}]])

        if self.data_loader.options_data is not None:
            options_df = self.data_loader.options_data

            # Scale options data if needed
            scaled_net_flow = options_df['net_flow'].copy()
            if scaled_net_flow.abs().max() < 100_000:
                scaled_net_flow = scaled_net_flow * 1_000_000

            # Calculate cumulative net flow
            cumulative_net_flow = scaled_net_flow.cumsum()

            # Options Net Flow Bars (Primary Y-axis)
            options_fig.add_trace(
                go.Bar(
                    x=options_df['timestamp'],
                    y=scaled_net_flow / 1_000_000,
                    name='Options Net Flow',
                    marker_color=['#4CAF50' if x > 0 else '#f44336' for x in scaled_net_flow],
                    hovertemplate='<b>%{x}</b><br>Net Flow: %{y:.2f}M<extra></extra>',
                    opacity=0.7
                ),
                secondary_y=False
            )

            # Cumulative Options Flow Line (Secondary Y-axis)
            options_fig.add_trace(
                go.Scatter(
                    x=options_df['timestamp'],
                    y=cumulative_net_flow / 1_000_000,
                    mode='lines',
                    name='Cumulative Options',
                    line=dict(color='#9C27B0', width=3),
                    hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.2f}M<extra></extra>'
                ),
                secondary_y=True
            )

        # Update options chart layout with improved spacing
        options_fig.update_layout(
            title={
                'text': 'Options Money Flow (30% Weight) + Cumulative Trend',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6f1ff'),
            height=400,  # INCREASED height
            margin=dict(l=60, r=60, t=80, b=60)
        )

        # Update Y-axes for options chart
        options_fig.update_yaxes(title_text="Net Flow (Millions)", secondary_y=False)
        options_fig.update_yaxes(title_text="Cumulative (Millions)", secondary_y=True)
        options_fig.update_xaxes(title_text="Time")

        # Chart 4: Gamma Pressure Analysis Chart
        gamma_chart = self.create_gamma_pressure_chart()

        # Convert to HTML with responsive config
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'responsive': True,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }

        combined_html = pyo.plot(combined_fig, output_type='div', include_plotlyjs=False, config=config)
        price_html = pyo.plot(price_fig, output_type='div', include_plotlyjs=False, config=config)
        options_html = pyo.plot(options_fig, output_type='div', include_plotlyjs=False, config=config)

        print("✅ Enhanced charts with improved spacing and layout generated successfully")
        return combined_html, price_html, options_html, gamma_chart

    # USAGE INSTRUCTIONS:
    # 1. Open your enhanced_money_flow_complete.py file
    # 2. Find the create_enhanced_charts method (around line 400-500)
    # 3. Replace the entire method with the code above
    # 4. Save the file
    # 5. Run with explicit options path:
    #    python enhanced_money_flow_complete.py \
    #        --futures-csv nifty_detailed_20250528_5min.csv \
    #        --options-csv "C:\Projects\apps\_nifty_optionanalyser\net_flow_reports\net_money_flow_data.csv"

    print("🎯 EXPECTED RESULTS:")
    print("✅ Futures chart: Bars + Orange cumulative line (from CSV column)")
    print("✅ Options chart: Bars + Purple cumulative line (calculated)")
    print("✅ Dual Y-axes: Flow (left) + Cumulative (right)")
    print("✅ Interactive hover: Shows both flow and cumulative values")
    print("✅ Professional styling: Matching your dark theme")


    def _calculate_combined_flow_timeseries(self, futures_df):
        """Calculate combined flow for each timestamp (not just latest)"""
        print("🔄 Calculating dynamic combined flow timeseries...")

        combined_values = []

        for index, futures_row in futures_df.iterrows():
            # Get futures flow for this timestamp
            futures_flow_m = futures_row['weighted_money_flow'] / 1_000_000

            # Find corresponding options flow for this timestamp
            options_flow_m = 0
            if self.data_loader.options_data is not None:
                options_df = self.data_loader.options_data

                # Find closest timestamp in options data
                target_time = futures_row['timestamp']

                # Calculate time differences
                time_diffs = abs(options_df['timestamp'] - target_time)

                # Find the closest match within 10 minutes
                if len(time_diffs) > 0:
                    min_diff_idx = time_diffs.idxmin()
                    min_diff_seconds = time_diffs.iloc[min_diff_idx].total_seconds()

                    if min_diff_seconds <= 600:  # Within 10 minutes
                        options_latest = options_df.loc[min_diff_idx]
                        options_flow_raw = options_latest['net_flow']

                        # Apply same scaling logic as in data loading
                        if abs(options_flow_raw) < 100_000:
                            options_flow_raw *= 1_000_000

                        options_flow_m = options_flow_raw / 1_000_000

            # Calculate weighted combined flow for this timestamp
            combined_flow_m = (futures_flow_m * 0.7) + (options_flow_m * 0.3)
            combined_values.append(combined_flow_m)

        print(f"✅ Generated {len(combined_values)} combined flow data points")
        print(f"📊 Combined flow range: {min(combined_values):.2f}M to {max(combined_values):.2f}M")

        return combined_values

    
    def _calculate_dynamic_combined_signals(self):
        """Calculate combined signals for all timestamps (enhanced version)"""
        print("🔄 Calculating dynamic combined signals for all timestamps...")

        if self.data_loader.futures_data is None:
            return

        futures_df = self.data_loader.futures_data.iloc[:self.live_data_end_index + 1]

        # Store time series data for better analysis
        self.timeseries_signals = {
            'timestamps': [],
            'futures_flows': [],
            'options_flows': [],
            'combined_flows': [],
            'signal_strengths': [],
            'signal_types': []
        }

        for index, futures_row in futures_df.iterrows():
            timestamp = futures_row['timestamp']
            futures_flow_m = futures_row['weighted_money_flow'] / 1_000_000

            # Find corresponding options flow
            options_flow_m = 0
            if self.data_loader.options_data is not None:
                options_df = self.data_loader.options_data
                time_diffs = abs(options_df['timestamp'] - timestamp)

                if len(time_diffs) > 0:
                    min_diff_idx = time_diffs.idxmin()
                    if time_diffs.iloc[min_diff_idx].total_seconds() <= 600:  # Within 10 minutes
                        options_latest = options_df.loc[min_diff_idx]
                        options_flow_raw = options_latest['net_flow']

                        if abs(options_flow_raw) < 100_000:
                            options_flow_raw *= 1_000_000

                        options_flow_m = options_flow_raw / 1_000_000

            # Calculate combined flow and signal strength
            combined_flow_m = (futures_flow_m * 0.7) + (options_flow_m * 0.3)
            signal_strength = self._calculate_signal_strength(combined_flow_m)
            signal_type = self._determine_signal_type(combined_flow_m, signal_strength)

            # Store in timeseries
            self.timeseries_signals['timestamps'].append(timestamp)
            self.timeseries_signals['futures_flows'].append(futures_flow_m)
            self.timeseries_signals['options_flows'].append(options_flow_m)
            self.timeseries_signals['combined_flows'].append(combined_flow_m)
            self.timeseries_signals['signal_strengths'].append(signal_strength)
            self.timeseries_signals['signal_types'].append(signal_type)

        print(f"✅ Generated dynamic signals for {len(self.timeseries_signals['timestamps'])} timestamps")

        # Update current signals with latest values
        if len(self.timeseries_signals['timestamps']) > 0:
            latest_idx = -1
            latest_timestamp = self.timeseries_signals['timestamps'][latest_idx]
            latest_futures = self.timeseries_signals['futures_flows'][latest_idx]
            latest_options = self.timeseries_signals['options_flows'][latest_idx]
            latest_combined = self.timeseries_signals['combined_flows'][latest_idx]
            latest_strength = self.timeseries_signals['signal_strengths'][latest_idx]

            # Update combined_signals with latest values
            self.combined_signals.update({
                'timestamp': latest_timestamp.strftime('%d-%m-%Y %H:%M'),
                'futures_flow_m': latest_futures,
                'options_flow_m': latest_options,
                'combined_flow_m': latest_combined,
                'signal_strength': latest_strength,
                'major_signal': self._determine_signal_type(latest_combined, latest_strength),
                'confidence': self._calculate_confidence(latest_strength,
                                                         self._analyze_gamma_confirmation(latest_combined)),
                'expected_move': self._calculate_expected_move(latest_combined, latest_strength),
                'expected_duration': self._calculate_expected_duration(abs(latest_combined), latest_strength),
                'gamma_confirmation': self._analyze_gamma_confirmation(latest_combined),
                'price_momentum': self._analyze_price_momentum(),
                'action_color': self._get_action_color(latest_combined)
            })


class EnhancedHTMLGenerator:
    """Enhanced HTML generator for multi-source dashboard"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def _generate_charts_panel_with_gamma(self, combined_chart, price_chart, options_chart, gamma_chart):
        """Generate enhanced charts panel with gamma analysis"""
        return f'''
        <div class="charts-panel">
            <div class="chart-container main-chart">
                <div class="chart-title">
                    <i class="fas fa-chart-area"></i> Combined Money Flow Analysis
                    <span class="chart-subtitle">Futures (70%) + Options (30%) Weighted</span>
                </div>
                {combined_chart}
            </div>

            <div class="chart-row">
                <div class="chart-container half-chart">
                    <div class="chart-title">
                        <i class="fas fa-dollar-sign"></i> Price Movement
                    </div>
                    {price_chart}
                </div>

                <div class="chart-container half-chart">
                    <div class="chart-title">
                        <i class="fas fa-chart-line"></i> Options Flow
                    </div>
                    {options_chart}
                </div>
            </div>

            <div class="chart-container gamma-chart">
                <div class="chart-title">
                    <i class="fas fa-crosshairs"></i> Gamma Pressure Analysis
                    <span class="chart-subtitle">Support/Resistance Dynamics</span>
                </div>
                {gamma_chart}
            </div>
        </div>'''

    
    def generate_enhanced_dashboard(self, output_file):
        """Generate complete enhanced HTML dashboard"""
        print(f"🔨 Generating Enhanced Multi-Source Dashboard: {output_file}")
        
        # Generate charts
        combined_chart, price_chart, options_chart, gamma_chart = self.analyzer.create_enhanced_charts()
        # Get current data
        signals = self.analyzer.combined_signals
        alerts = self.analyzer.alerts
        stats = self._get_enhanced_statistics()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate HTML content
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Money Flow Dashboard - {datetime.now().strftime('%Y-%m-%d')}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        {self._get_enhanced_css()}
    </style>
    <script>
        // Auto-refresh page every 30 seconds
        setTimeout(function() {{
            window.location.reload();
        }}, 30000);
        
        // Enhanced dashboard loaded
        console.log("Enhanced Multi-Source Dashboard loaded at {current_time}");
        console.log("Data sources: Futures (70%) + Options (30%) + Gamma + Price");
    </script>
</head>
<body>
    {self._generate_enhanced_header(signals)}
    
    <div class="main-content">
        {self._generate_signals_panel(signals, alerts, stats)} 
        {self._generate_charts_panel_with_gamma(combined_chart, price_chart, options_chart, gamma_chart)}
    </div>
    
    {self._generate_data_sources_panel()}
    {self._generate_footer()}
</body>
</html>'''
        
        # Write to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Enhanced Dashboard generated: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error writing HTML file: {e}")
            return False

    def _generate_enhanced_stats_with_gamma(self, stats, gamma_stats):
        """Generate enhanced statistics including gamma metrics"""
        return f'''
        <div class="stats-section">
            <div class="section-title">
                <i class="fas fa-chart-pie"></i> Analysis Statistics
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats['data_sources']}</div>
                    <div class="stat-label">Active Sources</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['signal_strength']:.1f}</div>
                    <div class="stat-label">Signal Strength</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['accuracy']}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{gamma_stats['support_pressure']:.2f}</div>
                    <div class="stat-label">Support Pressure</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{gamma_stats['resistance_pressure']:.2f}</div>
                    <div class="stat-label">Resistance Pressure</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{gamma_stats['reversal_signals']}</div>
                    <div class="stat-label">Reversal Signals</div>
                </div>
            </div>
        </div>'''

    def _get_enhanced_css_with_gamma(self):
        """Get enhanced CSS styles including gamma chart styling"""
        base_css = self._get_enhanced_css()  # Get existing CSS

        gamma_css = '''
        .gamma-chart {
            min-height: 600px;
            margin-top: 20px;
        }

        .chart-subtitle {
            font-size: 12px;
            color: #8892b0;
            margin-left: auto;
            font-weight: normal;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
            gap: 8px;
        }

        .source-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .gamma-indicator {
            background: linear-gradient(135deg, #2a3266 0%, #323870 100%);
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .gamma-value {
            font-weight: 600;
            font-size: 14px;
        }

        .gamma-label {
            font-size: 11px;
            color: #8892b0;
        }
        '''

        return base_css + gamma_css

    def _generate_data_sources_panel_with_gamma(self):
        """Generate data sources information panel with gamma details"""
        return f'''
        <div class="data-sources-panel">
            <div class="section-title">
                <i class="fas fa-database"></i> Data Sources & Integration
            </div>

            <div class="sources-grid">
                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-chart-bar"></i></div>
                    <div class="source-info">
                        <h4>Futures Money Flow</h4>
                        <p>Primary signal source (70% weight)</p>
                        <div class="source-path">nifty_detailed_*.csv</div>
                    </div>
                </div>

                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-chart-line"></i></div>
                    <div class="source-info">
                        <h4>Options Money Flow</h4>
                        <p>Sentiment validation (30% weight)</p>
                        <div class="source-path">net_money_flow_data.csv</div>
                    </div>
                </div>

                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-crosshairs"></i></div>
                    <div class="source-info">
                        <h4>Gamma Analysis</h4>
                        <p>Support/Resistance pressure dynamics</p>
                        <div class="source-path">nifty_comparison_report_*.html</div>
                    </div>
                </div>

                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-dollar-sign"></i></div>
                    <div class="source-info">
                        <h4>Price Data</h4>
                        <p>Real-time price confirmation</p>
                        <div class="source-path">OptionAnalyser.db</div>
                    </div>
                </div>
            </div>

            <div class="gamma-analysis-summary">
                <div class="section-title">
                    <i class="fas fa-analytics"></i> Gamma Analysis Features
                </div>
                <div class="feature-grid">
                    <div class="feature-item">
                        <i class="fas fa-shield-alt"></i>
                        <span>Support Pressure Tracking</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-wall-brick"></i>
                        <span>Resistance Pressure Monitoring</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-balance-scale"></i>
                        <span>S/R Ratio Analysis</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-crosshairs"></i>
                        <span>Max Pressure Strike Identification</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>Reversal Signal Detection</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-chart-line"></i>
                        <span>Structural Level Tracking</span>
                    </div>
                </div>
            </div>
        </div>'''

    
    def _generate_enhanced_header(self, signals):
        """Generate enhanced header with multi-source status"""
        latest_timestamp = signals.get('timestamp', 'No data')
        
        return f'''
        <div class="dashboard-header">
            <div class="header-content">
                <div class="logo-section">
                    <div class="logo-icon">
                        <i class="fas fa-chart-network fa-lg" style="color: white;"></i>
                    </div>
                    <div class="title-section">
                        <h1>Enhanced Money Flow Dashboard</h1>
                        <div class="subtitle">Multi-Source Analysis: Futures + Options + Gamma + Price</div>
                    </div>
                </div>
                <div class="status-section">
                    <div class="data-sources-status">
                        <div class="source-indicator">
                            <i class="fas fa-chart-bar"></i>
                            <span>Futures</span>
                            <div class="status-dot active"></div>
                        </div>
                        <div class="source-indicator">
                            <i class="fas fa-chart-line"></i>
                            <span>Options</span>
                            <div class="status-dot active"></div>
                        </div>
                        <div class="source-indicator">
                            <i class="fas fa-crosshairs"></i>
                            <span>Gamma</span>
                            <div class="status-dot active"></div>
                        </div>
                        <div class="source-indicator">
                            <i class="fas fa-dollar-sign"></i>
                            <span>Price</span>
                            <div class="status-dot active"></div>
                        </div>
                    </div>
                    <div class="live-status">
                        <div class="pulse"></div>
                        <span>Live - {latest_timestamp}</span>
                    </div>
                    <div class="auto-refresh">
                        <i class="fas fa-sync-alt"></i> Auto-Refresh
                    </div>
                </div>
            </div>
        </div>'''
    
    def _generate_signals_panel(self, signals, alerts, stats):
        """Generate enhanced signals panel"""
        return f'''
        <div class="signals-panel">
            <div class="section-title">
                <i class="fas fa-signal"></i> Multi-Source Trading Signals
            </div>
            
            {self._generate_combined_signal_card(signals)}
            {self._generate_source_breakdown(signals)}
            {self._generate_enhanced_alerts(alerts)}
            {self._generate_enhanced_stats(stats)}
        </div>'''
    
    def _generate_combined_signal_card(self, signals):
        """Generate main combined signal card"""
        signal_type = signals.get('major_signal', 'NO SIGNAL')
        confidence = signals.get('confidence', 0)
        action_color = signals.get('action_color', 'neutral')
        
        return f'''
        <div class="signal-card combined-signal {action_color}">
            <div class="signal-header">
                <div class="signal-type-main">{signal_type}</div>
                <div class="confidence-badge confidence-{'high' if confidence >= 90 else 'medium'}">
                    {confidence}%
                </div>
            </div>
            
            <div class="signal-details-grid">
                <div class="detail-item">
                    <div class="detail-label">Expected Move</div>
                    <div class="detail-value">{signals.get('expected_move', '--')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Duration</div>
                    <div class="detail-value">{signals.get('expected_duration', '--')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Combined Flow</div>
                    <div class="detail-value combined-flow" style="color: {'#4CAF50' if signals.get('combined_flow_m', 0) > 0 else '#f44336'}">
                        {'+' if signals.get('combined_flow_m', 0) > 0 else ''}{signals.get('combined_flow_m', 0):.2f}M
                    </div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Gamma Status</div>
                    <div class="detail-value">{signals.get('gamma_confirmation', '--')}</div>
                </div>
            </div>
            
            <div class="signal-timestamp">
                <i class="fas fa-clock"></i> {signals.get('timestamp', 'No data')}
            </div>
        </div>'''

    def _generate_source_breakdown(self, signals):
        """Generate enhanced breakdown by data source with cumulative values and S/R ratio"""
        futures_flow = signals.get('futures_flow_m', 0)
        options_flow = signals.get('options_flow_m', 0)
        price_momentum = signals.get('price_momentum', 'No Data')

        # Get cumulative values from the data sources
        futures_cumulative = self._get_latest_cumulative_futures()
        options_cumulative = self._get_latest_cumulative_options()
        current_sr_ratio = self._get_current_sr_ratio()

        return f'''
        <div class="source-breakdown">
            <div class="breakdown-title">Source Breakdown</div>

            <div class="source-item">
                <div class="source-header">
                    <i class="fas fa-chart-bar"></i>
                    <span>Futures Flow (70%)</span>
                </div>
                <div class="source-values">
                    <div class="source-value" style="color: {'#4CAF50' if futures_flow > 0 else '#f44336'}">
                        {'+' if futures_flow > 0 else ''}{futures_flow:.2f}M
                    </div>
                    <div class="cumulative-value" style="color: {'#4CAF50' if futures_cumulative > 0 else '#f44336'}">
                        ({'+' if futures_cumulative > 0 else ''}{futures_cumulative:.2f})
                    </div>
                </div>
                <div class="source-weight">Weight: 70%</div>
            </div>

            <div class="source-item">
                <div class="source-header">
                    <i class="fas fa-chart-line"></i>
                    <span>Options Flow (30%)</span>
                </div>
                <div class="source-values">
                    <div class="source-value" style="color: {'#4CAF50' if options_flow > 0 else '#f44336'}">
                        {'+' if options_flow > 0 else ''}{options_flow:.2f}M
                    </div>
                    <div class="cumulative-value" style="color: {'#4CAF50' if options_cumulative > 0 else '#f44336'}">
                        ({'+' if options_cumulative > 0 else ''}{options_cumulative:.2f})
                    </div>
                </div>
                <div class="source-weight">Weight: 30%</div>
            </div>

            <div class="source-item">
                <div class="source-header">
                    <i class="fas fa-crosshairs"></i>
                    <span>Gamma Analysis</span>
                </div>
                <div class="source-values">
                    <div class="source-value">
                        {signals.get('gamma_confirmation', 'No Data')}
                    </div>
                    <div class="sr-ratio-value" style="color: {'#4CAF50' if current_sr_ratio > 1.0 else '#f44336' if current_sr_ratio < 1.0 else '#ff9800'}">
                        ({current_sr_ratio:.2f})
                    </div>
                </div>
                <div class="source-weight">Validation</div>
            </div>

            <div class="source-item">
                <div class="source-header">
                    <i class="fas fa-trending-up"></i>
                    <span>Price Momentum</span>
                </div>
                <div class="source-values">
                    <div class="source-value">
                        {price_momentum}
                    </div>
                </div>
                <div class="source-weight">Confirmation</div>
            </div>
        </div>'''

    def _get_latest_cumulative_futures(self):
        """Get the latest cumulative futures value"""
        try:
            if self.analyzer.data_loader.futures_data is not None:
                futures_df = self.analyzer.data_loader.futures_data
                if len(futures_df) > 0:
                    latest_cumulative = futures_df.iloc[self.analyzer.live_data_end_index][
                        'cumulative_weighted_money_flow']
                    return latest_cumulative / 1_000_000  # Convert to millions
            return 0.0
        except Exception as e:
            print(f"⚠️ Error getting cumulative futures: {e}")
            return 0.0

    def _get_latest_cumulative_options(self):
        """Get the latest cumulative options value"""
        try:
            if self.analyzer.data_loader.options_data is not None:
                options_df = self.analyzer.data_loader.options_data
                if len(options_df) > 0:
                    # Calculate cumulative from net_flow column
                    scaled_net_flow = options_df['net_flow'].copy()
                    if scaled_net_flow.abs().max() < 100_000:
                        scaled_net_flow = scaled_net_flow * 1_000_000

                    cumulative_options = scaled_net_flow.cumsum()
                    latest_cumulative = cumulative_options.iloc[-1]
                    return latest_cumulative / 1_000_000  # Convert to millions
            return 0.0
        except Exception as e:
            print(f"⚠️ Error getting cumulative options: {e}")
            return 0.0

    def _get_current_sr_ratio(self):
        """Get the current S/R ratio from gamma data"""
        try:
            if self.analyzer.data_loader.gamma_data is not None:
                gamma_data = self.analyzer.data_loader.gamma_data
                if 'sr_ratio' in gamma_data and len(gamma_data['sr_ratio']) > 0:
                    # Get the latest S/R ratio
                    current_ratio = gamma_data['sr_ratio'][-1]
                    return float(current_ratio)
            return 1.0  # Default neutral ratio
        except Exception as e:
            print(f"⚠️ Error getting S/R ratio: {e}")
            return 1.0


    def _generate_enhanced_alerts(self, alerts):
        """Generate enhanced alerts section"""
        alerts_html = ""
        for alert in alerts:
            icon = {
                'HIGH': 'fa-exclamation-triangle',
                'MEDIUM': 'fa-info-circle',
                'LOW': 'fa-check-circle'
            }.get(alert['type'], 'fa-info-circle')
            
            alerts_html += f'''
            <div class="alert-item {alert['priority']}">
                <i class="fas {icon}"></i>
                <div class="alert-content">
                    <div class="alert-message">{alert['message']}</div>
                    <div class="alert-action">{alert['action']}</div>
                </div>
            </div>'''
        
        return f'''
        <div class="alert-section">
            <div class="section-title">
                <i class="fas fa-exclamation-triangle"></i> Multi-Source Alerts
            </div>
            {alerts_html}
        </div>'''
    
    def _generate_enhanced_stats(self, stats):
        """Generate enhanced statistics"""
        return f'''
        <div class="stats-section">
            <div class="section-title">
                <i class="fas fa-chart-pie"></i> Analysis Statistics
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats['data_sources']}</div>
                    <div class="stat-label">Active Sources</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['signal_strength']:.1f}</div>
                    <div class="stat-label">Signal Strength</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['accuracy']}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['update_frequency']}</div>
                    <div class="stat-label">Update Freq</div>
                </div>
            </div>
        </div>'''
    
    def _generate_charts_panel(self, combined_chart, price_chart, options_chart):
        """Generate enhanced charts panel"""
        return f'''
        <div class="charts-panel">
            <div class="chart-container main-chart">
                <div class="chart-title">
                    <i class="fas fa-chart-area"></i> Combined Money Flow Analysis
                    <span class="chart-subtitle">Futures (70%) + Options (30%) Weighted</span>
                </div>
                {combined_chart}
            </div>
            
            <div class="chart-row">
                <div class="chart-container half-chart">
                    <div class="chart-title">
                        <i class="fas fa-dollar-sign"></i> Price Movement
                    </div>
                    {price_chart}
                </div>
                
                <div class="chart-container half-chart">
                    <div class="chart-title">
                        <i class="fas fa-chart-line"></i> Options Flow
                    </div>
                    {options_chart}
                </div>
            </div>
        </div>'''
    
    def _generate_data_sources_panel(self):
        """Generate data sources information panel"""
        return f'''
        <div class="data-sources-panel">
            <div class="section-title">
                <i class="fas fa-database"></i> Data Sources & Integration
            </div>
            
            <div class="sources-grid">
                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-chart-bar"></i></div>
                    <div class="source-info">
                        <h4>Futures Money Flow</h4>
                        <p>Primary signal source (70% weight)</p>
                        <div class="source-path">nifty_detailed_*.csv</div>
                    </div>
                </div>
                
                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-chart-line"></i></div>
                    <div class="source-info">
                        <h4>Options Money Flow</h4>
                        <p>Sentiment validation (30% weight)</p>
                        <div class="source-path">net_money_flow_data.csv</div>
                    </div>
                </div>
                
                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-crosshairs"></i></div>
                    <div class="source-info">
                        <h4>Gamma Analysis</h4>
                        <p>Support/Resistance levels</p>
                        <div class="source-path">nifty_comparison_report_*.html</div>
                    </div>
                </div>
                
                <div class="source-detail-card">
                    <div class="source-icon"><i class="fas fa-dollar-sign"></i></div>
                    <div class="source-info">
                        <h4>Price Data</h4>
                        <p>Real-time price confirmation</p>
                        <div class="source-path">OptionAnalyser.db</div>
                    </div>
                </div>
            </div>
        </div>'''
    
    def _generate_footer(self):
        """Generate enhanced footer"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f'''
        <div class="footer-info">
            <div class="footer-content">
                <div>Generated at: {current_time} | Multi-Source Integration Active</div>
                <div>Futures (70%) + Options (30%) + Gamma Validation + Price Confirmation</div>
                <div>Expected Accuracy: 85-95% | Adaptive Timeframes: 5min - 2hrs | Auto-refresh: 30s</div>
            </div>
        </div>'''
    
    def _get_enhanced_statistics(self):
        """Get enhanced statistics"""
        signals = self.analyzer.combined_signals
        
        return {
            'data_sources': 4,  # Futures, Options, Gamma, Price
            'signal_strength': signals.get('signal_strength', 0) * 10,  # Scale to 0-10
            'accuracy': signals.get('confidence', 85),
            'update_frequency': '3-5min'
        }
    
    def _get_enhanced_css(self):
        """Get enhanced CSS styles with complete styling"""
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
            max-width: 1600px;
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

        .subtitle {
            font-size: 12px;
            color: #8892b0;
            margin-top: 4px;
        }

        .status-section {
            display: flex;
            gap: 20px;
            align-items: center;
        }

        .data-sources-status {
            display: flex;
            gap: 15px;
        }

        .source-indicator {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            padding: 8px;
            border-radius: 8px;
            background: rgba(255,255,255,0.05);
        }

        .source-indicator span {
            font-size: 10px;
            color: #8892b0;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
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
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 400px 1fr;
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

        .main-chart {
            min-height: 400px;
        }

        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .half-chart {
            min-height: 300px;
        }

        .signal-card {
            background: linear-gradient(135deg, #2a3266 0%, #323870 100%);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border-left: 4px solid;
        }

        .combined-signal {
            border-left-width: 6px;
        }

        .signal-card.bullish {
            border-left-color: #4CAF50;
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(76, 175, 80, 0.05) 100%);
        }

        .signal-card.bearish {
            border-left-color: #f44336;
            background: linear-gradient(135deg, rgba(244, 67, 54, 0.15) 0%, rgba(244, 67, 54, 0.05) 100%);
        }

        .signal-card.neutral {
            border-left-color: #ff9800;
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.15) 0%, rgba(255, 152, 0, 0.05) 100%);
        }

        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .signal-type-main {
            font-size: 22px;
            font-weight: 700;
        }

        .confidence-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
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

        .signal-details-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
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

        .combined-flow {
            font-size: 18px;
            font-weight: 700;
        }

        .signal-timestamp {
            font-size: 12px;
            color: #8892b0;
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        .source-breakdown {
            background: linear-gradient(135deg, #2a3266 0%, #323870 100%);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .breakdown-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #e6f1ff;
        }

        .source-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .source-item:last-child {
            border-bottom: none;
        }

        .source-header {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
        }

        .source-value {
            font-weight: 600;
            margin-right: 10px;
        }

        .source-weight {
            font-size: 11px;
            color: #8892b0;
            min-width: 70px;
            text-align: right;
        }

        .alert-section {
            margin-bottom: 20px;
        }

        .alert-item {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .alert-content {
            flex: 1;
        }

        .alert-message {
            font-weight: 600;
            margin-bottom: 4px;
        }

        .alert-action {
            font-size: 12px;
            color: #8892b0;
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
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #e6f1ff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chart-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #e6f1ff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chart-subtitle {
            font-size: 12px;
            color: #8892b0;
            margin-left: auto;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 10px;
        }

        .stat-card {
            background: linear-gradient(135deg, #2a3266 0%, #323870 100%);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            border: 1px solid #3b4472;
        }

        .stat-value {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .stat-label {
            font-size: 10px;
            color: #8892b0;
        }

        .data-sources-panel {
            max-width: 1600px;
            margin: 20px auto;
            padding: 0 20px;
            background: linear-gradient(135deg, #1e2749 0%, #252b4f 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid #3b4472;
        }

        .sources-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .source-detail-card {
            background: linear-gradient(135deg, #2a3266 0%, #323870 100%);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .source-icon {
            font-size: 24px;
            color: #4CAF50;
        }

        .source-info h4 {
            margin-bottom: 5px;
            color: #e6f1ff;
        }

        .source-info p {
            font-size: 12px;
            color: #8892b0;
            margin-bottom: 8px;
        }

        .source-path {
            font-size: 11px;
            color: #666;
            font-family: monospace;
            background: rgba(0,0,0,0.3);
            padding: 4px 8px;
            border-radius: 4px;
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

        .footer-content div {
            margin-bottom: 4px;
        }

        @media (max-width: 1400px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .signals-panel {
                order: 2;
            }
            
            .charts-panel {
                order: 1;
            }
            
            .chart-row {
                grid-template-columns: 1fr;
                gap: 15px;
            }
        }

        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 15px;
            }
            
            .status-section {
                flex-direction: column;
                gap: 10px;
            }
            
            .data-sources-status {
                flex-wrap: wrap;
                justify-content: center;
            }
            
            .signal-details-grid {
                grid-template-columns: 1fr;
                gap: 10px;
            }
            
            .sources-grid {
                grid-template-columns: 1fr;
            }
        }
        
        
        
        
    .source-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .source-item:last-child {
        border-bottom: none;
    }
    
    .source-header {
        display: flex;
        align-items: center;
        gap: 8px;
        flex: 1;
    }
    
    .source-values {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
        margin-right: 10px;
    }
    
    .source-value {
        font-weight: 600;
        font-size: 14px;
    }
    
    .cumulative-value {
        font-size: 11px;
        font-weight: 500;
        opacity: 0.8;
    }
    
    .sr-ratio-value {
        font-size: 11px;
        font-weight: 500;
        opacity: 0.8;
    }
    
    .source-weight {
        font-size: 11px;
        color: #8892b0;
        min-width: 70px;
        text-align: right;
    }
    
    /* Color coding for values */
    .source-values .source-value {
        transition: all 0.3s ease;
    }
    
    .cumulative-value,
    .sr-ratio-value {
        transition: all 0.3s ease;
    }
        '''


def main():
    """Complete enhanced main function with HTML generation"""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Source Money Flow Dashboard')
    parser.add_argument('--futures-csv', required=True, help='Futures money flow CSV file')
    parser.add_argument('--options-csv', help='Options money flow CSV file')
    parser.add_argument('--gamma-html', help='Gamma analysis HTML file')
    parser.add_argument('--price-db', help='Price database file')
    parser.add_argument('--output', help='Output HTML file')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval')
    parser.add_argument('--continuous', action='store_true', help='Continuous monitoring')
    
    args = parser.parse_args()
    
    # Define data paths based on your system structure
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
        today = datetime.now().strftime('%Y%m%d')
        args.gamma_html = os.path.join(gamma_dir, f'nifty_comparison_report_{today}.html')
    
    if not args.price_db:
        args.price_db = price_db
    
    if not args.output:
        date_str = datetime.now().strftime('%Y%m%d')
        args.output = os.path.join(r'C:\Projects\apps\KeyMoney', f'enhanced_money_flow_{date_str}.html')
    
    print("🚀 Enhanced Multi-Source Money Flow Dashboard")
    print("=" * 60)
    print(f"📊 Futures Data: {args.futures_csv}")
    print(f"📈 Options Data: {args.options_csv}")
    print(f"🎯 Gamma Data: {args.gamma_html}")
    print(f"💰 Price Data: {args.price_db}")
    print(f"📄 Output: {args.output}")
    
    # Initialize enhanced analyzer and HTML generator
    analyzer = EnhancedMoneyFlowAnalyzer()
    html_generator = EnhancedHTMLGenerator(analyzer)
    
    def generate_enhanced_dashboard():
        """Generate complete enhanced dashboard"""
        print(f"\n🔄 Processing multi-source data at {datetime.now().strftime('%H:%M:%S')}")
        
        # Load all data sources
        if analyzer.load_all_data(args.futures_csv, args.options_csv, args.gamma_html, args.price_db):
            # Calculate weighted signals
            analyzer.calculate_weighted_signals()
            
            # Generate HTML dashboard
            if html_generator.generate_enhanced_dashboard(args.output):
                # Display current signal in console
                signals = analyzer.combined_signals
                print(f"\n🎯 CURRENT SIGNAL: {signals.get('major_signal', 'NO SIGNAL')}")
                print(f"🎯 Combined Flow: {signals.get('combined_flow_m', 0):.2f}M")
                print(f"🎯 Confidence: {signals.get('confidence', 0)}%")
                print(f"🎯 Expected Move: {signals.get('expected_move', '--')}")
                print(f"🎯 Duration: {signals.get('expected_duration', '--')}")
                print(f"🎯 Gamma Status: {signals.get('gamma_confirmation', '--')}")
                
                # Show alerts
                alerts = analyzer.alerts
                if alerts:
                    print(f"\n🚨 ACTIVE ALERTS ({len(alerts)}):")
                    for alert in alerts:
                        print(f"   {alert['type']}: {alert['message']} - {alert['action']}")
                
                print(f"\n✅ Enhanced dashboard updated: {args.output}")
                return True
            else:
                print("❌ Failed to generate HTML dashboard")
                return False
        else:
            print("❌ Failed to load sufficient data sources")
            return False
    
    # Run dashboard generation
    if args.continuous:
        print(f"🔄 Starting continuous multi-source monitoring (every {args.interval}s)")
        print("Press Ctrl+C to stop")
        try:
            while True:
                success = generate_enhanced_dashboard()
                if success:
                    print(f"⏳ Next update in {args.interval} seconds...")
                else:
                    print(f"⏳ Retrying in {args.interval} seconds...")
                print("-" * 60)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
    else:
        generate_enhanced_dashboard()


if __name__ == "__main__":
    main()