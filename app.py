# app.py - Complete Money Flow Trading Dashboard with Session Persistence
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'money_flow_dashboard_secret_key_2025_v2'  # Required for sessions

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class MoneyFlowAnalyzer:
    def __init__(self):
        self.signals = {}
        self.alerts = []
        self.df = None
        self.live_data_end_index = 0
        self.is_data_loaded = False
        self.current_file_path = None

    def load_data(self, file_path):
        """Load and validate CSV data with session persistence"""
        try:
            print(f"📊 Loading data from: {file_path}")

            # Read CSV file
            self.df = pd.read_csv(file_path)
            self.current_file_path = file_path

            print(f"📈 Data shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")

            # Store file path in session for persistence
            session['current_file_path'] = file_path
            session['data_loaded'] = True

            # Required columns validation
            required_columns = [
                'timestamp', 'weighted_money_flow', 'cumulative_weighted_money_flow',
                'weighted_positive_money_flow', 'weighted_negative_money_flow'
            ]

            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
                self.is_data_loaded = False
                session['data_loaded'] = False
                return False, f"Missing required columns: {missing_columns}"

            # Convert timestamp to datetime if it's string
            if self.df['timestamp'].dtype == 'object':
                print("🕒 Converting timestamp to datetime...")
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')

            # Check for invalid timestamps
            invalid_timestamps = self.df['timestamp'].isna().sum()
            if invalid_timestamps > 0:
                print(f"⚠️ Found {invalid_timestamps} invalid timestamps")

            # Find the last row with actual data
            self._find_live_data_end()

            # Mark data as successfully loaded
            self.is_data_loaded = True
            session['data_loaded'] = True
            session['live_data_end_index'] = self.live_data_end_index

            print(f"✅ Data loaded successfully!")
            print(f"📊 Live data periods: {self.live_data_end_index + 1}")

            return True, "Data loaded successfully"

        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            print(f"❌ {error_msg}")
            self.is_data_loaded = False
            session['data_loaded'] = False
            return False, error_msg

        except pd.errors.EmptyDataError:
            error_msg = "CSV file is empty"
            print(f"❌ {error_msg}")
            self.is_data_loaded = False
            session['data_loaded'] = False
            return False, error_msg

        except pd.errors.ParserError as e:
            error_msg = f"CSV parsing error: {str(e)}"
            print(f"❌ {error_msg}")
            self.is_data_loaded = False
            session['data_loaded'] = False
            return False, error_msg

        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            print(f"❌ {error_msg}")
            print("📋 Traceback:")
            traceback.print_exc()
            self.is_data_loaded = False
            session['data_loaded'] = False
            return False, error_msg

    def restore_from_session(self):
        """Restore analyzer state from session"""
        try:
            if session.get('data_loaded') and session.get('current_file_path'):
                file_path = session['current_file_path']

                print(f"🔄 Attempting to restore from session: {file_path}")

                # Check if file still exists
                if os.path.exists(file_path):
                    success, message = self.load_data(file_path)
                    if success:
                        print("✅ Session restore successful")
                        return True
                    else:
                        print(f"❌ Session restore failed: {message}")
                        session.clear()
                        return False
                else:
                    print(f"❌ Session file not found: {file_path}")
                    session.clear()
                    return False
            else:
                print("ℹ️ No session data to restore")
                return False

        except Exception as e:
            print(f"❌ Error restoring from session: {e}")
            session.clear()
            return False

    def _find_live_data_end(self):
        """Find the last row with actual trading data"""
        if self.df is None:
            return

        print("🔍 Finding live data endpoint...")

        # Strategy 1: Find last non-zero weighted_money_flow
        non_zero_flows = self.df[self.df['weighted_money_flow'] != 0]

        if len(non_zero_flows) > 0:
            self.live_data_end_index = non_zero_flows.index[-1]
            print(f"📊 Strategy 1: Found non-zero flows, end index: {self.live_data_end_index}")
        else:
            # Strategy 2: Find last non-zero weighted_raw_money_flow
            if 'weighted_raw_money_flow' in self.df.columns:
                non_zero_raw = self.df[self.df['weighted_raw_money_flow'] != 0]
                if len(non_zero_raw) > 0:
                    self.live_data_end_index = non_zero_raw.index[-1]
                    print(f"📊 Strategy 2: Found non-zero raw flows, end index: {self.live_data_end_index}")
                else:
                    # Strategy 3: Find last row where any money flow column is non-zero
                    money_flow_cols = ['weighted_positive_money_flow', 'weighted_negative_money_flow']
                    for col in money_flow_cols:
                        if col in self.df.columns:
                            non_zero_data = self.df[self.df[col] != 0]
                            if len(non_zero_data) > 0:
                                self.live_data_end_index = non_zero_data.index[-1]
                                print(f"📊 Strategy 3: Found non-zero {col}, end index: {self.live_data_end_index}")
                                break
                    else:
                        # Fallback: Use first few rows if no non-zero data found
                        self.live_data_end_index = min(5, len(self.df) - 1)
                        print(f"📊 Fallback: Using first few rows, end index: {self.live_data_end_index}")
            else:
                self.live_data_end_index = min(5, len(self.df) - 1)
                print(f"📊 Fallback: No raw flow column, end index: {self.live_data_end_index}")

        print(f"✅ Live data endpoint determined: index {self.live_data_end_index}")
        if self.live_data_end_index < len(self.df):
            timestamp = self.df.iloc[self.live_data_end_index]['timestamp']
            print(f"🕒 Last live timestamp: {timestamp}")

    def get_live_data(self):
        """Get only the portion of data that has been populated so far"""
        if not self.is_data_loaded or self.df is None:
            return None

        return self.df.iloc[:self.live_data_end_index + 1].copy()

    def analyze_signals(self):
        """Main signal analysis with proper error handling and session restoration"""
        try:
            print("🔬 Starting signal analysis...")

            # Try to restore from session if data not loaded
            if not self.is_data_loaded:
                print("⚠️ Data not loaded, attempting session restore...")
                if not self.restore_from_session():
                    print("❌ Session restore failed")
                    return {
                        'error': 'No data loaded',
                        'timestamp': 'No Data Available',
                        'major_signal': 'Please Upload CSV File',
                        'strength': 'NEUTRAL',
                        'confidence': 0,
                        'expected_move': 'Upload data to begin analysis',
                        'lead_time': 'N/A',
                        'action': 'UPLOAD FILE',
                        'flow_m': 0,
                        'cumulative_m': 0,
                        'flow_direction': 'NEUTRAL',
                        'trend': 'NEUTRAL',
                        'action_color': 'secondary',
                        'alerts': [{
                            'type': 'INFO',
                            'message': 'Session expired - Please upload CSV file again',
                            'action': 'Upload your nifty_detailed_YYYYMMDD_5min.csv file to restart analysis',
                            'priority': 'MEDIUM',
                            'color': 'info'
                        }],
                        'data_status': 'Session expired - Please upload CSV file'
                    }

            live_df = self.get_live_data()

            if live_df is None or len(live_df) < 3:
                print(f"⚠️ Insufficient data: {len(live_df) if live_df is not None else 0} periods")
                return {
                    'timestamp': 'Insufficient Data',
                    'major_signal': 'Waiting for More Data',
                    'strength': 'NEUTRAL',
                    'confidence': 0,
                    'expected_move': 'Need at least 3 data points',
                    'lead_time': 'N/A',
                    'action': 'WAIT',
                    'flow_m': 0,
                    'cumulative_m': 0,
                    'flow_direction': 'NEUTRAL',
                    'trend': 'NEUTRAL',
                    'action_color': 'secondary',
                    'alerts': [{
                        'type': 'WARNING',
                        'message': 'Insufficient data for analysis',
                        'action': 'Need at least 3 periods with non-zero money flow data',
                        'priority': 'MEDIUM',
                        'color': 'warning'
                    }],
                    'data_status': f'Live data available: {len(live_df) if live_df is not None else 0} periods'
                }

            latest = live_df.iloc[-1]
            prev = live_df.iloc[-2]

            print(f"📊 Analyzing latest data point: {latest['timestamp']}")

            # Convert to millions for analysis
            flow_m = latest['weighted_money_flow'] / 1_000_000
            cumulative_m = latest['cumulative_weighted_money_flow'] / 1_000_000
            prev_cumulative_m = prev['cumulative_weighted_money_flow'] / 1_000_000

            print(f"💰 Current flow: {flow_m:.2f}M, Cumulative: {cumulative_m:.2f}M")

            signals = {
                'timestamp': str(latest['timestamp']),
                'major_signal': 'No Signal',
                'strength': 'NEUTRAL',
                'confidence': 0,
                'expected_move': '0 points',
                'lead_time': '5-15 min',
                'action': 'WAIT',
                'flow_m': round(flow_m, 2),
                'cumulative_m': round(cumulative_m, 2),
                'flow_direction': 'POSITIVE' if flow_m > 0 else 'NEGATIVE',
                'trend': 'UP' if cumulative_m > prev_cumulative_m else 'DOWN',
                'action_color': 'secondary',
                'data_status': f'Live data: {len(live_df)} periods | Last update: {latest["timestamp"]}'
            }

            # Major Move Signals (Based on research findings)
            if abs(flow_m) >= 300:
                signals.update({
                    'major_signal': 'MAJOR RALLY INCOMING' if flow_m > 0 else 'MAJOR CRASH WARNING',
                    'strength': 'EXTREME',
                    'confidence': 95,
                    'expected_move': '150-200+ points UP' if flow_m > 0 else '200+ points DOWN',
                    'action': 'STRONG BUY' if flow_m > 0 else 'STRONG SELL',
                    'lead_time': '15-45 min',
                    'action_color': 'success' if flow_m > 0 else 'danger'
                })
                print(f"🚨 EXTREME signal detected: {signals['major_signal']}")

            elif abs(flow_m) >= 100:
                signals.update({
                    'major_signal': 'STRONG RALLY SIGNAL' if flow_m > 0 else 'STRONG DECLINE SIGNAL',
                    'strength': 'HIGH',
                    'confidence': 90,
                    'expected_move': '75-150 points UP' if flow_m > 0 else '100-200 points DOWN',
                    'action': 'BUY' if flow_m > 0 else 'SELL',
                    'lead_time': '10-30 min',
                    'action_color': 'success' if flow_m > 0 else 'danger'
                })
                print(f"🔥 HIGH signal detected: {signals['major_signal']}")

            elif abs(flow_m) >= 50:
                signals.update({
                    'major_signal': 'Moderate Rally Expected' if flow_m > 0 else 'Moderate Decline Expected',
                    'strength': 'MEDIUM',
                    'confidence': 85,
                    'expected_move': '30-75 points UP' if flow_m > 0 else '40-100 points DOWN',
                    'action': 'BUY SIGNAL' if flow_m > 0 else 'SELL SIGNAL',
                    'lead_time': '5-20 min',
                    'action_color': 'success' if flow_m > 0 else 'danger'
                })
                print(f"⚡ MEDIUM signal detected: {signals['major_signal']}")

            elif abs(flow_m) <= 25:
                signals.update({
                    'major_signal': 'Consolidation Phase',
                    'strength': 'LOW',
                    'confidence': 80,
                    'expected_move': '15-40 points range',
                    'action': 'RANGE TRADE',
                    'lead_time': '30-120 min',
                    'action_color': 'warning'
                })
                print(f"📊 LOW signal detected: {signals['major_signal']}")

            else:
                signals.update({
                    'major_signal': 'Moderate Activity',
                    'strength': 'MEDIUM',
                    'confidence': 75,
                    'expected_move': '25-50 points move',
                    'action': 'CAUTIOUS',
                    'lead_time': '10-30 min',
                    'action_color': 'info'
                })
                print(f"🔄 MODERATE signal detected: {signals['major_signal']}")

            # Generate alerts using live data
            self.alerts = self._generate_alerts(live_df, cumulative_m, prev_cumulative_m)
            signals['alerts'] = self.alerts

            self.signals = signals
            print("✅ Signal analysis completed successfully")
            return signals

        except Exception as e:
            print(f"❌ Error in signal analysis: {e}")
            traceback.print_exc()
            return {
                'error': 'Analysis error',
                'timestamp': 'Error',
                'major_signal': 'Analysis Error',
                'strength': 'NEUTRAL',
                'confidence': 0,
                'expected_move': 'Error in analysis',
                'lead_time': 'N/A',
                'action': 'CHECK DATA',
                'flow_m': 0,
                'cumulative_m': 0,
                'flow_direction': 'NEUTRAL',
                'trend': 'NEUTRAL',
                'action_color': 'danger',
                'alerts': [{
                    'type': 'ERROR',
                    'message': f'Analysis error: {str(e)}',
                    'action': 'Check data format and try again',
                    'priority': 'HIGH',
                    'color': 'danger'
                }],
                'data_status': f'Analysis error: {str(e)}'
            }

    def _generate_alerts(self, live_df, cumulative_m, prev_cumulative_m):
        """Generate alerts based on cumulative flow analysis"""
        alerts = []

        try:
            # Data freshness alert
            current_time = datetime.now()
            last_data_time = live_df.iloc[-1]['timestamp']

            if isinstance(last_data_time, str):
                last_data_time = pd.to_datetime(last_data_time)

            time_diff = (current_time - last_data_time).total_seconds() / 60  # minutes

            if time_diff > 15:  # More than 15 minutes old
                alerts.append({
                    'type': 'WARNING',
                    'message': f'Data is {time_diff:.0f} minutes old - May not reflect current market',
                    'action': 'CHECK DATA SOURCE - Upload fresh data',
                    'priority': 'HIGH',
                    'color': 'warning'
                })

            # Cumulative Flow Analysis
            if cumulative_m > 600:
                alerts.append({
                    'type': 'WARNING',
                    'message': 'Cumulative flow >600M - Potential peak formation',
                    'action': 'SELL ALERT - Consider profit taking',
                    'priority': 'HIGH',
                    'color': 'warning'
                })
            elif cumulative_m > 200 and (cumulative_m - prev_cumulative_m) > 0:
                alerts.append({
                    'type': 'BULLISH',
                    'message': 'Strong accumulation phase - Sustained rally expected',
                    'action': 'STRONG BUY - 30-60 min trend',
                    'priority': 'HIGH',
                    'color': 'success'
                })
            elif cumulative_m < -500:
                alerts.append({
                    'type': 'BEARISH',
                    'message': 'Major distribution - Trending decline expected',
                    'action': 'AVOID/SHORT - Multi-hour bearish trend',
                    'priority': 'HIGH',
                    'color': 'danger'
                })

            # Trend Reversal Detection
            if len(live_df) >= 5:
                recent_cumulative = live_df['cumulative_weighted_money_flow'].tail(5).values
                if self._detect_trend_reversal(recent_cumulative):
                    alerts.append({
                        'type': 'REVERSAL',
                        'message': 'Trend reversal detected in cumulative flow',
                        'action': 'REVERSAL TRADE - 15-45 min lead time',
                        'priority': 'MEDIUM',
                        'color': 'info'
                    })

            # Session timing analysis
            try:
                latest_timestamp = live_df.iloc[-1]['timestamp']
                if hasattr(latest_timestamp, 'hour'):
                    hour = latest_timestamp.hour
                else:
                    hour = int(str(latest_timestamp).split(' ')[1].split(':')[0])

                if hour == 9:
                    alerts.append({
                        'type': 'INFO',
                        'message': 'Opening hour - High impact flows set session direction',
                        'action': 'MOMENTUM FOLLOWING strategy recommended',
                        'priority': 'LOW',
                        'color': 'primary'
                    })
                elif hour >= 14:
                    alerts.append({
                        'type': 'INFO',
                        'message': 'Closing hour - High impact flows for session conclusion',
                        'action': 'REVERSAL/CONTINUATION signals expected',
                        'priority': 'LOW',
                        'color': 'primary'
                    })
            except:
                pass

        except Exception as e:
            print(f"⚠️ Error generating alerts: {e}")
            alerts.append({
                'type': 'ERROR',
                'message': f'Error generating alerts: {str(e)}',
                'action': 'Check data format',
                'priority': 'LOW',
                'color': 'warning'
            })

        return alerts

    def _detect_trend_reversal(self, values):
        """Detect trend reversal in cumulative flow"""
        try:
            if len(values) < 5:
                return False

            recent = values[-3:]
            earlier = values[:3]

            recent_trend = recent[-1] - recent[0]
            earlier_trend = earlier[-1] - earlier[0]

            return (recent_trend > 0 and earlier_trend < 0) or (recent_trend < 0 and earlier_trend > 0)
        except:
            return False

    def create_charts(self):
        """Create interactive charts with error handling"""
        try:
            if not self.is_data_loaded:
                print("⚠️ Charts: No data loaded")
                return None, None

            live_df = self.get_live_data()

            if live_df is None or len(live_df) == 0:
                print("⚠️ Charts: No live data available")
                return None, None

            print(f"📊 Creating charts for {len(live_df)} data points")

            # Prepare data - use live data or last 50 points, whichever is smaller
            df_plot = live_df.tail(50).copy()
            df_plot['flow_millions'] = df_plot['weighted_money_flow'] / 1_000_000
            df_plot['cumulative_millions'] = df_plot['cumulative_weighted_money_flow'] / 1_000_000

            # Money Flow Chart
            fig1 = go.Figure()

            colors = ['green' if x > 0 else 'red' for x in df_plot['flow_millions']]
            fig1.add_trace(go.Bar(
                x=df_plot['timestamp'],
                y=df_plot['flow_millions'],
                name='Money Flow',
                marker_color=colors,
                opacity=0.7
            ))

            fig1.add_hline(y=0, line_dash="dash", line_color="gray")
            fig1.update_layout(
                title=f"Money Flow Analysis - {len(live_df)} periods",
                xaxis_title="Time",
                yaxis_title="Flow (Millions)",
                height=400,
                showlegend=False
            )

            # Cumulative Flow Chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_plot['timestamp'],
                y=df_plot['cumulative_millions'],
                mode='lines',
                name='Cumulative Flow',
                line=dict(color='blue', width=3)
            ))

            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(
                title=f"Cumulative Money Flow - Last: {live_df.iloc[-1]['timestamp']}",
                xaxis_title="Time",
                yaxis_title="Cumulative Flow (Millions)",
                height=400,
                showlegend=False
            )

            print("✅ Charts created successfully")

            return (
                json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
                json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
            )

        except Exception as e:
            print(f"❌ Error creating charts: {e}")
            traceback.print_exc()
            return None, None

    def get_statistics(self):
        """Calculate session statistics with error handling"""
        try:
            if not self.is_data_loaded:
                return {
                    'total_positive': 0,
                    'total_negative': 0,
                    'net_flow': 0,
                    'max_inflow': 0,
                    'max_outflow': 0,
                    'avg_flow': 0,
                    'data_points': 0,
                    'live_periods': 0,
                    'data_coverage': '0%',
                    'status': 'No data loaded'
                }

            live_df = self.get_live_data()

            if live_df is None or len(live_df) == 0:
                return {
                    'total_positive': 0,
                    'total_negative': 0,
                    'net_flow': 0,
                    'max_inflow': 0,
                    'max_outflow': 0,
                    'avg_flow': 0,
                    'data_points': 0,
                    'live_periods': 0,
                    'data_coverage': '0%',
                    'status': 'No live data found'
                }

            positive_flows = live_df[live_df['weighted_money_flow'] > 0]['weighted_money_flow']
            negative_flows = live_df[live_df['weighted_money_flow'] < 0]['weighted_money_flow']

            total_possible_periods = len(self.df)
            live_periods = len(live_df)
            coverage_percent = (live_periods / total_possible_periods) * 100

            stats = {
                'total_positive': round(positive_flows.sum() / 1_000_000, 1),
                'total_negative': round(negative_flows.sum() / 1_000_000, 1),
                'net_flow': round((positive_flows.sum() + negative_flows.sum()) / 1_000_000, 1),
                'max_inflow': round(live_df['weighted_money_flow'].max() / 1_000_000, 1),
                'max_outflow': round(live_df['weighted_money_flow'].min() / 1_000_000, 1),
                'avg_flow': round(live_df['weighted_money_flow'].mean() / 1_000_000, 1),
                'data_points': live_periods,
                'live_periods': live_periods,
                'total_periods': total_possible_periods,
                'data_coverage': f'{coverage_percent:.1f}%',
                'session_progress': f'{live_periods}/{total_possible_periods} periods',
                'status': 'Data loaded and analyzed'
            }

            print(f"📊 Statistics calculated: {live_periods} live periods, {coverage_percent:.1f}% coverage")
            return stats

        except Exception as e:
            print(f"❌ Error calculating statistics: {e}")
            return {
                'total_positive': 0,
                'total_negative': 0,
                'net_flow': 0,
                'max_inflow': 0,
                'max_outflow': 0,
                'avg_flow': 0,
                'data_points': 0,
                'live_periods': 0,
                'data_coverage': '0%',
                'status': f'Error: {str(e)}'
            }


# Global analyzer instance
analyzer = MoneyFlowAnalyzer()


@app.route('/')
def index():
    """Main dashboard page"""
    print("🏠 Index page accessed")
    return render_template('dashboard.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis with comprehensive error handling"""
    try:
        print("\n" + "=" * 50)
        print("📁 UPLOAD REQUEST RECEIVED")
        print("=" * 50)

        # Log request details
        print(f"🔍 Request method: {request.method}")
        print(f"🔍 Content type: {request.content_type}")
        print(f"🔍 Files in request: {list(request.files.keys())}")

        # Check if file is in request
        if 'file' not in request.files:
            print("❌ No 'file' field in request")
            return jsonify({'success': False, 'message': 'No file uploaded'})

        file = request.files['file']
        print(f"📄 File object: {file}")
        print(f"📄 Filename: {file.filename}")
        print(f"📄 Content type: {file.content_type}")

        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'message': 'No file selected'})

        if not file.filename.lower().endswith('.csv'):
            print(f"❌ Invalid file extension: {file.filename}")
            return jsonify({'success': False, 'message': 'Please upload a CSV file'})

        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_{timestamp}.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            print(f"💾 Saving file to: {filepath}")
            print(f"📁 Upload directory exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}")

            # Save the file
            file.save(filepath)

            # Verify file was saved
            if not os.path.exists(filepath):
                print("❌ File save failed - file does not exist after save")
                return jsonify({'success': False, 'message': 'Failed to save uploaded file'})

            file_size = os.path.getsize(filepath)
            print(f"✅ File saved successfully: {file_size} bytes")

            # Load and analyze data
            print("🔄 Starting data analysis...")
            success, message = analyzer.load_data(filepath)

            if success:
                print("✅ Data loaded successfully, starting signal analysis...")
                signals = analyzer.analyze_signals()
                print("✅ Analysis completed successfully")

                return jsonify({
                    'success': True,
                    'message': 'File uploaded and analyzed successfully',
                    'redirect': url_for('analysis')
                })
            else:
                print(f"❌ Data loading failed: {message}")
                # Clean up failed file
                try:
                    os.remove(filepath)
                except:
                    pass
                return jsonify({'success': False, 'message': message})

        except Exception as file_error:
            print(f"❌ Error processing file: {file_error}")
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error processing file: {str(file_error)}'})

    except Exception as e:
        print(f"❌ Upload route error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Upload error: {str(e)}'})


@app.route('/analysis')
def analysis():
    """Analysis results page with session restoration"""
    try:
        print("\n" + "=" * 50)
        print("📊 ANALYSIS PAGE ACCESSED")
        print("=" * 50)

        # Check if data is loaded
        print(f"🔍 Data loaded status: {analyzer.is_data_loaded}")
        print(f"🔍 Session data loaded: {session.get('data_loaded', False)}")

        # Try to restore from session if data not loaded
        if not analyzer.is_data_loaded:
            print("⚠️ Analysis page: Data not loaded, attempting session restore...")
            if not analyzer.restore_from_session():
                print("❌ Analysis page: Session restore failed, redirecting to upload")
                return redirect(url_for('index'))

        # Generate analysis data
        print("🔄 Generating analysis data...")
        signals = analyzer.signals if analyzer.signals else analyzer.analyze_signals()
        charts = analyzer.create_charts()
        stats = analyzer.get_statistics()

        print("✅ Analysis page data prepared successfully")

        return render_template('analysis.html',
                               signals=signals,
                               charts=charts,
                               stats=stats)

    except Exception as e:
        print(f"❌ Analysis page error: {e}")
        traceback.print_exc()
        return redirect(url_for('index'))


@app.route('/api/refresh')
def refresh_analysis():
    """API endpoint to refresh analysis with session persistence"""
    try:
        print("\n" + "=" * 30)
        print("🔄 REFRESH API CALLED")
        print("=" * 30)

        print(f"🔍 Data loaded status: {analyzer.is_data_loaded}")
        print(f"🔍 Session data loaded: {session.get('data_loaded', False)}")

        # Try to restore from session if data not loaded
        if not analyzer.is_data_loaded:
            print("⚠️ Refresh: Data not loaded, attempting session restore...")
            if not analyzer.restore_from_session():
                print("❌ Refresh: Session restore failed")
                return jsonify({
                    'success': False,
                    'message': 'Session expired - Data needs to be re-uploaded',
                    'redirect': url_for('index')
                })

        # Re-detect live data end (in case file was updated)
        print("🔍 Re-detecting live data endpoint...")
        analyzer._find_live_data_end()

        # Generate fresh analysis
        print("🔄 Generating fresh analysis...")
        signals = analyzer.analyze_signals()
        charts = analyzer.create_charts()
        stats = analyzer.get_statistics()

        print("✅ Refresh: Analysis completed successfully")

        return jsonify({
            'success': True,
            'signals': signals,
            'charts': charts,
            'stats': stats
        })

    except Exception as e:
        print(f"❌ Refresh error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error refreshing analysis: {str(e)}'
        })


@app.route('/api/status')
def get_status():
    """API endpoint to check system status"""
    try:
        status = {
            'data_loaded': analyzer.is_data_loaded,
            'session_active': session.get('data_loaded', False),
            'current_file': session.get('current_file_path', None),
            'live_periods': len(analyzer.get_live_data()) if analyzer.get_live_data() is not None else 0,
            'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            'session_id': session.get('_id', 'No session')
        }
        print(f"📊 Status check: {status}")
        return jsonify(status)
    except Exception as e:
        print(f"❌ Status error: {e}")
        return jsonify({'error': str(e)})


@app.route('/debug')
def debug():
    """Debug endpoint to check system health"""
    try:
        debug_info = {
            'flask_app': 'Running',
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            'session_data': dict(session),
            'analyzer_loaded': analyzer.is_data_loaded,
            'analyzer_file': analyzer.current_file_path,
            'current_time': datetime.now().isoformat()
        }

        return f"""
        <h1>Money Flow Dashboard Debug</h1>
        <h2>System Status: ✅ RUNNING</h2>
        <pre>{json.dumps(debug_info, indent=2)}</pre>
        <a href="/">← Back to Dashboard</a>
        """
    except Exception as e:
        return f"<h1>Debug Error: {e}</h1>"


@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    print("❌ File too large error")
    return jsonify({'success': False, 'message': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    print(f"❌ Internal server error: {e}")
    traceback.print_exc()
    return jsonify({'success': False, 'message': 'Internal server error. Please try again.'}), 500


@app.before_request
def log_request():
    """Log all requests for debugging"""
    if request.endpoint not in ['static']:  # Don't log static file requests
        print(f"🌐 {request.method} {request.path} from {request.remote_addr}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 STARTING MONEY FLOW TRADING DASHBOARD")
    print("=" * 60)
    print("📝 Features enabled:")
    print("   ✅ Session persistence")
    print("   ✅ Auto-refresh with session restore")
    print("   ✅ Comprehensive error handling")
    print("   ✅ Debug logging")
    print("   ✅ File upload validation")
    print("   ✅ Chart generation")
    print("   ✅ Signal analysis")
    print("   ✅ Alert system")
    print(f"\n📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📏 Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024):.0f}MB")
    print(f"\n🌐 Access URLs:")
    print("   • Main Dashboard: http://localhost:5001/")
    print("   • Debug Info: http://localhost:5001/debug")
    print("   • System Status: http://localhost:5001/api/status")
    print("\n🎯 Ready for CSV upload!")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5001)