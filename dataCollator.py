import pandas as pd
import sqlite3
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NiftyDataIntegrator:
    def __init__(self):
        # File paths
        self.futures_base_path = r"C:\Projects\apps\_keystock_analyser\output"
        self.options_base_path = r"C:\Projects\apps\_nifty_optionanalyser\net_flow_reports"
        self.gamma_base_path = r"C:\Projects\apps\_nifty_optionanalyser\option_analysis_reports"
        self.db_path = r"C:\Projects\apps\_nifty_optionanalyser\OptionAnalyser.db"

        self.historical_breakdown_signals = deque(maxlen=1000)  # Keep last 1000 signals
        self.historical_price_reversals = deque(maxlen=500)  # Keep last 500 reversals
        self.persistent_data_file = "nifty_historical_data.pkl"

        # Load existing historical data on startup
        self.load_historical_data()


        # Weights for signal calculation
        self.futures_weight = 0.70
        self.options_weight = 0.30
        
        # Data storage
        self.unified_data = []
        self.latest_data = {}
        self.running = False
        
        # Initialize data structure
        self.initialize_data_structure()
    
    def initialize_data_structure(self):
        """Initialize the unified data structure"""
        self.unified_data_template = {
            'timestamp': None,
            'futures_data': {
                'weighted_positive_money_flow': 0.0,
                'weighted_negative_money_flow': 0.0,
                'weighted_money_flow': 0.0,
                'cumulative_weighted_money_flow': 0.0
            },
            'options_data': {
                'net_flow': 0.0,
                'total_flow': 0.0,
                'bullish_flow': 0.0,
                'bearish_flow': 0.0,
                'sentiment': 'Neutral',
                'call_buying': 0.0,
                'put_writing': 0.0,
                'call_short_covering': 0.0,
                'put_unwinding': 0.0,
                'put_buying': 0.0,
                'call_writing': 0.0,
                'put_short_covering': 0.0,
                'call_unwinding': 0.0
            },
            'gamma_data': {
                'support_pressure': 0.0,
                'resistance_pressure': 0.0,
                'sr_ratio': 0.0,
                'max_pressure_strike': 0.0,
                'max_pressure_value': 0.0,
                'price_reversals': [],
                'breakdown_signals': [],
                'support_levels': [],
                'resistance_levels': [],
                'spot_price': 0.0,
                'spot_change': 0.0,
                'trend_direction': 'NEUTRAL'
            },
            'spot_data': {
                'spot_price': 0.0,
                'price_change': 0.0,
                'price_change_pct': 0.0
            },
            'signals': {
                'combined_signal': 0.0,
                'signal_strength': 0.0,
                'direction': 'Neutral',
                'confidence': 0.0
            }
        }

    def get_enhanced_latest_data(self) -> Dict:
        """Get latest data with enhanced historical context"""
        latest = self.get_latest_data()
        if not latest:
            return {}

        # Add historical breakdown signals from today
        today = datetime.now().strftime('%Y-%m-%d')
        today_breakdown_signals = [
            signal for signal in self.historical_breakdown_signals
            if signal.get('date') == today
        ]

        # Add recent price reversals
        recent_reversals = [
            reversal for reversal in self.historical_price_reversals
            if reversal.get('date') == today
        ]

        # Enhance the latest data
        enhanced_data = latest.copy()
        enhanced_data['historical_breakdown_signals'] = today_breakdown_signals[-20:]  # Last 20 signals
        enhanced_data['recent_price_reversals'] = recent_reversals[-10:]  # Last 10 reversals

        return enhanced_data

    def get_historical_breakdown_signals(self, hours: int = 24) -> List[Dict]:
        """Get historical breakdown signals for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            signal for signal in self.historical_breakdown_signals
            if signal.get('timestamp', datetime.now()) >= cutoff_time
        ]

    def get_historical_price_reversals(self, hours: int = 24) -> List[Dict]:
        """Get historical price reversals for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            reversal for reversal in self.historical_price_reversals
            if reversal.get('timestamp', datetime.now()) >= cutoff_time
        ]

    def save_historical_data(self):
        """Save historical data to disk"""
        try:
            historical_data = {
                'breakdown_signals': list(self.historical_breakdown_signals),
                'price_reversals': list(self.historical_price_reversals),
                'last_saved': datetime.now().isoformat()
            }

            with open(self.persistent_data_file, 'wb') as f:
                pickle.dump(historical_data, f)

            logger.info(
                f"Historical data saved: {len(self.historical_breakdown_signals)} signals, {len(self.historical_price_reversals)} reversals")

        except Exception as e:
            logger.error(f"Error saving historical data: {e}")

    def load_historical_data(self):
        """Load historical data from disk"""
        try:
            if os.path.exists(self.persistent_data_file):
                with open(self.persistent_data_file, 'rb') as f:
                    historical_data = pickle.load(f)

                # Load breakdown signals
                if 'breakdown_signals' in historical_data:
                    self.historical_breakdown_signals.extend(historical_data['breakdown_signals'])

                # Load price reversals
                if 'price_reversals' in historical_data:
                    self.historical_price_reversals.extend(historical_data['price_reversals'])

                logger.info(
                    f"Historical data loaded: {len(self.historical_breakdown_signals)} signals, {len(self.historical_price_reversals)} reversals")

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")


    def get_latest_futures_file(self) -> Optional[str]:
        """Get the latest futures data file"""
        try:
            files = [f for f in os.listdir(self.futures_base_path) if f.startswith('nifty_detailed_') and f.endswith('_5min.csv')]
            if not files:
                logger.warning("No futures files found")
                return None
            
            # Sort by date in filename
            files.sort(reverse=True)
            return os.path.join(self.futures_base_path, files[0])
        except Exception as e:
            logger.error(f"Error finding futures file: {e}")
            return None
    
    def read_futures_data(self) -> Dict:
        """Read futures money flow data"""
        file_path = self.get_latest_futures_file()
        if not file_path:
            return {}
        
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get latest data
            latest = df.iloc[-1]
            
            return {
                'weighted_positive_money_flow': float(latest['weighted_positive_money_flow']),
                'weighted_negative_money_flow': float(latest['weighted_negative_money_flow']),
                'weighted_money_flow': float(latest['weighted_money_flow']),
                'cumulative_weighted_money_flow': float(latest['cumulative_weighted_money_flow']),
                'timestamp': latest['timestamp']
            }
        except Exception as e:
            logger.error(f"Error reading futures data: {e}")
            return {}
    
    def read_options_data(self) -> Dict:
        """Read options money flow data"""
        file_path = os.path.join(self.options_base_path, 'net_money_flow_data.csv')

        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Get latest data
            latest = df.iloc[-1]

            return {
                'net_flow': float(latest['net_flow']),
                'total_flow': float(latest['total_flow']),
                'bullish_flow': float(latest['bullish_flow']),
                'bearish_flow': float(latest['bearish_flow']),
                'sentiment': str(latest['sentiment']),
                'call_buying': float(latest['call_buying']),
                'put_writing': float(latest['put_writing']),
                'call_short_covering': float(latest['call_short_covering']),
                'put_unwinding': float(latest['put_unwinding']),
                'put_buying': float(latest['put_buying']),
                'call_writing': float(latest['call_writing']),
                'put_short_covering': float(latest['put_short_covering']),
                'call_unwinding': float(latest['call_unwinding']),
                'timestamp': latest['timestamp']
            }
        except Exception as e:
            logger.error(f"Error reading options data: {e}")
            return {}
        
    def get_latest_gamma_file(self) -> Optional[str]:
        """Get the latest gamma analysis HTML file"""
        try:
            files = [f for f in os.listdir(self.gamma_base_path) if f.startswith('nifty_comparison_report_') and f.endswith('.html')]
            if not files:
                logger.warning("No gamma files found")
                return None
            
            files.sort(reverse=True)
            return os.path.join(self.gamma_base_path, files[0])
        except Exception as e:
            logger.error(f"Error finding gamma file: {e}")
            return None

    # Replace the breakdown signals parsing section in your dataCollator.py
    # Around line 280-320 in the parse_gamma_html method

    def parse_gamma_html(self, html_content: str) -> Dict:
        """Parse gamma analysis from HTML content - Clean enhanced version"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()

            gamma_data = {
                'support_pressure': 0.0,
                'resistance_pressure': 0.0,
                'sr_ratio': 0.0,
                'max_pressure_strike': 0.0,
                'max_pressure_value': 0.0,
                'price_reversals': [],
                'breakdown_signals': [],
                'support_levels': [],
                'resistance_levels': [],
                'spot_price': 0.0,
                'spot_change': 0.0,
                'trend_direction': 'NEUTRAL'
            }

            # Extract current spot price and change from the last report table
            tables = soup.find_all('table')
            if len(tables) >= 2:  # Last table has all reports
                last_table = tables[-1]
                rows = last_table.find_all('tr')
                if len(rows) > 1:  # Skip header row
                    last_row = rows[-1]  # Get the most recent data
                    cells = last_row.find_all('td')
                    if len(cells) >= 7:
                        try:
                            gamma_data['spot_price'] = float(cells[1].text.strip())
                            gamma_data['support_pressure'] = float(cells[5].text.strip())
                            gamma_data['resistance_pressure'] = float(cells[6].text.strip())
                            gamma_data['sr_ratio'] = float(cells[7].text.strip())
                        except (ValueError, IndexError):
                            pass

            # Extract max pressure strike from comparison table
            if len(tables) >= 1:
                comparison_table = tables[0]  # First table is the comparison
                rows = comparison_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2 and 'Max Pressure Strike' in cells[0].text:
                        try:
                            gamma_data['max_pressure_strike'] = float(cells[2].text.strip())
                        except (ValueError, IndexError):
                            pass
                    elif len(cells) >= 2 and 'Max Pressure Value' in cells[0].text:
                        try:
                            gamma_data['max_pressure_value'] = float(cells[2].text.strip())
                        except (ValueError, IndexError):
                            pass

            # Extract spot price change from summary
            spot_change_match = re.search(r'decreased by ([\d.]+) points', text_content)
            if not spot_change_match:
                spot_change_match = re.search(r'increased by ([\d.]+) points', text_content)
                if spot_change_match:
                    gamma_data['spot_change'] = float(spot_change_match.group(1))
            else:
                gamma_data['spot_change'] = -float(spot_change_match.group(1))

            # Extract trend direction
            if 'DOWNWARD' in text_content:
                gamma_data['trend_direction'] = 'BEARISH'
            elif 'UPWARD' in text_content:
                gamma_data['trend_direction'] = 'BULLISH'

            # ENHANCED: Extract breakdown signals with proper cleaning
            breakdown_signals = []

            # Method 1: Strong Breakdown Signals
            strong_matches = re.findall(
                r'(\d{2}:\d{2}:\d{2}):\s*(BREAKDOWN CONFIRMATION[^0-9]*?)(?=\d{2}:\d{2}:\d{2}|Support Erosion|$)',
                text_content, re.DOTALL
            )
            for time_str, signal_text in strong_matches:
                clean_signal = re.sub(r'\s+', ' ', signal_text.strip())
                if clean_signal:
                    breakdown_signals.append({
                        'time': time_str,
                        'signal': clean_signal,
                        'type': 'CONFIRMATION'
                    })

            # Method 2: Breakdown Signals
            signal_matches = re.findall(
                r'(\d{2}:\d{2}:\d{2}):\s*(BREAKDOWN SIGNAL[^0-9]*?)(?=\d{2}:\d{2}:\d{2}|Support Erosion|$)',
                text_content, re.DOTALL
            )
            for time_str, signal_text in signal_matches:
                clean_signal = re.sub(r'\s+', ' ', signal_text.strip())
                if clean_signal:
                    breakdown_signals.append({
                        'time': time_str,
                        'signal': clean_signal,
                        'type': 'WARNING'
                    })

            # Method 3: Support Erosion Events (cleaned)
            erosion_matches = re.findall(
                r'(\d{2}:\d{2}:\d{2}):\s*Support pressure decreased by ([\d,]+)\s*\(([\d.]+)%\)',
                text_content
            )
            for time_str, amount, percentage in erosion_matches:
                # Only include significant erosions (>10%)
                if float(percentage) > 10:
                    breakdown_signals.append({
                        'time': time_str,
                        'signal': f'Support Erosion: {amount} ({percentage}%)',
                        'type': 'EROSION'
                    })

            # Method 4: Critical BEARISH reversals only (cleaned)
            reversal_matches = re.findall(
                r'(\d{2}:\d{2}:\d{2}):\s*BEARISH reversal at price ([\d.]+)',
                text_content
            )
            for time_str, price in reversal_matches:
                breakdown_signals.append({
                    'time': time_str,
                    'signal': f'BEARISH reversal at price {price}',
                    'type': 'REVERSAL'
                })

            # Method 5: Any other critical breakdown patterns
            critical_matches = re.findall(
                r'(\d{2}:\d{2}:\d{2}):\s*(CRITICAL[^0-9]*?)(?=\d{2}:\d{2}:\d{2}|$)',
                text_content, re.DOTALL
            )
            for time_str, signal_text in critical_matches:
                clean_signal = re.sub(r'\s+', ' ', signal_text.strip())
                if clean_signal and len(clean_signal) > 10:  # Avoid incomplete signals
                    breakdown_signals.append({
                        'time': time_str,
                        'signal': clean_signal,
                        'type': 'CRITICAL'
                    })

            # Remove duplicates and incomplete signals
            unique_signals = {}
            for signal in breakdown_signals:
                key = signal['time']
                # Skip incomplete signals
                if len(signal['signal']) < 10 or 'decreased by' in signal['signal'] and '(' not in signal['signal']:
                    continue

                # Keep the longest/most complete signal for each time
                if key not in unique_signals or len(signal['signal']) > len(unique_signals[key]['signal']):
                    unique_signals[key] = signal

            # Sort by time and store
            gamma_data['breakdown_signals'] = sorted(unique_signals.values(), key=lambda x: x['time'])

            # Extract price reversals separately (for the price reversals section)
            reversal_pattern = r'(\d{2}:\d{2}:\d{2}):\s*(BEARISH|BULLISH)\s*reversal at price ([\d.]+)'
            reversals = re.findall(reversal_pattern, text_content)
            for time_str, direction, price in reversals:
                gamma_data['price_reversals'].append({
                    'time': time_str,
                    'direction': direction,
                    'price': float(price)
                })

            # Extract support and resistance levels
            support_added_pattern = r'New support level\(s\) added: ([\d, ]+)'
            resistance_added_pattern = r'New resistance level\(s\) added: ([\d, ]+)'

            support_added = re.findall(support_added_pattern, text_content)
            resistance_added = re.findall(resistance_added_pattern, text_content)

            # Parse support levels
            for levels_str in support_added:
                levels = [int(level.strip()) for level in levels_str.split(',') if level.strip().isdigit()]
                gamma_data['support_levels'].extend(levels)

            # Parse resistance levels
            for levels_str in resistance_added:
                levels = [int(level.strip()) for level in levels_str.split(',') if level.strip().isdigit()]
                gamma_data['resistance_levels'].extend(levels)

            # Remove duplicates and sort
            gamma_data['support_levels'] = sorted(list(set(gamma_data['support_levels'])))
            gamma_data['resistance_levels'] = sorted(list(set(gamma_data['resistance_levels'])))

            return gamma_data

        except Exception as e:
            logger.error(f"Error parsing gamma HTML: {e}")
            return {
                'support_pressure': 0.0,
                'resistance_pressure': 0.0,
                'sr_ratio': 0.0,
                'max_pressure_strike': 0.0,
                'max_pressure_value': 0.0,
                'price_reversals': [],
                'breakdown_signals': [],
                'support_levels': [],
                'resistance_levels': [],
                'spot_price': 0.0,
                'spot_change': 0.0,
                'trend_direction': 'NEUTRAL'
            }
    
    def read_gamma_data(self) -> Dict:
        """Read gamma analysis data"""
        file_path = self.get_latest_gamma_file()
        if not file_path:
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return self.parse_gamma_html(html_content)
            
        except Exception as e:
            logger.error(f"Error reading gamma data: {e}")
            return {}
    
    def read_spot_data(self) -> Dict:
        """Read spot price data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get today's date
            today = datetime.now().strftime('%Y-%m-%d')
            
            query = """
            SELECT DISTINCT strftime('%H:%M', timestamp) as time, 
                   Spot as spot,
                   timestamp
            FROM option_chain_data 
            WHERE date(timestamp) = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn, params=[today])
            conn.close()
            
            if df.empty:
                return {'spot_price': 0.0, 'price_change': 0.0, 'price_change_pct': 0.0}
            
            current_price = float(df.iloc[0]['spot'])
            
            # Calculate price change (you might want to store previous price)
            # For now, using a simple calculation
            return {
                'spot_price': current_price,
                'price_change': 0.0,  # Calculate based on previous reading
                'price_change_pct': 0.0  # Calculate based on previous reading
            }
            
        except Exception as e:
            logger.error(f"Error reading spot data: {e}")
            return {'spot_price': 0.0, 'price_change': 0.0, 'price_change_pct': 0.0}
    
    def calculate_combined_signal(self, futures_data: Dict, options_data: Dict) -> Dict:
        """Calculate combined signal using 70/30 weighting"""
        try:
            # Normalize futures signal (-1 to 1)
            futures_signal = futures_data.get('weighted_money_flow', 0.0)
            futures_normalized = max(-1, min(1, futures_signal / 1000000))  # Adjust divisor based on your data scale
            
            # Normalize options signal (-1 to 1)
            options_signal = options_data.get('net_flow', 0.0)
            options_normalized = max(-1, min(1, options_signal / 100000))  # Adjust divisor based on your data scale
            
            # Combined signal with weighting
            combined_signal = (self.futures_weight * futures_normalized) + (self.options_weight * options_normalized)
            
            # Signal strength (0 to 10)
            signal_strength = abs(combined_signal) * 10
            
            # Direction
            if combined_signal > 0.2:
                direction = 'Bullish'
            elif combined_signal < -0.2:
                direction = 'Bearish'
            else:
                direction = 'Neutral'
            
            # Confidence based on agreement between futures and options
            agreement = 1 - abs(futures_normalized - options_normalized) / 2
            confidence = agreement * 100
            
            return {
                'combined_signal': combined_signal,
                'signal_strength': signal_strength,
                'direction': direction,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating combined signal: {e}")
            return {
                'combined_signal': 0.0,
                'signal_strength': 0.0,
                'direction': 'Neutral',
                'confidence': 0.0
            }

    def collect_unified_data(self) -> Dict:
        """Enhanced data collection with historical storage"""
        logger.info("Collecting data from all sources...")

        # Read data from all sources
        futures_data = self.read_futures_data()
        options_data = self.read_options_data()
        gamma_data = self.read_gamma_data()
        spot_data = self.read_spot_data()

        # Calculate combined signals
        signals = self.calculate_combined_signal(futures_data, options_data)

        # Create unified data structure
        unified_data = {
            'timestamp': datetime.now(),
            'futures_data': futures_data,
            'options_data': options_data,
            'gamma_data': gamma_data,
            'spot_data': spot_data,
            'signals': signals
        }

        # Store latest data
        self.latest_data = unified_data

        # Add to historical data
        self.unified_data.append(unified_data)

        # ENHANCED: Store breakdown signals with timestamps
        if gamma_data.get('breakdown_signals'):
            for signal in gamma_data['breakdown_signals']:
                signal_with_full_timestamp = {
                    'timestamp': datetime.now(),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': signal['time'],
                    'signal': signal['signal'],
                    'type': signal.get('type', 'UNKNOWN'),
                    'spot_price': gamma_data.get('spot_price', 0.0),
                    'sr_ratio': gamma_data.get('sr_ratio', 0.0)
                }
                self.historical_breakdown_signals.append(signal_with_full_timestamp)

        # ENHANCED: Store price reversals with timestamps
        if gamma_data.get('price_reversals'):
            for reversal in gamma_data['price_reversals']:
                reversal_with_full_timestamp = {
                    'timestamp': datetime.now(),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': reversal['time'],
                    'direction': reversal['direction'],
                    'price': reversal['price']
                }
                self.historical_price_reversals.append(reversal_with_full_timestamp)

        # Keep only last 24 hours of unified data (288 data points at 5-minute intervals)
        if len(self.unified_data) > 288:
            self.unified_data = self.unified_data[-288:]

        # Periodically save historical data
        if len(self.unified_data) % 12 == 0:  # Save every hour
            self.save_historical_data()

        logger.info(
            f"Data collection complete. Signal: {signals['direction']} (Strength: {signals['signal_strength']:.1f})")

        return unified_data
    
    def get_latest_data(self) -> Dict:
        """Get the latest unified data"""
        return self.latest_data
    
    def get_historical_data(self, hours: int = 24) -> List[Dict]:
        """Get historical data for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [data for data in self.unified_data if data['timestamp'] >= cutoff_time]
    
    def export_data_to_json(self, filename: str = None) -> str:
        """Export current data to JSON file"""
        if not filename:
            filename = f"unified_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Convert datetime objects to strings for JSON serialization
            export_data = []
            for data in self.unified_data:
                data_copy = data.copy()
                data_copy['timestamp'] = data_copy['timestamp'].isoformat()
                export_data.append(data_copy)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return ""
    
    def start_continuous_collection(self, interval_minutes: int = 5):
        """Start continuous data collection every N minutes"""
        self.running = True
        
        def collection_loop():
            while self.running:
                try:
                    self.collect_unified_data()
                    time.sleep(interval_minutes * 60)  # Convert minutes to seconds
                except Exception as e:
                    logger.error(f"Error in collection loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info(f"Started continuous data collection every {interval_minutes} minutes")
    
    def stop_continuous_collection(self):
        """Stop continuous data collection"""
        self.running = False
        logger.info("Stopped continuous data collection")

    def get_summary_stats(self) -> Dict:
        """Enhanced summary statistics with historical data"""
        if not self.unified_data:
            return {}

        try:
            signals = [data['signals']['combined_signal'] for data in self.unified_data]
            strengths = [data['signals']['signal_strength'] for data in self.unified_data]

            # Count breakdown signals by type
            breakdown_by_type = {}
            for signal in self.historical_breakdown_signals:
                signal_type = signal.get('type', 'UNKNOWN')
                breakdown_by_type[signal_type] = breakdown_by_type.get(signal_type, 0) + 1

            return {
                'total_data_points': len(self.unified_data),
                'avg_signal': sum(signals) / len(signals),
                'avg_strength': sum(strengths) / len(strengths),
                'max_signal': max(signals),
                'min_signal': min(signals),
                'bullish_count': len([s for s in self.unified_data if s['signals']['direction'] == 'Bullish']),
                'bearish_count': len([s for s in self.unified_data if s['signals']['direction'] == 'Bearish']),
                'neutral_count': len([s for s in self.unified_data if s['signals']['direction'] == 'Neutral']),
                'total_breakdown_signals': len(self.historical_breakdown_signals),
                'breakdown_by_type': breakdown_by_type,
                'total_price_reversals': len(self.historical_price_reversals)
            }
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Initialize the data integrator
    integrator = NiftyDataIntegrator()
    
    # Collect data once
    data = integrator.collect_unified_data()
    print("Latest unified data:")
    print(json.dumps(data, indent=2, default=str))
    
    # Start continuous collection (uncomment to enable)
    # integrator.start_continuous_collection(interval_minutes=5)
    
    # Get summary stats
    stats = integrator.get_summary_stats()
    print("\nSummary Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Export data to JSON
    # filename = integrator.export_data_to_json()
    # print(f"Data exported to: {filename}")