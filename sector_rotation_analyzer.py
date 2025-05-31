#!/usr/bin/env python3
"""
Sector Rotation Analysis System
Calculates relative sector performance to Nifty Index and generates rotation signals

Path: C:\Projects\apps\SectorAnalysis\sector_rotation_analyzer.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the mysql library path
sys.path.append(r'C:\Projects\library\mysql')
from mydbmanager import DatabaseManager

class SectorRotationAnalyzer:
    def __init__(self, db_config):
        """
        Initialize the Sector Rotation Analyzer
        
        Args:
            db_config (dict): Database configuration containing host, user, password, database
        """
        self.db = DatabaseManager(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        
        # Define sector mappings
        self.nifty_sectors = [
            'Nifty Auto', 'Nifty Bank', 'NIFTY IT', 'NIFTY INFRA', 'NIFTY REALTY',
            'NIFTY CONSUMPTION', 'NIFTY PVT BANK', 'NIFTY PHARMA', 'NIFTY COMMODITIES',
            'NIFTY FMCG', 'NIFTY SERV SECTOR', 'NIFTY MEDIA', 'NIFTY PSU BANK',
            'NIFTY ENERGY', 'NIFTY METAL', 'NIFTY FIN SERVICE', 'NIFTY OIL AND GAS',
            'NIFTY HEALTHCARE', 'NIFTY CONSR DURBL'
        ]
        
        # Rotation thresholds
        self.thresholds = {
            'short_term': {'alert': 7, 'strong': 10},    # 1-3 months
            'medium_term': {'alert': 15, 'strong': 25},  # 3-12 months
            'long_term': {'alert': 30, 'strong': 50}     # 1-3 years
        }
        
    def connect_database(self):
        """Connect to the database"""
        try:
            self.db.connect()
            print("Successfully connected to database for sector rotation analysis")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise
    
    def get_nifty_data(self, days_back=365):
        """
        Fetch Nifty 50 index data
        
        Args:
            days_back (int): Number of days to fetch data for
            
        Returns:
            pd.DataFrame: Nifty data with date, close price, and returns
        """
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            query = f"""
            SELECT timestamp, close, p_change, volume
            FROM _nse_sectorindex 
            WHERE sector = 'Nifty 50' 
            AND timestamp >= '{start_date.strftime('%Y-%m-%d')}'
            ORDER BY timestamp ASC
            """
            
            result = self.db.execute_query(query, fetch=True)
            if not result:
                print("No Nifty data found")
                return None
                
            # Parse the result
            lines = result.split('#r#')
            header = lines[0].split('|')
            
            data = []
            for line in lines[1:]:
                values = line.split('|')
                data.append(values)
            
            df = pd.DataFrame(data, columns=header)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['p_change'] = pd.to_numeric(df['p_change'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Calculate returns if not available
            if 'p_change' not in df.columns or df['p_change'].isna().all():
                df['p_change'] = df['close'].pct_change() * 100
            
            df = df.dropna()
            print(f"Fetched {len(df)} records for Nifty 50")
            return df
            
        except Exception as e:
            print(f"Error fetching Nifty data: {e}")
            return None
    
    def get_sector_data(self, days_back=365):
        """
        Fetch sector data for all tracked sectors
        
        Args:
            days_back (int): Number of days to fetch data for
            
        Returns:
            dict: Dictionary with sector names as keys and DataFrames as values
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Create IN clause for sectors
            sectors_str = "', '".join(self.nifty_sectors)
            
            query = f"""
            SELECT sector, timestamp, close, p_change, volume
            FROM _nse_sectorindex 
            WHERE sector IN ('{sectors_str}')
            AND timestamp >= '{start_date.strftime('%Y-%m-%d')}'
            ORDER BY sector, timestamp ASC
            """
            
            result = self.db.execute_query(query, fetch=True)
            if not result:
                print("No sector data found")
                return {}
                
            # Parse the result
            lines = result.split('#r#')
            header = lines[0].split('|')
            
            data = []
            for line in lines[1:]:
                values = line.split('|')
                data.append(values)
            
            df = pd.DataFrame(data, columns=header)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['p_change'] = pd.to_numeric(df['p_change'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Group by sector
            sector_data = {}
            for sector in self.nifty_sectors:
                sector_df = df[df['sector'] == sector].copy()
                if not sector_df.empty:
                    # Calculate returns if not available
                    if 'p_change' not in sector_df.columns or sector_df['p_change'].isna().all():
                        sector_df['p_change'] = sector_df['close'].pct_change() * 100
                    
                    sector_df = sector_df.dropna()
                    sector_data[sector] = sector_df
                    print(f"Fetched {len(sector_df)} records for {sector}")
            
            return sector_data
            
        except Exception as e:
            print(f"Error fetching sector data: {e}")
            return {}
    
    def calculate_relative_performance(self, nifty_data, sector_data, periods=[30, 90, 180]):
        """
        Calculate relative performance of sectors vs Nifty
        
        Args:
            nifty_data (pd.DataFrame): Nifty 50 data
            sector_data (dict): Dictionary of sector DataFrames
            periods (list): List of periods in days to calculate performance
            
        Returns:
            pd.DataFrame: Relative performance matrix
        """
        try:
            results = []
            
            for sector_name, sector_df in sector_data.items():
                if sector_df.empty:
                    continue
                    
                sector_result = {'sector': sector_name}
                
                for period in periods:
                    # Get data for the specific period
                    end_date = nifty_data['timestamp'].max()
                    start_date = end_date - timedelta(days=period)
                    
                    # Filter data for the period
                    nifty_period = nifty_data[
                        (nifty_data['timestamp'] >= start_date) & 
                        (nifty_data['timestamp'] <= end_date)
                    ]
                    
                    sector_period = sector_df[
                        (sector_df['timestamp'] >= start_date) & 
                        (sector_df['timestamp'] <= end_date)
                    ]
                    
                    if len(nifty_period) > 1 and len(sector_period) > 1:
                        # Calculate cumulative returns
                        nifty_return = ((nifty_period['close'].iloc[-1] / nifty_period['close'].iloc[0]) - 1) * 100
                        sector_return = ((sector_period['close'].iloc[-1] / sector_period['close'].iloc[0]) - 1) * 100
                        
                        # Calculate relative performance
                        relative_performance = sector_return - nifty_return
                        
                        sector_result[f'{period}d_nifty_return'] = round(nifty_return, 2)
                        sector_result[f'{period}d_sector_return'] = round(sector_return, 2)
                        sector_result[f'{period}d_relative_performance'] = round(relative_performance, 2)
                        
                        # Determine signal strength
                        if period <= 90:  # Short term
                            thresholds = self.thresholds['short_term']
                        elif period <= 365:  # Medium term
                            thresholds = self.thresholds['medium_term']
                        else:  # Long term
                            thresholds = self.thresholds['long_term']
                        
                        if abs(relative_performance) >= thresholds['strong']:
                            signal = 'STRONG'
                        elif abs(relative_performance) >= thresholds['alert']:
                            signal = 'ALERT'
                        else:
                            signal = 'NORMAL'
                        
                        direction = 'OUTPERFORM' if relative_performance > 0 else 'UNDERPERFORM'
                        sector_result[f'{period}d_signal'] = f"{signal}_{direction}"
                    else:
                        # Insufficient data
                        sector_result[f'{period}d_nifty_return'] = None
                        sector_result[f'{period}d_sector_return'] = None
                        sector_result[f'{period}d_relative_performance'] = None
                        sector_result[f'{period}d_signal'] = 'NO_DATA'
                
                results.append(sector_result)
            
            df_results = pd.DataFrame(results)
            return df_results
            
        except Exception as e:
            print(f"Error calculating relative performance: {e}")
            return pd.DataFrame()
    
    def generate_rotation_signals(self, relative_performance_df):
        """
        Generate sector rotation signals based on relative performance
        
        Args:
            relative_performance_df (pd.DataFrame): Relative performance data
            
        Returns:
            dict: Rotation signals and recommendations
        """
        try:
            signals = {
                'rotation_opportunities': [],
                'outperforming_sectors': [],
                'underperforming_sectors': [],
                'momentum_alerts': [],
                'summary': {}
            }
            
            for _, row in relative_performance_df.iterrows():
                sector = row['sector']
                
                # Analyze signals across timeframes
                short_signal = row.get('30d_signal', 'NO_DATA')
                medium_signal = row.get('90d_signal', 'NO_DATA')
                long_signal = row.get('180d_signal', 'NO_DATA')
                
                short_rel_perf = row.get('30d_relative_performance', 0)
                medium_rel_perf = row.get('90d_relative_performance', 0)
                long_rel_perf = row.get('180d_relative_performance', 0)
                
                # Rotation opportunity detection
                if (short_rel_perf > 7 and medium_rel_perf < -5):
                    signals['rotation_opportunities'].append({
                        'sector': sector,
                        'type': 'BULLISH_REVERSAL',
                        'description': f"Recent outperformance ({short_rel_perf:.1f}%) after medium-term underperformance",
                        'confidence': 'HIGH' if abs(short_rel_perf) > 10 else 'MEDIUM'
                    })
                elif (short_rel_perf < -7 and medium_rel_perf > 5):
                    signals['rotation_opportunities'].append({
                        'sector': sector,
                        'type': 'BEARISH_REVERSAL',
                        'description': f"Recent underperformance ({short_rel_perf:.1f}%) after medium-term outperformance",
                        'confidence': 'HIGH' if abs(short_rel_perf) > 10 else 'MEDIUM'
                    })
                
                # Consistent performers
                if all(perf > 5 for perf in [short_rel_perf, medium_rel_perf, long_rel_perf] if perf is not None):
                    signals['outperforming_sectors'].append({
                        'sector': sector,
                        'avg_outperformance': np.mean([p for p in [short_rel_perf, medium_rel_perf, long_rel_perf] if p is not None]),
                        'consistency': 'HIGH'
                    })
                elif all(perf < -5 for perf in [short_rel_perf, medium_rel_perf, long_rel_perf] if perf is not None):
                    signals['underperforming_sectors'].append({
                        'sector': sector,
                        'avg_underperformance': np.mean([p for p in [short_rel_perf, medium_rel_perf, long_rel_perf] if p is not None]),
                        'consistency': 'HIGH'
                    })
                
                # Momentum alerts
                if 'STRONG' in short_signal:
                    signals['momentum_alerts'].append({
                        'sector': sector,
                        'signal': short_signal,
                        'performance': short_rel_perf,
                        'timeframe': '30_days'
                    })
            
            # Generate summary
            signals['summary'] = {
                'total_sectors_analyzed': len(relative_performance_df),
                'rotation_opportunities': len(signals['rotation_opportunities']),
                'strong_outperformers': len(signals['outperforming_sectors']),
                'strong_underperformers': len(signals['underperforming_sectors']),
                'momentum_alerts': len(signals['momentum_alerts']),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return signals
            
        except Exception as e:
            print(f"Error generating rotation signals: {e}")
            return {}
    
    def create_daily_monitoring_ratios(self, nifty_data, sector_data):
        """
        Create daily monitoring ratios for key sectors
        
        Args:
            nifty_data (pd.DataFrame): Nifty 50 data
            sector_data (dict): Dictionary of sector DataFrames
            
        Returns:
            pd.DataFrame: Daily ratios for monitoring
        """
        try:
            # Key sectors for daily monitoring
            key_sectors = {
                'Bank': ['Nifty Bank', 'NIFTY FIN SERVICE'],
                'IT': ['NIFTY IT'],
                'FMCG': ['NIFTY FMCG'],
                'Auto': ['Nifty Auto'],
                'Metal': ['NIFTY METAL']
            }
            
            # Get common dates
            common_dates = set(nifty_data['timestamp'])
            for sector_df in sector_data.values():
                common_dates = common_dates.intersection(set(sector_df['timestamp']))
            
            common_dates = sorted(list(common_dates))[-30:]  # Last 30 days
            
            ratios_data = []
            
            for date in common_dates:
                nifty_close = nifty_data[nifty_data['timestamp'] == date]['close'].iloc[0]
                ratio_record = {'date': date, 'nifty_close': nifty_close}
                
                for ratio_name, sector_list in key_sectors.items():
                    sector_close = 0
                    sector_count = 0
                    
                    for sector_name in sector_list:
                        if sector_name in sector_data:
                            sector_df = sector_data[sector_name]
                            sector_row = sector_df[sector_df['timestamp'] == date]
                            if not sector_row.empty:
                                sector_close += sector_row['close'].iloc[0]
                                sector_count += 1
                    
                    if sector_count > 0:
                        avg_sector_close = sector_close / sector_count
                        ratio = avg_sector_close / nifty_close
                        ratio_record[f'{ratio_name.lower()}_nifty_ratio'] = round(ratio, 4)
                    else:
                        ratio_record[f'{ratio_name.lower()}_nifty_ratio'] = None
                
                ratios_data.append(ratio_record)
            
            ratios_df = pd.DataFrame(ratios_data)
            
            # Calculate ratio changes
            for col in ratios_df.columns:
                if col.endswith('_ratio'):
                    ratios_df[f'{col}_change'] = ratios_df[col].pct_change() * 100
            
            return ratios_df
            
        except Exception as e:
            print(f"Error creating daily monitoring ratios: {e}")
            return pd.DataFrame()
    
    def generate_report(self, relative_performance_df, rotation_signals, ratios_df):
        """
        Generate a comprehensive sector rotation report
        
        Args:
            relative_performance_df (pd.DataFrame): Relative performance data
            rotation_signals (dict): Rotation signals
            ratios_df (pd.DataFrame): Daily monitoring ratios
            
        Returns:
            str: Formatted report
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("SECTOR ROTATION ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Summary section
            report.append("EXECUTIVE SUMMARY:")
            report.append("-" * 40)
            summary = rotation_signals.get('summary', {})
            for key, value in summary.items():
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
            report.append("")
            
            # Rotation opportunities
            if rotation_signals.get('rotation_opportunities'):
                report.append("ROTATION OPPORTUNITIES:")
                report.append("-" * 40)
                for opp in rotation_signals['rotation_opportunities']:
                    report.append(f"  • {opp['sector']} - {opp['type']}")
                    report.append(f"    {opp['description']}")
                    report.append(f"    Confidence: {opp['confidence']}")
                    report.append("")
            
            # Strong performers
            if rotation_signals.get('outperforming_sectors'):
                report.append("CONSISTENT OUTPERFORMERS:")
                report.append("-" * 40)
                for sector in rotation_signals['outperforming_sectors']:
                    report.append(f"  • {sector['sector']}: Avg +{sector['avg_outperformance']:.1f}%")
                report.append("")
            
            # Underperformers
            if rotation_signals.get('underperforming_sectors'):
                report.append("CONSISTENT UNDERPERFORMERS:")
                report.append("-" * 40)
                for sector in rotation_signals['underperforming_sectors']:
                    report.append(f"  • {sector['sector']}: Avg {sector['avg_underperformance']:.1f}%")
                report.append("")
            
            # Momentum alerts
            if rotation_signals.get('momentum_alerts'):
                report.append("MOMENTUM ALERTS:")
                report.append("-" * 40)
                for alert in rotation_signals['momentum_alerts']:
                    report.append(f"  • {alert['sector']}: {alert['signal']} ({alert['performance']:.1f}%)")
                report.append("")
            
            # Recent ratio changes
            if not ratios_df.empty:
                report.append("RECENT RATIO CHANGES (Latest vs Previous):")
                report.append("-" * 40)
                latest_ratios = ratios_df.iloc[-1]
                for col in ratios_df.columns:
                    if col.endswith('_ratio_change') and not pd.isna(latest_ratios[col]):
                        ratio_name = col.replace('_ratio_change', '').replace('_', ' ').title()
                        change = latest_ratios[col]
                        direction = "↑" if change > 0 else "↓"
                        report.append(f"  • {ratio_name}: {direction} {abs(change):.2f}%")
                report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return "Error generating report"
    
    def run_analysis(self, days_back=365, export_csv=True):
        """
        Run complete sector rotation analysis
        
        Args:
            days_back (int): Number of days to analyze
            export_csv (bool): Whether to export results to CSV
            
        Returns:
            dict: Analysis results
        """
        try:
            print("Starting Sector Rotation Analysis...")
            print("=" * 50)
            
            # Connect to database
            self.connect_database()
            
            # Fetch data
            print("1. Fetching Nifty 50 data...")
            nifty_data = self.get_nifty_data(days_back)
            if nifty_data is None or nifty_data.empty:
                print("Error: No Nifty data available")
                return None
            
            print("2. Fetching sector data...")
            sector_data = self.get_sector_data(days_back)
            if not sector_data:
                print("Error: No sector data available")
                return None
            
            # Calculate relative performance
            print("3. Calculating relative performance...")
            relative_performance_df = self.calculate_relative_performance(nifty_data, sector_data)
            
            # Generate rotation signals
            print("4. Generating rotation signals...")
            rotation_signals = self.generate_rotation_signals(relative_performance_df)
            
            # Create daily monitoring ratios
            print("5. Creating daily monitoring ratios...")
            ratios_df = self.create_daily_monitoring_ratios(nifty_data, sector_data)
            
            # Generate report
            print("6. Generating report...")
            report = self.generate_report(relative_performance_df, rotation_signals, ratios_df)
            
            # Export to CSV if requested
            if export_csv:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Export relative performance
                rel_perf_file = f"sector_relative_performance_{timestamp}.csv"
                relative_performance_df.to_csv(rel_perf_file, index=False)
                print(f"Relative performance exported to: {rel_perf_file}")
                
                # Export ratios
                if not ratios_df.empty:
                    ratios_file = f"daily_sector_ratios_{timestamp}.csv"
                    ratios_df.to_csv(ratios_file, index=False)
                    print(f"Daily ratios exported to: {ratios_file}")
                
                # Export report
                report_file = f"sector_rotation_report_{timestamp}.txt"
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"Report exported to: {report_file}")
            
            # Print report
            print("\n" + report)
            
            # Close database connection
            self.db.close_connection()
            
            return {
                'relative_performance': relative_performance_df,
                'rotation_signals': rotation_signals,
                'daily_ratios': ratios_df,
                'report': report
            }
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return None

def main():
    """
    Main function to run sector rotation analysis
    """
    # Database configuration
    db_config = {
        'host': 'localhost',        # Update with your database host
        'user': 'your_username',    # Update with your database username
        'password': 'your_password', # Update with your database password
        'database': 'your_database'  # Update with your database name
    }
    
    try:
        # Initialize analyzer
        analyzer = SectorRotationAnalyzer(db_config)
        
        # Run analysis
        results = analyzer.run_analysis(days_back=365, export_csv=True)
        
        if results:
            print("\nAnalysis completed successfully!")
            print(f"Analyzed {len(results['relative_performance'])} sectors")
            print(f"Found {len(results['rotation_signals']['rotation_opportunities'])} rotation opportunities")
        else:
            print("Analysis failed!")
            
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()