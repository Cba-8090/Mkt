#!/usr/bin/env python3
"""
Enhanced Sector Rotation Analyzer with Daily Monitoring
Path: C:\Projects\apps\institutional_flow_quant\enhanced_sector_analyzer.py

Features:
- Daily ratio tracking (Bank/Nifty, IT/Nifty, etc.)
- Momentum alerts
- Trend analysis
- Export capabilities
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import csv
warnings.filterwarnings('ignore')

# Add the mysql library path
sys.path.append(r'C:\Projects\library\mysql')
from mydbmanager import DatabaseManager

class EnhancedSectorRotationAnalyzer:
    def __init__(self, db_config):
        self.db = DatabaseManager(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        
        # Column mappings
        self.timestamp_col = 'timestamp'
        self.close_col = 'close'
        self.change_col = 'perchng'
        
        # Key sectors for ratio tracking
        self.key_sectors = {
            'Bank': 'Nifty Bank',
            'IT': 'NIFTY IT',
            'FMCG': 'NIFTY FMCG',
            'Auto': 'Nifty Auto',
            'Metal': 'NIFTY METAL',
            'Energy': 'NIFTY ENERGY',
            'Pharma': 'NIFTY PHARMA',
            'Realty': 'NIFTY REALTY',
            'Media': 'NIFTY MEDIA',
            'PSU_Bank': 'NIFTY PSU BANK'
        }
        
        # All sectors
        self.all_sectors = [
            'Nifty Auto', 'Nifty Bank', 'NIFTY IT', 'NIFTY INFRA', 'NIFTY REALTY',
            'NIFTY CONSUMPTION', 'NIFTY PVT BANK', 'NIFTY PHARMA', 'NIFTY COMMODITIES',
            'NIFTY FMCG', 'NIFTY SERV SECTOR', 'NIFTY MEDIA', 'NIFTY PSU BANK',
            'NIFTY ENERGY', 'NIFTY METAL', 'NIFTY FIN SERVICE', 'NIFTY OIL AND GAS',
            'NIFTY HEALTHCARE', 'NIFTY CONSR DURBL'
        ]
    
    def connect_database(self):
        try:
            self.db.connect()
            print("Successfully connected to database")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise
    
    def get_sector_data(self, sector, days_back=365):
        """Get data for a specific sector"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            query = f"""
            SELECT {self.timestamp_col}, {self.close_col}, {self.change_col}
            FROM _nse_sectorindex 
            WHERE sector = '{sector}' 
            AND {self.timestamp_col} >= '{start_date.strftime('%Y-%m-%d')}'
            ORDER BY {self.timestamp_col} ASC
            """
            
            result = self.db.execute_query(query, fetch=True)
            if not result:
                return None
                
            lines = result.split('#r#')
            header = lines[0].split('|')
            
            data = []
            for line in lines[1:]:
                if line.strip():
                    values = line.split('|')
                    data.append(values)
            
            if not data:
                return None
                
            df = pd.DataFrame(data, columns=header)
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df[self.close_col] = pd.to_numeric(df[self.close_col], errors='coerce')
            df[self.change_col] = pd.to_numeric(df[self.change_col], errors='coerce')
            
            df = df.dropna(subset=[self.close_col])
            return df
            
        except Exception as e:
            print(f"Error fetching data for {sector}: {e}")
            return None
    
    def calculate_sector_ratios(self, days_back=90):
        """Calculate key sector ratios vs Nifty"""
        try:
            print("Calculating daily sector ratios...")
            
            # Get Nifty 50 data
            nifty_data = self.get_sector_data('Nifty 50', days_back)
            if nifty_data is None:
                print("Error: Cannot get Nifty 50 data")
                return None
            
            ratios = {}
            
            for sector_name, sector_code in self.key_sectors.items():
                sector_data = self.get_sector_data(sector_code, days_back)
                
                if sector_data is not None:
                    # Merge data on timestamp
                    merged = pd.merge(
                        sector_data[['timestamp', 'close']].rename(columns={'close': 'sector_close'}),
                        nifty_data[['timestamp', 'close']].rename(columns={'close': 'nifty_close'}),
                        on='timestamp',
                        how='inner'
                    )
                    
                    if len(merged) > 0:
                        # Calculate ratio
                        merged['ratio'] = merged['sector_close'] / merged['nifty_close']
                        
                        # Calculate ratio changes
                        merged['ratio_change'] = merged['ratio'].pct_change() * 100
                        merged['ratio_sma_5'] = merged['ratio'].rolling(5).mean()
                        merged['ratio_sma_20'] = merged['ratio'].rolling(20).mean()
                        
                        ratios[sector_name] = merged
                        print(f"   ✓ {sector_name}: {len(merged)} data points")
                    else:
                        print(f"   ✗ {sector_name}: No matching data")
                else:
                    print(f"   ✗ {sector_name}: No sector data")
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return None
    
    def generate_enhanced_signals(self, ratios):
        """Generate enhanced rotation signals with momentum analysis"""
        signals = {
            'momentum_breakouts': [],
            'ratio_reversals': [],
            'trend_continuations': [],
            'alerts': []
        }
        
        for sector_name, data in ratios.items():
            if len(data) < 20:  # Need enough data for analysis
                continue
            
            latest = data.iloc[-1]
            prev_5d = data.iloc[-6] if len(data) >= 6 else data.iloc[0]
            prev_20d = data.iloc[-21] if len(data) >= 21 else data.iloc[0]
            
            current_ratio = latest['ratio']
            ratio_5d_change = ((current_ratio - prev_5d['ratio']) / prev_5d['ratio']) * 100
            ratio_20d_change = ((current_ratio - prev_20d['ratio']) / prev_20d['ratio']) * 100
            
            # Check for momentum breakouts
            if (current_ratio > latest['ratio_sma_20'] and 
                latest['ratio_sma_5'] > latest['ratio_sma_20'] and
                ratio_5d_change > 2):
                signals['momentum_breakouts'].append({
                    'sector': sector_name,
                    'signal': 'BULLISH_BREAKOUT',
                    'current_ratio': current_ratio,
                    '5d_change': ratio_5d_change,
                    '20d_change': ratio_20d_change
                })
            
            # Check for ratio reversals
            if (current_ratio < latest['ratio_sma_20'] and
                latest['ratio_sma_5'] < latest['ratio_sma_20'] and
                ratio_5d_change < -2):
                signals['ratio_reversals'].append({
                    'sector': sector_name,
                    'signal': 'BEARISH_REVERSAL',
                    'current_ratio': current_ratio,
                    '5d_change': ratio_5d_change,
                    '20d_change': ratio_20d_change
                })
            
            # Strong momentum alerts
            if abs(ratio_5d_change) > 5:
                direction = "UP" if ratio_5d_change > 0 else "DOWN"
                signals['alerts'].append({
                    'sector': sector_name,
                    'alert': f'STRONG_MOMENTUM_{direction}',
                    'magnitude': abs(ratio_5d_change),
                    'direction': direction
                })
        
        return signals
    
    def export_to_csv(self, ratios, signals, filename_prefix="sector_analysis"):
        """Export analysis results to CSV files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export ratios
            ratios_file = f"{filename_prefix}_ratios_{timestamp}.csv"
            with open(ratios_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Sector', 'Ratio', 'Ratio_Change_Pct', 'SMA_5', 'SMA_20'])
                
                for sector_name, data in ratios.items():
                    for _, row in data.iterrows():
                        writer.writerow([
                            row['timestamp'].strftime('%Y-%m-%d'),
                            sector_name,
                            round(row['ratio'], 4),
                            round(row['ratio_change'], 2) if pd.notna(row['ratio_change']) else '',
                            round(row['ratio_sma_5'], 4) if pd.notna(row['ratio_sma_5']) else '',
                            round(row['ratio_sma_20'], 4) if pd.notna(row['ratio_sma_20']) else ''
                        ])
            
            # Export signals
            signals_file = f"{filename_prefix}_signals_{timestamp}.csv"
            with open(signals_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Signal_Type', 'Sector', 'Signal', 'Details'])
                
                for signal_type, signal_list in signals.items():
                    for signal in signal_list:
                        details = "; ".join([f"{k}: {v}" for k, v in signal.items() if k not in ['sector']])
                        writer.writerow([
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            signal_type,
                            signal.get('sector', ''),
                            signal.get('signal', signal.get('alert', '')),
                            details
                        ])
            
            print(f"\n📊 Exported to:")
            print(f"   Ratios: {ratios_file}")
            print(f"   Signals: {signals_file}")
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    def run_enhanced_analysis(self, days_back=90, export_csv=True):
        """Run enhanced sector rotation analysis"""
        try:
            print("=" * 60)
            print("ENHANCED SECTOR ROTATION ANALYSIS")
            print("=" * 60)
            
            self.connect_database()
            
            # Calculate sector ratios
            ratios = self.calculate_sector_ratios(days_back)
            if not ratios:
                print("Error: Could not calculate sector ratios")
                return None
            
            # Generate enhanced signals
            print("\nGenerating enhanced rotation signals...")
            signals = self.generate_enhanced_signals(ratios)
            
            # Display enhanced results
            self.display_enhanced_results(ratios, signals)
            
            # Export to CSV if requested
            if export_csv:
                self.export_to_csv(ratios, signals)
            
            self.db.close_connection()
            return {'ratios': ratios, 'signals': signals}
            
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            return None
    
    def display_enhanced_results(self, ratios, signals):
        """Display enhanced analysis results"""
        print("\n" + "=" * 50)
        print("DAILY SECTOR RATIOS (Latest)")
        print("=" * 50)
        
        print(f"{'Sector':<15} {'Current Ratio':<12} {'5D Change':<10} {'20D Change':<11} {'Trend':<10}")
        print("-" * 60)
        
        for sector_name, data in ratios.items():
            if len(data) > 0:
                latest = data.iloc[-1]
                prev_5d = data.iloc[-6] if len(data) >= 6 else data.iloc[0]
                prev_20d = data.iloc[-21] if len(data) >= 21 else data.iloc[0]
                
                current_ratio = latest['ratio']
                ratio_5d_change = ((current_ratio - prev_5d['ratio']) / prev_5d['ratio']) * 100
                ratio_20d_change = ((current_ratio - prev_20d['ratio']) / prev_20d['ratio']) * 100
                
                # Trend determination
                if latest['ratio_sma_5'] > latest['ratio_sma_20']:
                    trend = "UP" if pd.notna(latest['ratio_sma_5']) and pd.notna(latest['ratio_sma_20']) else "FLAT"
                else:
                    trend = "DOWN" if pd.notna(latest['ratio_sma_5']) and pd.notna(latest['ratio_sma_20']) else "FLAT"
                
                print(f"{sector_name:<15} {current_ratio:<12.4f} {ratio_5d_change:>8.2f}% {ratio_20d_change:>9.2f}% {trend:<10}")
        
        # Display signals
        print("\n" + "=" * 50)
        print("ROTATION SIGNALS & ALERTS")
        print("=" * 50)
        
        if signals['momentum_breakouts']:
            print("\n🚀 MOMENTUM BREAKOUTS (BULLISH):")
            for signal in signals['momentum_breakouts']:
                print(f"   {signal['sector']}: Ratio breakout with {signal['5d_change']:.1f}% 5-day gain")
        
        if signals['ratio_reversals']:
            print("\n📉 RATIO REVERSALS (BEARISH):")
            for signal in signals['ratio_reversals']:
                print(f"   {signal['sector']}: Ratio reversal with {signal['5d_change']:.1f}% 5-day decline")
        
        if signals['alerts']:
            print("\n⚠️  MOMENTUM ALERTS:")
            for alert in signals['alerts']:
                print(f"   {alert['sector']}: {alert['alert']} - {alert['magnitude']:.1f}% move")
        
        if not any([signals['momentum_breakouts'], signals['ratio_reversals'], signals['alerts']]):
            print("\n📊 No significant rotation signals detected at this time.")
        
        # Summary recommendations
        print("\n" + "=" * 50)
        print("ACTION ITEMS")
        print("=" * 50)
        
        print("\n📋 Today's Focus:")
        strong_signals = len(signals['momentum_breakouts']) + len(signals['ratio_reversals'])
        
        if strong_signals > 0:
            print(f"   • {strong_signals} sectors showing strong rotation signals")
            print("   • Consider portfolio rebalancing")
            print("   • Monitor intraday for entry/exit points")
        else:
            print("   • No immediate rotation required")
            print("   • Continue monitoring daily ratios")
        
        if signals['alerts']:
            print(f"   • {len(signals['alerts'])} sectors with high momentum - watch for continuation")

def main():
    """Main execution function"""
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'assetmanagement'
    }
    
    analyzer = EnhancedSectorRotationAnalyzer(db_config)
    results = analyzer.run_enhanced_analysis(days_back=90, export_csv=True)
    
    if results:
        print(f"\n🎉 Enhanced analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nFiles generated for further analysis and record keeping.")
    else:
        print("❌ Analysis failed - check error messages above")

if __name__ == "__main__":
    main()