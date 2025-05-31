#!/usr/bin/env python3
r"""
Database Column Checker and Fixer
Path: C:\Projects\apps\SectorAnalysis\check_columns.py

This script checks what columns exist in your table and fixes the queries accordingly.
"""

import sys
sys.path.append(r'C:\Projects\library\mysql')

def check_table_structure():
    """Check the actual table structure"""
    try:
        from mydbmanager import DatabaseManager
        
        print("Checking table structure...")
        
        db = DatabaseManager(
            host='localhost',
            user='root',
            password='',
            database='assetmanagement'
        )
        
        db.connect()
        
        # Check table structure
        result = db.execute_query("DESCRIBE _nse_sectorindex", fetch=True)
        
        if result:
            print("\n📋 TABLE STRUCTURE:")
            print("-" * 50)
            lines = result.split('#r#')
            headers = lines[0].split('|')
            print(f"{'Column':<20} {'Type':<15} {'Null':<5} {'Key':<5} {'Default':<10}")
            print("-" * 60)
            
            available_columns = []
            for line in lines[1:]:
                if line.strip():
                    parts = line.split('|')
                    column_name = parts[0]
                    available_columns.append(column_name)
                    
                    column_type = parts[1] if len(parts) > 1 else ''
                    null_allowed = parts[2] if len(parts) > 2 else ''
                    key = parts[3] if len(parts) > 3 else ''
                    default = parts[4] if len(parts) > 4 else ''
                    
                    print(f"{column_name:<20} {column_type:<15} {null_allowed:<5} {key:<5} {default:<10}")
            
            print(f"\n📊 Available columns: {', '.join(available_columns)}")
            
            # Check for common column variations
            print("\n🔍 COLUMN ANALYSIS:")
            
            # Check for price-related columns
            price_columns = [col for col in available_columns if any(word in col.lower() for word in ['close', 'price', 'value'])]
            print(f"   Price columns: {price_columns}")
            
            # Check for change-related columns
            change_columns = [col for col in available_columns if any(word in col.lower() for word in ['change', 'pct', 'percent', 'return'])]
            print(f"   Change columns: {change_columns}")
            
            # Check for date/time columns
            date_columns = [col for col in available_columns if any(word in col.lower() for word in ['date', 'time', 'timestamp'])]
            print(f"   Date columns: {date_columns}")
            
            # Check for volume columns
            volume_columns = [col for col in available_columns if any(word in col.lower() for word in ['volume', 'vol', 'qty'])]
            print(f"   Volume columns: {volume_columns}")
            
            # Sample data
            print("\n📋 SAMPLE DATA:")
            result = db.execute_query("SELECT * FROM _nse_sectorindex WHERE sector = 'Nifty 50' LIMIT 3", fetch=True)
            
            if result:
                lines = result.split('#r#')
                headers = lines[0].split('|')
                print("\nHeaders:", headers)
                for i, line in enumerate(lines[1:4]):  # Show 3 sample rows
                    if line.strip():
                        values = line.split('|')
                        print(f"Row {i+1}:", values)
            
            db.close_connection()
            return available_columns
            
    except Exception as e:
        print(f"Error checking table structure: {e}")
        return []

def create_fixed_analyzer():
    """Create a fixed version of the analyzer based on actual columns"""
    
    # First check what columns we have
    columns = check_table_structure()
    
    if not columns:
        print("❌ Could not determine table structure")
        return
    
    # Determine column mappings
    timestamp_col = None
    close_col = None
    change_col = None
    volume_col = None
    
    # Find the right columns
    for col in columns:
        if 'timestamp' in col.lower() or 'date' in col.lower():
            timestamp_col = col
        elif 'close' in col.lower() or ('price' in col.lower() and 'close' not in col.lower()):
            close_col = col
        elif any(word in col.lower() for word in ['change', 'pct', 'percent', 'return']):
            change_col = col
        elif 'volume' in col.lower():
            volume_col = col
    
    print(f"\n🔧 DETECTED COLUMN MAPPINGS:")
    print(f"   Timestamp: {timestamp_col}")
    print(f"   Close Price: {close_col}")
    print(f"   Price Change: {change_col}")
    print(f"   Volume: {volume_col}")
    
    if not timestamp_col or not close_col:
        print("❌ Missing essential columns (timestamp and close price)")
        return
    
    # Create fixed analyzer
    fixed_analyzer_code = f'''#!/usr/bin/env python3
"""
Fixed Sector Rotation Analyzer
Auto-generated based on your actual database columns
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the mysql library path
sys.path.append(r'C:\\Projects\\library\\mysql')
from mydbmanager import DatabaseManager

class FixedSectorRotationAnalyzer:
    def __init__(self, db_config):
        self.db = DatabaseManager(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        
        # Column mappings based on your actual table
        self.timestamp_col = '{timestamp_col}'
        self.close_col = '{close_col}'
        self.change_col = '{change_col}' if '{change_col}' != 'None' else None
        self.volume_col = '{volume_col}' if '{volume_col}' != 'None' else None
        
        # Define sector mappings
        self.nifty_sectors = [
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
            print(f"Failed to connect to database: {{e}}")
            raise
    
    def get_nifty_data(self, days_back=365):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Build query with available columns
            columns = [self.timestamp_col, self.close_col]
            if self.change_col:
                columns.append(self.change_col)
            if self.volume_col:
                columns.append(self.volume_col)
            
            columns_str = ', '.join(columns)
            
            query = f"""
            SELECT {{columns_str}}
            FROM _nse_sectorindex 
            WHERE sector = 'Nifty 50' 
            AND {{self.timestamp_col}} >= '{{start_date.strftime('%Y-%m-%d')}}'
            ORDER BY {{self.timestamp_col}} ASC
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
                if line.strip():
                    values = line.split('|')
                    data.append(values)
            
            df = pd.DataFrame(data, columns=header)
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df[self.close_col] = pd.to_numeric(df[self.close_col], errors='coerce')
            
            # Calculate returns if change column doesn't exist
            if self.change_col and self.change_col in df.columns:
                df[self.change_col] = pd.to_numeric(df[self.change_col], errors='coerce')
            else:
                df['calculated_return'] = df[self.close_col].pct_change() * 100
                self.change_col = 'calculated_return'
            
            if self.volume_col and self.volume_col in df.columns:
                df[self.volume_col] = pd.to_numeric(df[self.volume_col], errors='coerce')
            
            df = df.dropna(subset=[self.close_col])
            print(f"Fetched {{len(df)}} records for Nifty 50")
            return df
            
        except Exception as e:
            print(f"Error fetching Nifty data: {{e}}")
            return None
    
    def run_simple_analysis(self, days_back=90):
        """Run a simplified analysis"""
        try:
            print("Starting Fixed Sector Rotation Analysis...")
            print("=" * 50)
            
            self.connect_database()
            
            print("1. Fetching Nifty 50 data...")
            nifty_data = self.get_nifty_data(days_back)
            
            if nifty_data is None or nifty_data.empty:
                print("Error: No Nifty data available")
                return None
            
            print("2. Calculating basic performance metrics...")
            
            # Calculate simple performance metrics
            latest_price = nifty_data[self.close_col].iloc[-1]
            earliest_price = nifty_data[self.close_col].iloc[0]
            total_return = ((latest_price - earliest_price) / earliest_price) * 100
            
            print(f"📈 NIFTY 50 PERFORMANCE (Last {{days_back}} days):")
            print(f"   Start Price: {{earliest_price:.2f}}")
            print(f"   Current Price: {{latest_price:.2f}}")
            print(f"   Total Return: {{total_return:.2f}}%")
            
            # Calculate volatility
            if self.change_col in nifty_data.columns:
                volatility = nifty_data[self.change_col].std()
                print(f"   Volatility: {{volatility:.2f}}%")
            
            print("\\n✅ Analysis completed successfully!")
            print("\\nNext steps:")
            print("1. This confirms your database connection works")
            print("2. We can now build the full sector rotation analysis")
            print("3. Run: python fixed_analyzer.py")
            
            self.db.close_connection()
            return {{
                'nifty_data': nifty_data,
                'performance': {{
                    'total_return': total_return,
                    'latest_price': latest_price,
                    'start_price': earliest_price
                }}
            }}
            
        except Exception as e:
            print(f"Error in analysis: {{e}}")
            return None

def main():
    db_config = {{
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'assetmanagement'
    }}
    
    analyzer = FixedSectorRotationAnalyzer(db_config)
    results = analyzer.run_simple_analysis(days_back=90)
    
    if results:
        print("🎉 Fixed analyzer working perfectly!")
    else:
        print("❌ Still having issues - check error messages above")

if __name__ == "__main__":
    main()
'''
    
    # Write the fixed analyzer
    with open('fixed_analyzer.py', 'w') as f:
        f.write(fixed_analyzer_code)
    
    print(f"\n✅ Created fixed_analyzer.py with correct column mappings!")
    print("\nRun it with: python fixed_analyzer.py")

def main():
    print("=" * 60)
    print("DATABASE COLUMN CHECKER AND FIXER")
    print("=" * 60)
    
    create_fixed_analyzer()

if __name__ == "__main__":
    main()