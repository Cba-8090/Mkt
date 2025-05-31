#!/usr/bin/env python3
"""
Quick Start Runner for Sector Rotation Analysis
Path: C:\Projects\apps\SectorAnalysis\run_analysis.py

This script provides a simple way to run the sector rotation analysis
with your existing MySQL database setup.
"""

import sys
import os
from datetime import datetime

# Add the mysql library path
sys.path.append(r'C:\Projects\library\mysql')

# Import our modules
from sector_rotation_analyzer import SectorRotationAnalyzer
from config import Config

def setup_environment():
    """Setup the environment and directories"""
    # Create output directory if it doesn't exist
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        print(f"Created output directory: {Config.OUTPUT_DIR}")
    
    # Change to output directory for file exports
    os.chdir(Config.OUTPUT_DIR)
    print(f"Working directory: {os.getcwd()}")

def update_database_config():
    """
    Update database configuration interactively
    You can modify this or update config.py directly
    """
    print("Current Database Configuration:")
    print(f"Host: {Config.DATABASE_CONFIG['host']}")
    print(f"User: {Config.DATABASE_CONFIG['user']}")
    print(f"Database: {Config.DATABASE_CONFIG['database']}")
    print()
    
    # You can uncomment these lines for interactive configuration
    # Config.DATABASE_CONFIG['host'] = input("Enter MySQL host (press Enter for localhost): ") or 'localhost'
    # Config.DATABASE_CONFIG['user'] = input("Enter MySQL username: ")
    # Config.DATABASE_CONFIG['password'] = input("Enter MySQL password: ")
    # Config.DATABASE_CONFIG['database'] = input("Enter database name: ")
    
    # Or update directly here:
    Config.DATABASE_CONFIG.update({
        'host': 'localhost',
        'user': 'root',  # Update with your MySQL username
        'password': 'your_password',  # Update with your MySQL password
        'database': 'nse_data'  # Update with your database name
    })

def run_quick_analysis():
    """Run a quick analysis with default settings"""
    print("=" * 60)
    print("SECTOR ROTATION ANALYSIS - QUICK START")
    print("=" * 60)
    
    try:
        # Setup environment
        setup_environment()
        
        # Update database config
        update_database_config()
        
        # Initialize analyzer
        print("Initializing Sector Rotation Analyzer...")
        analyzer = SectorRotationAnalyzer(Config.DATABASE_CONFIG)
        
        # Run analysis
        print("Starting analysis...")
        results = analyzer.run_analysis(days_back=180, export_csv=True)
        
        if results:
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Print key findings
            rotation_signals = results['rotation_signals']
            summary = rotation_signals.get('summary', {})
            
            print(f"✅ Analyzed {summary.get('total_sectors_analyzed', 0)} sectors")
            print(f"🔄 Found {summary.get('rotation_opportunities', 0)} rotation opportunities")
            print(f"📈 {summary.get('strong_outperformers', 0)} strong outperforming sectors")
            print(f"📉 {summary.get('strong_underperformers', 0)} strong underperforming sectors")
            print(f"⚡ {summary.get('momentum_alerts', 0)} momentum alerts")
            
            # Show top rotation opportunities
            if rotation_signals.get('rotation_opportunities'):
                print("\n🎯 TOP ROTATION OPPORTUNITIES:")
                for i, opp in enumerate(rotation_signals['rotation_opportunities'][:3], 1):
                    print(f"   {i}. {opp['sector']} - {opp['type']}")
                    print(f"      {opp['description']}")
                    print(f"      Confidence: {opp['confidence']}")
                    print()
            
            # Show files created
            print("📁 FILES CREATED:")
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f"   - sector_relative_performance_{current_time}.csv")
            print(f"   - daily_sector_ratios_{current_time}.csv")
            print(f"   - sector_rotation_report_{current_time}.txt")
            
        else:
            print("❌ Analysis failed! Please check your database connection and data.")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your database connection settings in config.py")
        print("2. Ensure your MySQL server is running")
        print("3. Verify the table '_nse_sectorindex' exists and has data")
        print("4. Check that all required Python packages are installed")

def run_custom_analysis():
    """Run analysis with custom parameters"""
    print("=" * 60)
    print("SECTOR ROTATION ANALYSIS - CUSTOM RUN")
    print("=" * 60)
    
    try:
        # Get user preferences
        days_back = int(input("Enter number of days to analyze (default 365): ") or 365)
        export_files = input("Export CSV files? (y/n, default y): ").lower() != 'n'
        
        # Setup environment
        setup_environment()
        update_database_config()
        
        # Initialize analyzer
        analyzer = SectorRotationAnalyzer(Config.DATABASE_CONFIG)
        
        # Customize thresholds if needed
        print("\nCurrent rotation thresholds:")
        print(f"Short-term alert: {Config.ROTATION_THRESHOLDS['short_term']['alert']}%")
        print(f"Short-term strong: {Config.ROTATION_THRESHOLDS['short_term']['strong']}%")
        
        # Run analysis
        results = analyzer.run_analysis(days_back=days_back, export_csv=export_files)
        
        if results:
            print(f"\n✅ Custom analysis completed successfully!")
            
            # Option to create visualizations
            create_charts = input("\nCreate visualization charts? (y/n): ").lower() == 'y'
            
            if create_charts:
                try:
                    from sector_utils import VisualizationUtils
                    
                    print("Creating visualizations...")
                    
                    # Create heatmap
                    VisualizationUtils.plot_relative_performance_heatmap(
                        results['relative_performance'],
                        save_path=f"relative_performance_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    )
                    
                    # Create ratio trends
                    if not results['daily_ratios'].empty:
                        VisualizationUtils.plot_sector_ratios_trend(
                            results['daily_ratios'],
                            save_path=f"sector_ratios_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        )
                    
                    # Create rotation signals dashboard
                    VisualizationUtils.plot_rotation_signals_dashboard(
                        results['rotation_signals'],
                        save_path=f"rotation_signals_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    )
                    
                    print("📊 Visualization charts created successfully!")
                    
                except Exception as e:
                    print(f"⚠️ Error creating charts: {e}")
                    print("Charts require matplotlib and seaborn packages")
        
    except Exception as e:
        print(f"❌ Error during custom analysis: {e}")

def show_menu():
    """Show the main menu"""
    print("\n" + "=" * 60)
    print("SECTOR ROTATION ANALYSIS SYSTEM")
    print("=" * 60)
    print("1. Quick Analysis (Last 6 months)")
    print("2. Custom Analysis")
    print("3. View Database Tables")
    print("4. Test Database Connection")
    print("5. Exit")
    print("=" * 60)

def test_database_connection():
    """Test database connection and show available data"""
    try:
        from mydbmanager import DatabaseManager
        
        print("Testing database connection...")
        db = DatabaseManager(
            host=Config.DATABASE_CONFIG['host'],
            user=Config.DATABASE_CONFIG['user'],
            password=Config.DATABASE_CONFIG['password'],
            database=Config.DATABASE_CONFIG['database']
        )
        
        db.connect()
        print("✅ Database connection successful!")
        
        # Check if table exists and has data
        query = "SELECT COUNT(*) as count, MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM _nse_sectorindex"
        result = db.execute_query(query, fetch=True)
        
        if result:
            lines = result.split('#r#')
            if len(lines) > 1:
                data = lines[1].split('|')
                print(f"📊 Table '_nse_sectorindex' contains {data[0]} records")
                print(f"📅 Date range: {data[1]} to {data[2]}")
        
        # Check available sectors
        query = "SELECT DISTINCT sector FROM _nse_sectorindex ORDER BY sector"
        result = db.execute_query(query, fetch=True)
        
        if result:
            lines = result.split('#r#')
            sectors = [line.split('|')[0] for line in lines[1:] if line.strip()]
            print(f"🏢 Available sectors ({len(sectors)}):")
            for sector in sectors[:10]:  # Show first 10
                print(f"   - {sector}")
            if len(sectors) > 10:
                print(f"   ... and {len(sectors) - 10} more")
        
        db.close_connection()
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nPlease check:")
        print("1. MySQL server is running")
        print("2. Database credentials in config.py")
        print("3. Database and table exist")

def main():
    """Main function"""
    while True:
        show_menu()
        
        try:
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                run_quick_analysis()
            elif choice == '2':
                run_custom_analysis()
            elif choice == '3':
                test_database_connection()
            elif choice == '4':
                test_database_connection()
            elif choice == '5':
                print("👋 Thank you for using Sector Rotation Analysis System!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()