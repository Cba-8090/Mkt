#!/usr/bin/env python3
"""
Simple HYG Alert System
Main application script for running the alert system
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add src directory to path if it exists
src_path = os.path.join(os.path.dirname(__file__), 'src')
if os.path.exists(src_path):
    sys.path.append(src_path)

from database import HYGDatabase
from config import AlertConfig
from data_loader import DataLoader
from alert_engine import AlertEngine
from reports import ReportGenerator


def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/hyg_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_historical_data(db, data_loader):
    """Load historical data from CSV files"""
    print("📁 Loading historical data from CSV files...")

    if data_loader.load_csv_data():
        summary = db.get_data_summary()
        print(f"✅ Successfully loaded {summary.get('total_records', 0)} records")
        print(f"📅 Date range: {summary.get('start_date')} to {summary.get('end_date')}")
        return True
    else:
        print("❌ Failed to load historical data")
        return False


def update_daily_data(data_loader):
    """Update with latest daily data using FRED CSV downloads"""
    print("📊 Updating daily market data using FRED CSV downloads...")
    print("🔗 Using direct FRED CSV downloads (no API key needed)")

    if data_loader.update_daily_data():
        print("✅ Daily data updated successfully")
        return True
    else:
        print("❌ Failed to update daily data")
        return False


def run_analysis(alert_engine, report_generator):
    """Run complete alert analysis"""
    print("🔍 Running alert analysis...")

    # Run analysis
    results = alert_engine.run_alert_analysis()

    if results.get('status') == 'error':
        print(f"❌ Analysis failed: {results.get('message')}")
        return None

    # Print summary
    current_spread = results.get('current_spread', 0)
    alert_level = results.get('alert_level', 'UNKNOWN')
    alerts_count = len(results.get('alerts_generated', []))
    patterns_count = len(results.get('patterns_detected', []))

    print(f"📊 Current HYG Spread: {current_spread:.2f}%")
    print(f"🚨 Alert Level: {alert_level}")
    print(f"⚡ Alerts Generated: {alerts_count}")
    print(f"🔍 Patterns Detected: {patterns_count}")

    # Show alerts
    if alerts_count > 0:
        print("\n🚨 ALERTS:")
        for alert in results['alerts_generated']:
            urgency = alert.get('urgency', 'UNKNOWN')
            action = alert.get('recommended_action', 'Monitor')
            print(f"   • {urgency}: {action}")

    # Show patterns
    if patterns_count > 0:
        print("\n🔍 PATTERNS DETECTED:")
        for pattern in results['patterns_detected']:
            pattern_name = pattern.get('pattern_type', 'unknown').replace('_', ' ').title()
            confidence = pattern.get('confidence', 0) * 100
            print(f"   • {pattern_name} ({confidence:.0f}% confidence)")

    return results


def generate_reports(report_generator, results, output_dir='reports'):
    """Generate and save reports"""
    print("📄 Generating reports...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate daily report
    daily_report = report_generator.generate_daily_report()
    daily_file = os.path.join(output_dir, f"daily_report_{datetime.now().strftime('%Y%m%d')}.md")

    with open(daily_file, 'w', encoding='utf-8') as f:
        f.write(daily_report)
    print(f"📝 Daily report saved: {daily_file}")

    # Generate action plan if alerts exist
    if results and results.get('alerts_generated'):
        action_plan = report_generator.generate_action_plan(results)
        action_file = os.path.join(output_dir, f"action_plan_{datetime.now().strftime('%Y%m%d')}.md")

        with open(action_file, 'w', encoding='utf-8') as f:
            f.write(action_plan)
        print(f"🎯 Action plan saved: {action_file}")

    # Generate email summary
    if results:
        email_summary = report_generator.generate_summary_email(results)
        email_file = os.path.join(output_dir, f"email_summary_{datetime.now().strftime('%Y%m%d')}.txt")

        with open(email_file, 'w', encoding='utf-8') as f:
            f.write(f"Subject: {email_summary['subject']}\n\n")
            f.write(email_summary['body'])
        print(f"📧 Email summary saved: {email_file}")


def show_system_status(db, data_loader):
    """Show comprehensive system status"""
    print("🔍 HYG Alert System Status")
    print("=" * 50)

    try:
        # Database status
        summary = db.get_data_summary()
        latest_data = db.get_latest_data()

        print(f"📊 Database Status:")
        print(f"   Total records: {summary.get('total_records', 0):,}")
        print(f"   Date range: {summary.get('start_date', 'N/A')} to {summary.get('end_date', 'N/A')}")
        print(f"   Total alerts: {summary.get('total_alerts', 0):,}")

        if latest_data:
            latest_date = datetime.strptime(latest_data['date'], '%Y-%m-%d').date()
            days_old = (datetime.now().date() - latest_date).days

            print(f"\n📈 Latest Market Data ({latest_data['date']}):")
            print(
                f"   HYG Spread: {latest_data.get('hyg_spread', 'N/A')}% {'✅' if latest_data.get('hyg_spread') else '❌'}")
            print(f"   HY Yield: {latest_data.get('hy_yield', 'N/A')}% {'✅' if latest_data.get('hy_yield') else '❌'}")
            print(
                f"   10Y Treasury: {latest_data.get('treasury_10y', 'N/A')}% {'✅' if latest_data.get('treasury_10y') else '❌'}")
            print(f"   Data age: {days_old} days {'✅' if days_old <= 3 else '⚠️' if days_old <= 7 else '❌'}")

            # Quick risk assessment
            if latest_data.get('hyg_spread'):
                spread = latest_data['hyg_spread']
                print(f"\n🚨 Quick Risk Assessment:")

                if spread < 2.8:
                    print("   🚨 EXTREME DANGER - Immediate action required")
                elif spread < 3.2:
                    print("   🔴 HIGH RISK - Defensive positioning recommended")
                elif spread < 4.5:
                    print("   🟡 WATCH - Monitor closely")
                elif spread > 7.0:
                    print("   💎 OPPORTUNITY - Consider buying quality assets")
                else:
                    print("   🟢 NORMAL - Standard monitoring")
        else:
            print(f"\n❌ No market data available")

        # Data validation
        validation = data_loader.validate_data_integrity()
        quality_score = validation.get('data_quality_score', 0)

        print(f"\n📊 Data Quality:")
        print(
            f"   Quality score: {quality_score:.1%} {'✅' if quality_score > 0.8 else '⚠️' if quality_score > 0.6 else '❌'}")

        if validation.get('issues'):
            print(f"   Issues: {len(validation['issues'])}")
            for issue in validation['issues'][:3]:  # Show first 3
                print(f"     • {issue}")

        if validation.get('warnings'):
            print(f"   Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings'][:3]:  # Show first 3
                print(f"     • {warning}")

        # Recommendations
        print(f"\n💡 Recommendations:")

        if days_old > 3:
            print("   • Run daily update: python main.py update")
        if summary.get('total_records', 0) < 100:
            print("   • Load historical data: python main.py load")
        if quality_score < 0.8:
            print("   • Check data sources and validate data quality")
        if not latest_data or not latest_data.get('hyg_spread'):
            print("   • Ensure HYG spread data is available")

        return True

    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


def run_full_workflow(db, data_loader, alert_engine, report_generator):
    """Run complete workflow: update -> analyze -> report"""
    print("🚀 Running Full HYG Alert Workflow")
    print("=" * 50)

    # Step 1: Update data
    print("\n📊 Step 1: Updating market data...")
    update_success = update_daily_data(data_loader)

    if not update_success:
        print("⚠️ Data update failed, continuing with existing data...")

    # Step 2: Run analysis
    print("\n🔍 Step 2: Running alert analysis...")
    results = run_analysis(alert_engine, report_generator)

    if not results:
        print("❌ Analysis failed, cannot continue")
        return False

    # Step 3: Generate reports
    print("\n📄 Step 3: Generating reports...")
    generate_reports(report_generator, results)

    print("\n🎉 Full workflow completed successfully!")
    print("📂 Check the reports/ directory for generated files")

    return True


def main():
    """Main function with HTML report integration"""
    parser = argparse.ArgumentParser(
        description='HYG Alert System - Market Intelligence and Risk Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py load                    # Load historical CSV data
  python main.py update                  # Update with latest data  
  python main.py analyze                 # Run alert analysis with HTML reports
  python main.py full                    # Complete workflow with HTML reports
  python main.py status                  # Check system status
  python main.py html                    # Generate HTML report only

Commands:
  load      Load historical data from CSV files
  update    Update with latest market data using FRED CSV
  analyze   Run alert analysis and generate reports (includes HTML)
  full      Run complete workflow (update + analyze + HTML reports)
  status    Show comprehensive system status
  html      Generate interactive HTML report only
        """
    )

    parser.add_argument('command',
                        choices=['load', 'update', 'analyze', 'full', 'status', 'html'],
                        help='Command to execute')

    parser.add_argument('--db-path',
                        type=str,
                        default='data/hyg_data.db',
                        help='Database path (default: data/hyg_data.db)')

    parser.add_argument('--log-level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help='Logging level (default: INFO)')

    parser.add_argument('--no-reports',
                        action='store_true',
                        help='Skip report generation (analyze command only)')

    parser.add_argument('--no-html',
                        action='store_true',
                        help='Skip HTML report generation')

    parser.add_argument('--reports-dir',
                        type=str,
                        default='reports',
                        help='Reports output directory (default: reports)')

    parser.add_argument('--open-browser',
                        action='store_true',
                        help='Auto-open HTML report in browser')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        print(f"🔧 HYG Alert System Starting...")
        print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Database: {args.db_path}")
        print(f"📋 Command: {args.command}")
        print("")

        # Initialize components
        db = HYGDatabase(args.db_path)
        data_loader = DataLoader(db)
        alert_engine = AlertEngine(db)
        report_generator = ReportGenerator(db, alert_engine)

        # Execute command
        if args.command == 'load':
            success = load_historical_data(db, data_loader)

        elif args.command == 'update':
            success = update_daily_data(data_loader)

        elif args.command == 'analyze':
            results = run_analysis(alert_engine, report_generator)
            if results and not args.no_reports:
                generate_reports_with_html(report_generator, results, args.reports_dir, args.no_html, args.open_browser)
            success = results is not None

        elif args.command == 'full':
            success = run_full_workflow_with_html(db, data_loader, alert_engine, report_generator, args.reports_dir,
                                                  args.no_html, args.open_browser)

        elif args.command == 'status':
            success = show_system_status(db, data_loader)

        elif args.command == 'html':
            # Generate HTML report only
            results = run_analysis(alert_engine, report_generator)
            if results:
                html_file = generate_html_only(report_generator, results, args.reports_dir, args.open_browser)
                success = html_file is not None
            else:
                success = False

        else:
            print(f"❌ Unknown command: {args.command}")
            success = False

        # Exit with appropriate code
        if success:
            print(f"\n✅ Command '{args.command}' completed successfully!")
            return 0
        else:
            print(f"\n❌ Command '{args.command}' failed!")
            return 1

    except KeyboardInterrupt:
        print(f"\n🛑 Operation cancelled by user")
        return 1

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


def generate_reports_with_html(report_generator, results, output_dir='reports', no_html=False, open_browser=False):
    """Generate all reports including HTML with charts"""
    print("📄 Generating comprehensive reports...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate Markdown daily report (existing)
    daily_report = report_generator.generate_daily_report()
    daily_file = os.path.join(output_dir, f"daily_report_{datetime.now().strftime('%Y%m%d')}.md")

    with open(daily_file, 'w', encoding='utf-8') as f:
        f.write(daily_report)
    print(f"📝 Daily report saved: {daily_file}")

    # Generate HTML report with charts (new)
    if not no_html:
        try:
            # Check if HTML generator is available
            try:
                from html_report_generator import HTMLReportGenerator

                html_generator = HTMLReportGenerator(report_generator.db, report_generator.alert_engine,
                                                     report_generator.config)
                html_file = html_generator.save_html_report(output_dir)

                print(f"🌐 HTML report saved: {html_file}")
                print(f"   📂 Open in browser: file://{os.path.abspath(html_file)}")

                # Auto-open in browser if requested
                if open_browser:
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(html_file)}")
                        print(f"   🚀 Opened in default browser")
                    except Exception as e:
                        print(f"   ⚠️ Could not auto-open browser: {e}")

            except ImportError:
                print(f"⚠️ HTML report generator not available (html_report_generator.py not found)")
            except Exception as e:
                print(f"⚠️ HTML report generation failed: {e}")
        except Exception as e:
            print(f"⚠️ Error generating HTML report: {e}")

    # Generate action plan if alerts exist (existing)
    if results and results.get('alerts_generated'):
        action_plan = report_generator.generate_action_plan(results)
        action_file = os.path.join(output_dir, f"action_plan_{datetime.now().strftime('%Y%m%d')}.md")

        with open(action_file, 'w', encoding='utf-8') as f:
            f.write(action_plan)
        print(f"🎯 Action plan saved: {action_file}")

    # Generate email summary (existing)
    if results:
        email_summary = report_generator.generate_summary_email(results)
        email_file = os.path.join(output_dir, f"email_summary_{datetime.now().strftime('%Y%m%d')}.txt")

        with open(email_file, 'w', encoding='utf-8') as f:
            f.write(f"Subject: {email_summary['subject']}\n\n")
            f.write(email_summary['body'])
        print(f"📧 Email summary saved: {email_file}")


def generate_html_only(report_generator, results, output_dir='reports', open_browser=False):
    """Generate HTML report only"""
    print("🌐 Generating HTML report with interactive charts...")

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Import and use HTML generator
        from html_report_generator import HTMLReportGenerator

        html_generator = HTMLReportGenerator(report_generator.db, report_generator.alert_engine,
                                             report_generator.config)
        html_file = html_generator.save_html_report(output_dir)

        print(f"✅ HTML report generated: {html_file}")
        print(f"📂 Open in browser: file://{os.path.abspath(html_file)}")

        # Auto-open in browser if requested
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(html_file)}")
                print(f"🚀 Opened in default browser")
            except Exception as e:
                print(f"⚠️ Could not auto-open browser: {e}")

        return html_file

    except ImportError:
        print(f"❌ HTML report generator not available")
        print(f"💡 Please ensure html_report_generator.py is in your project directory")
        return None
    except Exception as e:
        print(f"❌ HTML report generation failed: {e}")
        return None


def run_full_workflow_with_html(db, data_loader, alert_engine, report_generator, reports_dir='reports', no_html=False,
                                open_browser=False):
    """Run complete workflow with HTML reports"""
    print("🚀 Running Full HYG Alert Workflow with HTML Reports")
    print("=" * 60)

    # Step 1: Update data
    print("\n📊 Step 1: Updating market data...")
    update_success = update_daily_data(data_loader)

    if not update_success:
        print("⚠️ Data update failed, continuing with existing data...")

    # Step 2: Run analysis
    print("\n🔍 Step 2: Running alert analysis...")
    results = run_analysis(alert_engine, report_generator)

    if not results:
        print("❌ Analysis failed, cannot continue")
        return False

    # Step 3: Generate all reports including HTML
    print("\n📄 Step 3: Generating comprehensive reports...")
    generate_reports_with_html(report_generator, results, reports_dir, no_html, open_browser)

    print("\n🎉 Full workflow completed successfully!")
    print("📂 Check the reports/ directory for generated files:")
    print(f"   • Markdown report: daily_report_{datetime.now().strftime('%Y%m%d')}.md")
    if not no_html:
        print(f"   • HTML report: hyg_report_{datetime.now().strftime('%Y%m%d')}.html")
    if results.get('alerts_generated'):
        print(f"   • Action plan: action_plan_{datetime.now().strftime('%Y%m%d')}.md")
    print(f"   • Email summary: email_summary_{datetime.now().strftime('%Y%m%d')}.txt")

    return True


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)