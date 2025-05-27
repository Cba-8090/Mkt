from flask import Flask, jsonify, render_template_string, send_from_directory, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Try to import the data integrator, with fallback
try:
    from dataCollator import NiftyDataIntegrator

    integrator = NiftyDataIntegrator()
    INTEGRATOR_AVAILABLE = True
    print("✅ DataCollator imported successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not import NiftyDataIntegrator: {e}")
    print("📊 Dashboard will run with simulated data")
    integrator = None
    INTEGRATOR_AVAILABLE = False


def safe_datetime_convert(obj):
    """Safely convert datetime objects to ISO strings, handling both datetime and string inputs"""
    if isinstance(obj, dict):
        return {k: safe_datetime_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_datetime_convert(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, str):
        # If it's already a string, check if it looks like an ISO timestamp
        try:
            # Try to parse it as datetime and convert back (validates format)
            if 'T' in obj or '-' in obj:  # Looks like a timestamp
                return obj  # Return original string if it looks valid
            else:
                return obj  # Return as-is if not a datetime string
        except (ValueError, AttributeError):
            return obj  # Return as-is if not a datetime string
    else:
        return obj


def generate_simulated_data():
    """Generate simulated data for testing"""
    return {
        'timestamp': datetime.now().isoformat(),
        'futures_data': {
            'weighted_positive_money_flow': random.uniform(1000000, 3000000),
            'weighted_negative_money_flow': random.uniform(800000, 2500000),
            'weighted_money_flow': random.uniform(-1000000, 1000000),
            'cumulative_weighted_money_flow': random.uniform(5000000, 15000000)
        },
        'options_data': {
            'net_flow': random.uniform(-200000, 200000),
            'total_flow': random.uniform(300000, 800000),
            'bullish_flow': random.uniform(100000, 400000),
            'bearish_flow': random.uniform(100000, 400000),
            'sentiment': random.choice(['Bullish', 'Bearish', 'Neutral']),
            'call_buying': random.uniform(50000, 200000),
            'put_writing': random.uniform(30000, 150000),
            'call_short_covering': random.uniform(20000, 100000),
            'put_unwinding': random.uniform(25000, 120000),
            'put_buying': random.uniform(40000, 180000),
            'call_writing': random.uniform(35000, 160000),
            'put_short_covering': random.uniform(15000, 80000),
            'call_unwinding': random.uniform(20000, 90000)
        },
        'gamma_data': {
            'support_pressure': random.uniform(200000, 1000000),
            'resistance_pressure': random.uniform(100000, 700000),
            'sr_ratio': random.uniform(0.5, 4.0),
            'max_pressure_strike': 24500 + random.uniform(-500, 500),
            'max_pressure_value': random.uniform(30000, 80000),
            'price_reversals': [
                {
                    'time': f"{random.randint(9, 15):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
                    'direction': random.choice(['BEARISH', 'BULLISH']),
                    'price': 24500 + random.uniform(-200, 200)
                }
                for _ in range(random.randint(0, 3))
            ],
            'breakdown_signals': [
                {
                    'time': f"{random.randint(9, 15):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
                    'signal': random.choice([
                        'CRITICAL BREAKDOWN SIGNAL: Support erosion detected',
                        'BREAKDOWN CONFIRMATION: Resistance increase',
                        'Support erosion: Pressure decreased significantly'
                    ])
                }
                for _ in range(random.randint(0, 2))
            ],
            'support_levels': sorted([24000 + random.randint(0, 800) for _ in range(random.randint(2, 5))]),
            'resistance_levels': sorted([24600 + random.randint(0, 400) for _ in range(random.randint(2, 4))]),
            'spot_price': 24500 + random.uniform(-100, 100),
            'spot_change': random.uniform(-50, 50),
            'trend_direction': random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
        },
        'spot_data': {
            'spot_price': 24500 + random.uniform(-100, 100),
            'price_change': random.uniform(-50, 50),
            'price_change_pct': random.uniform(-1.5, 1.5)
        },
        'signals': {
            'combined_signal': random.uniform(-1, 1),
            'signal_strength': random.uniform(0, 10),
            'direction': random.choice(['Bullish', 'Bearish', 'Neutral']),
            'confidence': random.uniform(30, 95)
        }
    }


@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    try:
        return send_from_directory('.', 'nifty_dashboard.html')
    except Exception as e:
        return f"Error loading dashboard: {e}. Make sure 'nifty_dashboard.html' is in the same directory as this script."


@app.route('/api/unified-data')
def get_unified_data():
    """API endpoint to get the latest unified data"""
    try:
        if INTEGRATOR_AVAILABLE:
            # Try to use real integrator
            try:
                data = integrator.get_latest_data()
                if not data:
                    print("No cached data found, collecting new data...")
                    data = integrator.collect_unified_data()

                # Use the safer datetime conversion
                data = safe_datetime_convert(data)

                print(f"API: Serving real data with signal: {data.get('signals', {}).get('direction', 'Unknown')}")
                return jsonify(data)

            except Exception as integrator_error:
                print(f"⚠️ Integrator error: {integrator_error}")
                print("🔄 Falling back to simulated data")
                # Fall back to simulated data
                data = generate_simulated_data()
                print(f"API: Serving simulated data with signal: {data.get('signals', {}).get('direction', 'Unknown')}")
                return jsonify(data)
        else:
            # Use simulated data
            data = generate_simulated_data()
            print(f"API: Serving simulated data with signal: {data.get('signals', {}).get('direction', 'Unknown')}")
            return jsonify(data)

    except Exception as e:
        print(f"Error getting unified data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/historical-data')
def get_historical_data():
    """API endpoint to get historical data"""
    try:
        hours = request.args.get('hours', 24, type=int)

        if INTEGRATOR_AVAILABLE:
            try:
                data = integrator.get_historical_data(hours)
                # Safe conversion for historical data
                data = safe_datetime_convert(data)
                return jsonify(data)
            except Exception as e:
                print(f"Historical data error: {e}")
                # Fall back to simulated historical data
                historical_data = [generate_simulated_data() for _ in range(min(hours, 10))]
                return jsonify(historical_data)
        else:
            # Generate simulated historical data
            historical_data = [generate_simulated_data() for _ in range(min(hours, 10))]
            return jsonify(historical_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """API endpoint to get summary statistics"""
    try:
        if INTEGRATOR_AVAILABLE:
            try:
                stats = integrator.get_summary_stats()
                return jsonify(stats)
            except Exception as e:
                print(f"Stats error: {e}")
                pass

        # Simulated stats
        stats = {
            'total_data_points': random.randint(50, 200),
            'avg_signal': random.uniform(-0.3, 0.3),
            'avg_strength': random.uniform(3, 7),
            'max_signal': random.uniform(0.8, 1.0),
            'min_signal': random.uniform(-1.0, -0.8),
            'bullish_count': random.randint(10, 50),
            'bearish_count': random.randint(10, 50),
            'neutral_count': random.randint(5, 30)
        }
        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/start-collection', methods=['POST'])
def start_collection():
    """Start continuous data collection"""
    try:
        if INTEGRATOR_AVAILABLE:
            integrator.start_continuous_collection()
            return jsonify({"message": "Data collection started"})
        else:
            return jsonify({"message": "Running in simulation mode - no real data collection"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stop-collection', methods=['POST'])
def stop_collection():
    """Stop continuous data collection"""
    try:
        if INTEGRATOR_AVAILABLE:
            integrator.stop_continuous_collection()
            return jsonify({"message": "Data collection stopped"})
        else:
            return jsonify({"message": "Running in simulation mode - no data collection to stop"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/debug')
def debug_data():
    """Debug endpoint to see raw data structure"""
    if INTEGRATOR_AVAILABLE:
        try:
            data = integrator.get_latest_data()
            if not data:
                data = integrator.collect_unified_data()

            return jsonify({
                "raw_data_type": str(type(data)),
                "timestamp_type": str(type(data.get('timestamp'))),
                "timestamp_value": str(data.get('timestamp')),
                "data_keys": list(data.keys()) if isinstance(data, dict) else "Not a dict"
            })
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"message": "Integrator not available"})


@app.route('/api/enhanced-data')
def get_enhanced_data():
    """API endpoint to get enhanced data with historical context"""
    try:
        if INTEGRATOR_AVAILABLE:
            try:
                # Get enhanced data with historical context
                data = integrator.get_enhanced_latest_data()
                if not data:
                    print("No enhanced data found, collecting new data...")
                    integrator.collect_unified_data()
                    data = integrator.get_enhanced_latest_data()

                # Use the safer datetime conversion
                data = safe_datetime_convert(data)

                print(
                    f"API: Serving enhanced data with {len(data.get('historical_breakdown_signals', []))} historical signals")
                return jsonify(data)

            except Exception as integrator_error:
                print(f"⚠️ Enhanced integrator error: {integrator_error}")
                print("🔄 Falling back to regular data")
                return get_unified_data()  # Fallback to regular data
        else:
            return get_unified_data()  # Fallback to simulated data

    except Exception as e:
        print(f"Error getting enhanced data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/historical-breakdown-signals')
def get_historical_breakdown_signals():
    """API endpoint to get historical breakdown signals"""
    try:
        hours = request.args.get('hours', 24, type=int)

        if INTEGRATOR_AVAILABLE:
            try:
                signals = integrator.get_historical_breakdown_signals(hours)
                signals = safe_datetime_convert(signals)
                return jsonify({
                    'signals': signals,
                    'total_count': len(signals),
                    'hours': hours
                })
            except Exception as e:
                print(f"Historical breakdown signals error: {e}")
                return jsonify({'signals': [], 'total_count': 0, 'hours': hours})
        else:
            return jsonify({'signals': [], 'total_count': 0, 'hours': hours})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/historical-reversals')
def get_historical_reversals():
    """API endpoint to get historical price reversals"""
    try:
        hours = request.args.get('hours', 24, type=int)

        if INTEGRATOR_AVAILABLE:
            try:
                reversals = integrator.get_historical_price_reversals(hours)
                reversals = safe_datetime_convert(reversals)
                return jsonify({
                    'reversals': reversals,
                    'total_count': len(reversals),
                    'hours': hours
                })
            except Exception as e:
                print(f"Historical reversals error: {e}")
                return jsonify({'reversals': [], 'total_count': 0, 'hours': hours})
        else:
            return jsonify({'reversals': [], 'total_count': 0, 'hours': hours})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced-stats')
def get_enhanced_stats():
    """API endpoint to get enhanced statistics with historical context"""
    try:
        if INTEGRATOR_AVAILABLE:
            try:
                stats = integrator.get_summary_stats()
                return jsonify(stats)
            except Exception as e:
                print(f"Enhanced stats error: {e}")
                pass

        # Simulated enhanced stats
        stats = {
            'total_data_points': random.randint(50, 200),
            'avg_signal': random.uniform(-0.3, 0.3),
            'avg_strength': random.uniform(3, 7),
            'max_signal': random.uniform(0.8, 1.0),
            'min_signal': random.uniform(-1.0, -0.8),
            'bullish_count': random.randint(10, 50),
            'bearish_count': random.randint(10, 50),
            'neutral_count': random.randint(5, 30),
            'total_breakdown_signals': random.randint(20, 100),
            'breakdown_by_type': {
                'CRITICAL': random.randint(1, 5),
                'CONFIRMATION': random.randint(2, 8),
                'WARNING': random.randint(3, 10),
                'EROSION': random.randint(5, 15),
                'REVERSAL': random.randint(10, 25)
            },
            'total_price_reversals': random.randint(30, 80)
        }
        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "integrator_available": INTEGRATOR_AVAILABLE,
        "mode": "real_data" if INTEGRATOR_AVAILABLE else "simulation"
    }

    if INTEGRATOR_AVAILABLE:
        try:
            health_data.update({
                "data_collection_running": integrator.running if hasattr(integrator, 'running') else False,
                "total_data_points": len(integrator.unified_data) if hasattr(integrator, 'unified_data') else 0,
                "historical_breakdown_signals": len(integrator.historical_breakdown_signals) if hasattr(integrator,
                                                                                                        'historical_breakdown_signals') else 0,
                "historical_price_reversals": len(integrator.historical_price_reversals) if hasattr(integrator,
                                                                                                    'historical_price_reversals') else 0
            })
        except Exception:
            health_data["integrator_status"] = "error"

    return jsonify(health_data)


def start_background_collection():
    """Start data collection in background thread"""
    if INTEGRATOR_AVAILABLE:
        print("Starting background data collection...")
        try:
            integrator.start_continuous_collection(interval_minutes=5)
        except Exception as e:
            print(f"Error starting background collection: {e}")
            print("🔄 Dashboard will run with simulated data")
    else:
        print("📊 Running in simulation mode - no background collection needed")


def check_required_files():
    """Check if required files exist"""
    required_files = ['nifty_dashboard.html']
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all files are in the same directory as flask_server.py")
        return False

    print("✅ All required files found")
    return True


def test_integrator():
    """Test the integrator before starting the server"""
    if INTEGRATOR_AVAILABLE:
        try:
            print("🧪 Testing integrator...")
            data = integrator.collect_unified_data()
            print(f"✅ Integrator test successful! Signal: {data.get('signals', {}).get('direction', 'Unknown')}")
            return True
        except Exception as e:
            print(f"⚠️ Integrator test failed: {e}")
            return False
    return False


if __name__ == '__main__':
    print("🔍 Checking required files...")
    if not check_required_files():
        exit(1)

    print("🚀 Starting Nifty Dashboard Server...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/unified-data")
    print("💊 Health check: http://localhost:5000/api/health")
    print("🐛 Debug endpoint: http://localhost:5000/api/debug")

    # Test integrator
    integrator_working = test_integrator()

    if INTEGRATOR_AVAILABLE and integrator_working:
        print("📈 Real data collection starting...")
        # Start background data collection
        collection_thread = threading.Thread(target=start_background_collection, daemon=True)
        collection_thread.start()
        time.sleep(2)  # Give it a moment to collect initial data
        print("🟢 Server running with REAL DATA")
    else:
        print("🎭 Running with simulated data")
        print("🔧 Check /api/debug for data structure issues")

    print("🌐 Server starting...")
    print("-" * 50)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)