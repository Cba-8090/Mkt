from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time

# Import your data integrator
from dataCollator import NiftyDataIntegrator

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the data integrator
integrator = NiftyDataIntegrator()

# Store the HTML template (you can also save it as a separate file)
DASHBOARD_HTML = """
<!-- Your complete HTML dashboard code goes here -->
<!-- Copy the entire HTML from the artifact -->
"""

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    # If you save HTML as a separate file, use:
    # return send_from_directory('.', 'nifty_dashboard.html')
    
    # Or render directly:
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/unified-data')
def get_unified_data():
    """API endpoint to get the latest unified data"""
    try:
        # Get latest data from your integrator
        data = integrator.get_latest_data()
        
        # If no data available, collect new data
        if not data:
            data = integrator.collect_unified_data()
        
        # Convert datetime objects to ISO strings for JSON serialization
        if data.get('timestamp'):
            data['timestamp'] = data['timestamp'].isoformat()
        
        # Convert any nested datetime objects
        def convert_datetime(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        data = convert_datetime(data)
        
        return jsonify(data)
    
    except Exception as e:
        print(f"Error getting unified data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical-data')
def get_historical_data():
    """API endpoint to get historical data"""
    try:
        hours = request.args.get('hours', 24, type=int)
        data = integrator.get_historical_data(hours)
        
        # Convert datetime objects
        for item in data:
            if item.get('timestamp'):
                item['timestamp'] = item['timestamp'].isoformat()
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint to get summary statistics"""
    try:
        stats = integrator.get_summary_stats()
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start-collection')
def start_collection():
    """Start continuous data collection"""
    try:
        integrator.start_continuous_collection()
        return jsonify({"message": "Data collection started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop-collection')
def stop_collection():
    """Stop continuous data collection"""
    try:
        integrator.stop_continuous_collection()
        return jsonify({"message": "Data collection stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_background_collection():
    """Start data collection in background thread"""
    print("Starting background data collection...")
    integrator.start_continuous_collection(interval_minutes=5)

if __name__ == '__main__':
    # Start background data collection
    collection_thread = threading.Thread(target=start_background_collection, daemon=True)
    collection_thread.start()
    
    print("🚀 Starting Nifty Dashboard Server...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/unified-data")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)