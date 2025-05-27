#!/usr/bin/env python3
"""
HTML Report Generator for HYG Alert System
Generates professional-grade HTML reports with interactive charts
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

class HTMLReportGenerator:
    """Generates professional HTML reports with charts and analytics"""
    
    def __init__(self, db, alert_engine, config):
        self.db = db
        self.alert_engine = alert_engine
        self.config = config
    
    def generate_html_report(self, analysis_results: Dict = None) -> str:
        """Generate complete HTML report with charts and analytics"""
        try:
            # Get data for report
            if not analysis_results:
                analysis_results = self.alert_engine.run_alert_analysis()
            
            latest_data = self.db.get_latest_data()
            historical_data = self.db.get_historical_data(180)  # 6 months
            
            if not latest_data or not historical_data:
                return self._generate_error_report("Insufficient data for report generation")
            
            # Prepare chart data
            chart_data = self._prepare_chart_data(historical_data)
            correlation_data = self._calculate_correlations(historical_data)
            
            # Generate HTML
            html_content = self._build_html_template(
                latest_data, 
                analysis_results, 
                chart_data, 
                correlation_data,
                historical_data
            )
            
            return html_content
            
        except Exception as e:
            return self._generate_error_report(f"Report generation failed: {e}")

    def _prepare_chart_data(self, historical_data: List[Dict]) -> Dict:
        """Prepare data for JavaScript charts - FIXED VERSION"""
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(historical_data)

            # Ensure we have data
            if df.empty:
                return {'error': 'No historical data available'}

            # Convert date column and sort
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Convert numeric columns to proper types
            numeric_cols = ['hyg_spread', 'hy_yield', 'treasury_10y', 'calculated_spread']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"DEBUG: DataFrame shape: {df.shape}")
            print(f"DEBUG: Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"DEBUG: Columns: {df.columns.tolist()}")
            print(f"DEBUG: Sample data:\n{df.head()}")

            # Prepare time series data with proper handling of NaN values
            chart_data = {
                'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'hyg_spread': df['hyg_spread'].where(pd.notnull(df['hyg_spread']), None).tolist(),
                'hy_yield': df['hy_yield'].where(pd.notnull(df['hy_yield']), None).tolist(),
                'treasury_10y': df['treasury_10y'].where(pd.notnull(df['treasury_10y']), None).tolist(),
                'calculated_spread': df['calculated_spread'].where(pd.notnull(df['calculated_spread']), None).tolist()
            }

            # Calculate moving averages (only if we have enough data)
            if len(df) >= 20:
                # 20-day moving average
                ma20 = df['hyg_spread'].rolling(window=20, min_periods=10).mean()
                chart_data['hyg_spread_ma20'] = ma20.where(pd.notnull(ma20), None).tolist()
            else:
                chart_data['hyg_spread_ma20'] = [None] * len(df)

            if len(df) >= 50:
                # 50-day moving average
                ma50 = df['hyg_spread'].rolling(window=50, min_periods=25).mean()
                chart_data['hyg_spread_ma50'] = ma50.where(pd.notnull(ma50), None).tolist()
            else:
                chart_data['hyg_spread_ma50'] = [None] * len(df)

            # Volatility data (20-day rolling std) - only if we have enough data
            if len(df) >= 20:
                volatility = df['hyg_spread'].rolling(window=20, min_periods=10).std() * 100
                chart_data['hyg_spread_volatility'] = volatility.where(pd.notnull(volatility), None).tolist()
            else:
                chart_data['hyg_spread_volatility'] = [None] * len(df)

            # Alert zones data
            chart_data['alert_zones'] = {
                'extreme_danger': self.config.EXTREME_COMPLACENCY,
                'high_danger': self.config.HIGH_COMPLACENCY,
                'normal_lower': getattr(self.config, 'NORMAL_LOWER', 3.5),
                'normal_upper': getattr(self.config, 'NORMAL_UPPER', 4.0),
                'early_warning': getattr(self.config, 'EARLY_WARNING', 4.5),
                'crisis': getattr(self.config, 'CRISIS', 7.0)
            }

            print(f"DEBUG: Chart data prepared successfully")
            print(f"DEBUG: Data points: {len(chart_data['dates'])}")
            print(f"DEBUG: Sample dates: {chart_data['dates'][:5]} ... {chart_data['dates'][-5:]}")
            print(f"DEBUG: Sample HYG spread: {[x for x in chart_data['hyg_spread'][:5] if x is not None]}")

            return chart_data

        except Exception as e:
            print(f"ERROR in _prepare_chart_data: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Chart data preparation failed: {str(e)}'}

   

    # Add debugging function to test the data
    def debug_chart_data(self):
        """Debug function to check data preparation"""
        try:
            historical_data = self.db.get_historical_data(180)
            print(f"Retrieved {len(historical_data)} historical records")

            if historical_data:
                print("Sample records:")
                for i, record in enumerate(historical_data[:3]):
                    print(f"  Record {i}: {record}")

                # Test chart data preparation
                chart_data = self._prepare_chart_data(historical_data)
                print(f"Chart data keys: {chart_data.keys()}")

                if 'error' in chart_data:
                    print(f"Chart data error: {chart_data['error']}")
                else:
                    print(f"Successfully prepared chart data with {len(chart_data.get('dates', []))} points")

        except Exception as e:
            print(f"Debug error: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_correlations(self, historical_data: List[Dict]) -> Dict:
        """Calculate correlations between different metrics"""
        try:
            df = pd.DataFrame(historical_data)
            
            # Clean data
            numeric_cols = ['hyg_spread', 'hy_yield', 'treasury_10y', 'calculated_spread']
            df_clean = df[numeric_cols].dropna()
            
            if len(df_clean) < 10:
                return {'error': 'Insufficient data for correlation analysis'}
            
            # Calculate correlation matrix
            corr_matrix = df_clean.corr()
            
            # Convert to dict for JSON serialization
            correlation_data = {
                'matrix': corr_matrix.round(3).to_dict(),
                'key_correlations': {
                    'hyg_vs_treasury': float(corr_matrix.loc['hyg_spread', 'treasury_10y']),
                    'hyg_vs_hy_yield': float(corr_matrix.loc['hyg_spread', 'hy_yield']),
                    'treasury_vs_hy_yield': float(corr_matrix.loc['treasury_10y', 'hy_yield'])
                }
            }
            
            # Calculate recent vs historical correlations
            recent_df = df_clean.tail(30)  # Last 30 days
            if len(recent_df) >= 10:
                recent_corr = recent_df.corr()
                correlation_data['recent_vs_historical'] = {
                    'recent_hyg_vs_treasury': float(recent_corr.loc['hyg_spread', 'treasury_10y']),
                    'historical_hyg_vs_treasury': float(corr_matrix.loc['hyg_spread', 'treasury_10y']),
                    'divergence': float(recent_corr.loc['hyg_spread', 'treasury_10y'] - corr_matrix.loc['hyg_spread', 'treasury_10y'])
                }
            
            return correlation_data
            
        except Exception as e:
            return {'error': str(e)}

    def _build_html_template(self, latest_data: Dict, analysis_results: Dict,
                             chart_data: Dict, correlation_data: Dict, historical_data: List[Dict]) -> str:
        """Build complete HTML report template with DYNAMIC data"""

        # Extract dynamic values
        current_spread = latest_data.get('hyg_spread', 0)
        current_hy_yield = latest_data.get('hy_yield', 0)
        current_treasury = latest_data.get('treasury_10y', 0)
        current_date = latest_data.get('date', 'N/A')

        alert_level = analysis_results.get('alert_level', 'UNKNOWN')
        alerts = analysis_results.get('alerts_generated', [])
        patterns = analysis_results.get('patterns_detected', [])
        recommendations = analysis_results.get('recommendations', [])

        # Get alert configuration
        alert_config = self.config.get_alert_config(alert_level)
        accuracy = alert_config.get('accuracy', 0.5) * 100

        # Calculate previous day data for changes
        prev_data = historical_data[-2] if len(historical_data) >= 2 else {}
        prev_spread = prev_data.get('hyg_spread', current_spread)
        prev_hy_yield = prev_data.get('hy_yield', current_hy_yield)
        prev_treasury = prev_data.get('treasury_10y', current_treasury)

        # Calculate changes
        spread_change = current_spread - prev_spread
        hy_change = current_hy_yield - prev_hy_yield
        treasury_change = current_treasury - prev_treasury

        # Generate change CSS classes
        spread_change_class = 'change-positive' if spread_change > 0 else 'change-negative' if spread_change < 0 else 'change-neutral'
        hy_change_class = 'change-positive' if hy_change > 0 else 'change-negative' if hy_change < 0 else 'change-neutral'
        treasury_change_class = 'change-positive' if treasury_change > 0 else 'change-negative' if treasury_change < 0 else 'change-neutral'

        # Get status information
        risk_status = self._get_status_text(current_spread)
        urgency = alert_config.get('urgency', 'UNKNOWN')
        expected_outcome = alert_config.get('expected_outcome', 'Unknown')
        timeline = alert_config.get('timeline', 'Unknown')
        recommended_action = alert_config.get('action', 'Monitor situation')

        # Convert chart data to JSON string for JavaScript
        chart_data_json = json.dumps(chart_data) if 'error' not in chart_data else '{"error": "Chart data unavailable"}'

        # Generate correlation data
        correlation_html = self._generate_correlation_matrix(correlation_data)

        # Generate alerts section
        alerts_html = self._generate_alerts_section(alerts, patterns)

        # Generate historical context
        historical_html = self._generate_historical_context(current_spread)

        # Generate recommendations
        recommendations_html = self._generate_recommendations_html(recommendations)

        html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HYG Market Alert Report - {current_date}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.js"></script>
        <style>
            {self._get_css_styles()}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header Section -->
            <header class="report-header">
                <div class="header-content">
                    <div>
                        <h1>📊 HYG Market Alert Report</h1>
                        <p class="subtitle">Based on Current Market Conditions and Credit Spread Analysis</p>
                    </div>
                    <div class="header-info">
                        <div class="date-info">
                            <strong>Report Date:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
                            <strong>Data As Of:</strong> {current_date}
                        </div>
                        <div class="alert-badge {alert_level.lower()}">
                            {self._get_alert_icon(alert_level)} {alert_level.replace('_', ' ')}
                        </div>
                    </div>
                </div>
            </header>

            <!-- Executive Summary -->
            <section class="executive-summary">
                <h2>🎯 Executive Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Current HYG Spread</h3>
                        <div class="metric-value">{current_spread:.2f}%</div>
                        <div class="metric-status {self._get_status_class(current_spread)}">
                            {risk_status}
                        </div>
                    </div>
                    <div class="summary-card">
                        <h3>Alert Level</h3>
                        <div class="metric-value">{alert_level.replace('_', ' ')}</div>
                        <div class="metric-confidence">
                            Confidence: {accuracy:.0f}%
                        </div>
                    </div>
                    <div class="summary-card">
                        <h3>Recommended Action</h3>
                        <div class="action-text">{recommended_action}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Expected Outcome</h3>
                        <div class="timeline-text">{timeline}</div>
                        <div class="outcome-text">{expected_outcome}</div>
                    </div>
                </div>
            </section>

            <!-- Market Data Overview -->
            <section class="market-overview">
                <h2>📈 Current Market Data</h2>
                <div class="data-table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Current Value</th>
                                <th>Previous Day</th>
                                <th>Change</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>HYG Spread</strong></td>
                                <td>{current_spread:.2f}%</td>
                                <td>{prev_spread:.2f}%</td>
                                <td class="{spread_change_class}">{spread_change:+.2f}%</td>
                                <td>{self._get_status_text(current_spread)}</td>
                            </tr>
                            <tr>
                                <td><strong>HY Yield</strong></td>
                                <td>{current_hy_yield:.2f}%</td>
                                <td>{prev_hy_yield:.2f}%</td>
                                <td class="{hy_change_class}">{hy_change:+.2f}%</td>
                                <td>{self._get_metric_status('hy_yield', current_hy_yield)}</td>
                            </tr>
                            <tr>
                                <td><strong>10Y Treasury</strong></td>
                                <td>{current_treasury:.2f}%</td>
                                <td>{prev_treasury:.2f}%</td>
                                <td class="{treasury_change_class}">{treasury_change:+.2f}%</td>
                                <td>{self._get_metric_status('treasury_10y', current_treasury)}</td>
                            </tr>
                            <tr>
                                <td><strong>Calculated Spread</strong></td>
                                <td>{(current_hy_yield - current_treasury):.2f}%</td>
                                <td>{(prev_hy_yield - prev_treasury):.2f}%</td>
                                <td class="{spread_change_class}">{((current_hy_yield - current_treasury) - (prev_hy_yield - prev_treasury)):+.2f}%</td>
                                <td>{"Consistent" if abs(current_spread - (current_hy_yield - current_treasury)) < 0.3 else "Divergent"}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

            <!-- Charts Section -->
            <section class="charts-section">
                <h2>📊 Historical Analysis & Trends</h2>

                <!-- HYG Spread Chart -->
                <div class="chart-container">
                    <h3>HYG Spread Trend (6 Months)</h3>
                    <canvas id="hygSpreadChart"></canvas>
                </div>

                <!-- Yield Comparison Chart -->
                <div class="chart-container">
                    <h3>Yield Comparison</h3>
                    <canvas id="yieldChart"></canvas>
                </div>

                <!-- Correlation Analysis -->
                <div class="chart-container">
                    <h3>Correlation Analysis</h3>
                    {correlation_html}
                </div>
                
                <!-- Volatility Chart -->
                <div class="chart-container">
                    <h3>HYG Spread Volatility (20-Day Rolling)</h3>
                    <canvas id="volatilityChart"></canvas>
                    
                </div>
                
                
                
                <div class="volatility-interpretation" style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <h4>Volatility Levels:</h4>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li><strong>&lt;15%:</strong> Normal market conditions</li>
                            <li><strong>15-20%:</strong> Some market uncertainty</li>
                            <li><strong>20-25%:</strong> Market stress emerging</li>
                            <li><strong>&gt;25%:</strong> Significant market stress</li>
                        </ul>
                    </div>
                    
            </section>

            <!-- Alert Analysis -->
            {alerts_html}

            <!-- Historical Context -->
            <section class="historical-context">
                <h2>📚 Historical Context</h2>
                {historical_html}
            </section>

            <!-- Recommendations -->
            <section class="recommendations">
                <h2>💡 Action Recommendations</h2>
                {recommendations_html}
            </section>

            <!-- Footer -->
            <footer class="report-footer">
                <p>Generated by HYG Market Intelligence System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Disclaimer:</strong> This report provides market analysis for informational purposes. 
                   Investment decisions should be made based on individual circumstances and risk tolerance.</p>
                <p><strong>Data Quality:</strong> {len(historical_data)} historical records, Latest: {current_date}</p>
            </footer>
        </div>

        <script>
            // Chart data from database
            const chartData = {chart_data_json};

            {self._get_javascript_charts()}
        </script>
    </body>
    </html>
        """

        return html_template
    
    def _get_css_styles(self) -> str:
        """Generate CSS styles for the report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }

        .report-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .alert-badge {
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.1em;
            text-align: center;
            min-width: 150px;
        }

        .alert-badge.extreme_danger { background: #ff4757; }
        .alert-badge.high_danger { background: #ff6b35; }
        .alert-badge.moderate_watch { background: #ffa502; }
        .alert-badge.normal { background: #26de81; }
        .alert-badge.early_warning { background: #ffa502; }
        .alert-badge.high_alert { background: #ff6b35; }
        .alert-badge.danger { background: #ff4757; }
        .alert-badge.extreme_crisis { background: #8e44ad; }

        .executive-summary {
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .summary-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #667eea;
        }

        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }

        .metric-status.danger { color: #e74c3c; }
        .metric-status.warning { color: #f39c12; }
        .metric-status.normal { color: #27ae60; }
        .metric-status.opportunity { color: #8e44ad; }

        .market-overview, .charts-section, .historical-context, .recommendations {
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .data-table th, .data-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .data-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }

        .data-table tr:hover {
            background-color: #f5f5f5;
        }

        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            position: relative;
            height: 400px;
        }

        .chart-container h3 {
            margin-bottom: 15px;
            color: #2c3e50;
            text-align: center;
        }

        .correlation-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 20px;
        }

        .correlation-cell {
            padding: 15px;
            text-align: center;
            border-radius: 8px;
            font-weight: bold;
        }

        .correlation-positive { background: linear-gradient(135deg, #26de81, #20bf6b); color: white; }
        .correlation-negative { background: linear-gradient(135deg, #ff4757, #c44569); color: white; }
        .correlation-neutral { background: linear-gradient(135deg, #f1f2f6, #ddd); color: #333; }

        .alerts-section {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .alert-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 5px solid #ff4757;
        }

        .pattern-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 5px solid #ffa502;
        }

        .recommendations ul {
            list-style: none;
            padding: 0;
        }

        .recommendations li {
            background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
        }

        .report-footer {
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 30px;
        }

        .change-positive { color: #27ae60; font-weight: bold; }
        .change-negative { color: #e74c3c; font-weight: bold; }
        .change-neutral { color: #666; }

        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                text-align: center;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            .chart-container {
                height: 300px;
            }
        }
        """
    
    def _get_javascript_charts(self) -> str:
        """Generate JavaScript for interactive charts"""
        return """
        // Chart configuration
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.color = '#333';

        // HYG Spread Chart
        const hygSpreadCtx = document.getElementById('hygSpreadChart').getContext('2d');
        const hygSpreadChart = new Chart(hygSpreadCtx, {
            type: 'line',
            data: {
                labels: chartData.dates,
                datasets: [{
                    label: 'HYG Spread',
                    data: chartData.hyg_spread,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3
                }, {
                    label: '20-Day MA',
                    data: chartData.hyg_spread_ma20,
                    borderColor: '#ffa502',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3
                }, {
                    label: '50-Day MA',
                    data: chartData.hyg_spread_ma50,
                    borderColor: '#ff6b35',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'HYG Spread with Moving Averages'
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Spread (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 2,
                        hoverRadius: 5
                    }
                }
            }
        });

        // Add alert zone lines
        const alertZones = chartData.alert_zones;
        hygSpreadChart.options.plugins.annotation = {
            annotations: {
                extremeDanger: {
                    type: 'line',
                    yMin: alertZones.extreme_danger,
                    yMax: alertZones.extreme_danger,
                    borderColor: '#ff4757',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    label: {
                        content: 'Extreme Danger',
                        enabled: true,
                        position: 'start'
                    }
                },
                crisis: {
                    type: 'line',
                    yMin: alertZones.crisis,
                    yMax: alertZones.crisis,
                    borderColor: '#8e44ad',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    label: {
                        content: 'Crisis Level',
                        enabled: true,
                        position: 'start'
                    }
                }
            }
        };

        // Yield Comparison Chart
        const yieldCtx = document.getElementById('yieldChart').getContext('2d');
        const yieldChart = new Chart(yieldCtx, {
            type: 'line',
            data: {
                labels: chartData.dates,
                datasets: [{
                    label: 'HY Yield',
                    data: chartData.hy_yield,
                    borderColor: '#26de81',
                    backgroundColor: 'rgba(38, 222, 129, 0.1)',
                    borderWidth: 3,
                    yAxisID: 'y'
                }, {
                    label: '10Y Treasury',
                    data: chartData.treasury_10y,
                    borderColor: '#4834d4',
                    backgroundColor: 'rgba(72, 52, 212, 0.1)',
                    borderWidth: 3,
                    yAxisID: 'y'
                }, {
                    label: 'Calculated Spread',
                    data: chartData.calculated_spread,
                    borderColor: '#ff9ff3',
                    backgroundColor: 'rgba(255, 159, 243, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Yield Components Analysis'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Yield (%)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Spread (%)'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });

        // Volatility Chart
                    try {
                const volatilityCtx = document.getElementById('volatilityChart').getContext('2d');
                const volatilityChart = new Chart(volatilityCtx, {
                    type: 'line',  // ✅ Changed from 'area' to 'line'
                    data: {
                        labels: chartData.dates,
                        datasets: [{
                            label: 'HYG Spread Volatility (20D)',
                            data: chartData.hyg_spread_volatility,
                            borderColor: '#ff6b35',
                            backgroundColor: 'rgba(255, 107, 53, 0.3)',
                            borderWidth: 2,
                            fill: true,  // ✅ This creates the area effect
                            tension: 0.4,
                            spanGaps: true  // ✅ Handle null values
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Market Volatility Indicator'
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Volatility (% × 100)'
                                }
                            }
                        },
                        elements: {
                            point: {
                                radius: 1,
                                hoverRadius: 4
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Error creating Volatility chart:', error);
            }
        """
    
    def _generate_market_data_rows(self, latest_data: Dict, historical_data: List[Dict]) -> str:
        """Generate table rows for market data"""
        try:
            # Get previous day data for comparison
            prev_data = historical_data[-2] if len(historical_data) >= 2 else {}
            
            rows = []
            metrics = [
                ('HYG Spread', 'hyg_spread', '%'),
                ('HY Yield', 'hy_yield', '%'),
                ('10Y Treasury', 'treasury_10y', '%'),
                ('Calculated Spread', 'calculated_spread', '%')
            ]
            
            for label, key, unit in metrics:
                current = latest_data.get(key)
                previous = prev_data.get(key)
                
                if current is not None:
                    current_str = f"{current:.2f}{unit}"
                    
                    if previous is not None:
                        change = current - previous
                        change_str = f"{change:+.2f}{unit}"
                        change_class = 'change-positive' if change > 0 else 'change-negative' if change < 0 else 'change-neutral'
                        previous_str = f"{previous:.2f}{unit}"
                    else:
                        change_str = "N/A"
                        change_class = 'change-neutral'
                        previous_str = "N/A"
                    
                    status = self._get_metric_status(key, current)
                    
                    rows.append(f"""
                    <tr>
                        <td><strong>{label}</strong></td>
                        <td>{current_str}</td>
                        <td>{previous_str}</td>
                        <td class="{change_class}">{change_str}</td>
                        <td>{status}</td>
                    </tr>
                    """)
            
            return ''.join(rows)
            
        except Exception:
            return "<tr><td colspan='5'>Error generating market data</td></tr>"

    def _generate_volatility_chart_section(self) -> str:
        """Generate the volatility chart section HTML"""
        return """
        <!-- Volatility Chart -->
        <div class="chart-container">
            <h3>Market Stress Indicator (20-Day Rolling Volatility)</h3>
            <canvas id="volatilityChart"></canvas>
            <div class="volatility-interpretation" style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                <h4>Volatility Levels:</h4>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    <li><strong>&lt;15%:</strong> Normal market conditions</li>
                    <li><strong>15-20%:</strong> Some market uncertainty</li>
                    <li><strong>20-25%:</strong> Market stress emerging</li>
                    <li><strong>&gt;25%:</strong> Significant market stress</li>
                </ul>
            </div>
        </div>
        """


    def _generate_correlation_matrix(self, correlation_data: Dict) -> str:
        """Generate HTML for correlation matrix"""
        if 'error' in correlation_data:
            return f"<p>Error: {correlation_data['error']}</p>"
        
        try:
            html = """
            <div class="correlation-summary">
                <h4>Key Correlations</h4>
                <div class="correlation-grid">
            """
            
            key_corrs = correlation_data.get('key_correlations', {})
            
            correlations = [
                ('HYG vs Treasury', key_corrs.get('hyg_vs_treasury', 0)),
                ('HYG vs HY Yield', key_corrs.get('hyg_vs_hy_yield', 0)),
                ('Treasury vs HY Yield', key_corrs.get('treasury_vs_hy_yield', 0))
            ]
            
            for label, value in correlations:
                css_class = 'correlation-positive' if value > 0.3 else 'correlation-negative' if value < -0.3 else 'correlation-neutral'
                html += f"""
                <div class="correlation-cell {css_class}">
                    <div>{label}</div>
                    <div>{value:.3f}</div>
                </div>
                """
            
            html += "</div></div>"
            
            # Add interpretation
            html += """
            <div class="correlation-interpretation">
                <h4>Interpretation</h4>
                <ul>
                    <li><strong>Positive correlation (>0.3):</strong> Metrics move in same direction</li>
                    <li><strong>Negative correlation (<-0.3):</strong> Metrics move in opposite directions</li>
                    <li><strong>Low correlation (-0.3 to 0.3):</strong> Weak relationship</li>
                </ul>
            </div>
            """
            
            return html
            
        except Exception:
            return "<p>Error generating correlation matrix</p>"
    
    def _generate_alerts_section(self, alerts: List[Dict], patterns: List[Dict]) -> str:
        """Generate alerts section HTML"""
        if not alerts and not patterns:
            return """
            <section class="alerts-section">
                <h2>🟢 Alert Status</h2>
                <div class="no-alerts">
                    <h3>✅ No Active Alerts</h3>
                    <p>Current market conditions do not trigger any warning patterns or threshold alerts.</p>
                </div>
            </section>
            """
        
        html = """
        <section class="alerts-section">
            <h2>🚨 Active Alerts & Patterns</h2>
        """
        
        if alerts:
            html += "<h3>⚠️ Active Alerts</h3>"
            for alert in alerts:
                urgency_icon = "🚨" if alert.get('urgency') == 'IMMEDIATE' else "🔴" if alert.get('urgency') == 'HIGH' else "🟡"
                html += f"""
                <div class="alert-item">
                    <div class="alert-header">
                        <span class="alert-icon">{urgency_icon}</span>
                        <strong>{alert.get('alert_level', 'Unknown').replace('_', ' ')}</strong>
                        <span class="alert-confidence">Confidence: {alert.get('historical_accuracy', 0)*100:.0f}%</span>
                    </div>
                    <div class="alert-content">
                        <p><strong>Condition:</strong> {alert.get('trigger_condition', 'Unknown')}</p>
                        <p><strong>Action:</strong> {alert.get('recommended_action', 'Monitor situation')}</p>
                        <p><strong>Timeline:</strong> {alert.get('timeline', 'Unknown')}</p>
                    </div>
                </div>
                """
        
        if patterns:
            html += "<h3>🔍 Detected Patterns</h3>"
            for pattern in patterns:
                confidence = pattern.get('confidence', 0) * 100
                severity_icon = "🚨" if pattern.get('severity') == 'EXTREME' else "🔴" if pattern.get('severity') == 'HIGH' else "🟡"
                html += f"""
                <div class="pattern-item">
                    <div class="pattern-header">
                        <span class="pattern-icon">{severity_icon}</span>
                        <strong>{pattern.get('pattern_type', 'unknown').replace('_', ' ').title()}</strong>
                        <span class="pattern-confidence">Confidence: {confidence:.0f}%</span>
                    </div>
                    <div class="pattern-content">
                        <p>{pattern.get('description', 'No description available')}</p>
                        <p><strong>Expected Impact:</strong> {pattern.get('expected_impact', 'Unknown')}</p>
                        <p><strong>Timeline:</strong> {pattern.get('timeline', 'Unknown')}</p>
                    </div>
                </div>
                """
        
        html += "</section>"
        return html
    
    def _generate_historical_context(self, current_spread: float) -> str:
        """Generate historical context section"""
        try:
            # Find similar historical periods from config
            similar_examples = []
            
            for category, examples in self.config.HISTORICAL_EXAMPLES.items():
                for example in examples:
                    if abs(example.get('spread', 0) - current_spread) < 0.5:
                        similar_examples.append({
                            'category': category,
                            'date': example.get('date'),
                            'spread': example.get('spread'),
                            'outcome': example.get('outcome')
                        })
            
            if not similar_examples:
                return """
                <div class="historical-no-match">
                    <h3>📊 Current Reading: Unique Territory</h3>
                    <p>No direct historical precedents found for current spread level of <strong>{:.2f}%</strong>.</p>
                    <p>This suggests either stable market conditions or a unique market environment requiring careful monitoring.</p>
                </div>
                """.format(current_spread)
            
            html = "<h3>📚 Similar Historical Periods</h3>"
            html += "<div class='historical-examples'>"
            
            for example in similar_examples[:3]:  # Show top 3
                html += f"""
                <div class="historical-example">
                    <div class="example-header">
                        <strong>{example['date']}</strong> - {example['category'].replace('_', ' ').title()}
                    </div>
                    <div class="example-details">
                        <p><strong>Spread:</strong> {example['spread']:.2f}%</p>
                        <p><strong>Outcome:</strong> {example['outcome']}</p>
                    </div>
                </div>
                """
            
            html += "</div>"
            return html
            
        except Exception:
            return "<p>Error generating historical context</p>"
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations"""
        if not recommendations:
            return "<p>No specific recommendations at this time. Continue standard monitoring.</p>"
        
        html = "<ul>"
        for rec in recommendations:
            # Clean up emoji and formatting for HTML
            clean_rec = rec.replace('•', '').strip()
            html += f"<li>{clean_rec}</li>"
        html += "</ul>"
        
        return html
    
    def _get_alert_icon(self, alert_level: str) -> str:
        """Get emoji icon for alert level"""
        icons = {
            'EXTREME_DANGER': '🚨',
            'HIGH_DANGER': '🔴',
            'MODERATE_WATCH': '🟡',
            'NORMAL': '🟢',
            'EARLY_WARNING': '🟡',
            'HIGH_ALERT': '🔴',
            'DANGER': '🚨',
            'EXTREME_CRISIS': '💎'
        }
        return icons.get(alert_level, '❓')
    
    def _get_status_class(self, spread: float) -> str:
        """Get CSS class for spread status"""
        if spread < self.config.EXTREME_COMPLACENCY:
            return 'danger'
        elif spread < self.config.HIGH_COMPLACENCY:
            return 'warning'
        elif spread > self.config.CRISIS:
            return 'opportunity'
        else:
            return 'normal'
    
    def _get_status_text(self, spread: float) -> str:
        """Get status text for spread"""
        if spread < self.config.EXTREME_COMPLACENCY:
            return 'Extreme Danger'
        elif spread < self.config.HIGH_COMPLACENCY:
            return 'High Risk'
        elif spread < self.config.NORMAL_LOWER:
            return 'Elevated Risk'
        elif spread <= self.config.NORMAL_UPPER:
            return 'Normal Range'
        elif spread < self.config.EARLY_WARNING:
            return 'Mild Stress'
        elif spread < self.config.CRISIS:
            return 'Market Stress'
        else:
            return 'Crisis/Opportunity'
    
    def _get_metric_status(self, metric_key: str, value: float) -> str:
        """Get status for individual metrics"""
        if metric_key == 'hyg_spread':
            return self._get_status_text(value)
        elif metric_key == 'treasury_10y':
            if value < 2.0:
                return 'Very Low'
            elif value < 4.0:
                return 'Low'
            elif value < 6.0:
                return 'Normal'
            else:
                return 'High'
        elif metric_key == 'hy_yield':
            if value < 5.0:
                return 'Low Risk'
            elif value < 8.0:
                return 'Moderate Risk'
            elif value < 12.0:
                return 'High Risk'
            else:
                return 'Very High Risk'
        else:
            return 'Normal'
    
    def _generate_error_report(self, error_message: str) -> str:
        """Generate error report HTML"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>HYG Report Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .error {{ background: #ffe6e6; padding: 20px; border-radius: 8px; border-left: 4px solid #ff4444; }}
    </style>
</head>
<body>
    <div class="error">
        <h2>❌ Report Generation Error</h2>
        <p><strong>Error:</strong> {error_message}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h3>Troubleshooting Steps:</h3>
        <ol>
            <li>Check database connection and data availability</li>
            <li>Ensure historical data is loaded: <code>python main.py load</code></li>
            <li>Update with latest data: <code>python main.py update</code></li>
            <li>Check system logs for detailed error information</li>
        </ol>
    </div>
</body>
</html>
        """
    
    def save_html_report(self, output_dir: str = 'reports') -> str:
        """Generate and save HTML report to file"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate HTML content
            html_content = self.generate_html_report()
            
            # Save to file
            filename = f"hyg_report_{datetime.now().strftime('%Y%m%d')}.html"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return filepath
            
        except Exception as e:
            # Create error report
            error_html = self._generate_error_report(str(e))
            error_filename = f"hyg_report_error_{datetime.now().strftime('%Y%m%d')}.html"
            error_filepath = os.path.join(output_dir, error_filename)
            
            with open(error_filepath, 'w', encoding='utf-8') as f:
                f.write(error_html)
            
            return error_filepath

# Integration with existing reports.py
def integrate_html_generator():
    """
    Add this method to your existing ReportGenerator class in reports.py
    """
    code_to_add = '''
    def generate_html_report(self) -> str:
        """Generate professional HTML report with charts"""
        from html_report_generator import HTMLReportGenerator
        
        html_generator = HTMLReportGenerator(self.db, self.alert_engine, self.config)
        return html_generator.generate_html_report()
    
    def save_html_report(self, output_dir: str = 'reports') -> str:
        """Save HTML report to file"""
        from html_report_generator import HTMLReportGenerator
        
        html_generator = HTMLReportGenerator(self.db, self.alert_engine, self.config)
        return html_generator.save_html_report(output_dir)
    '''
    
    return code_to_add

# Usage example
def generate_sample_html_report():
    """Example of how to use the HTML report generator"""
    
    # This would be integrated into your main.py
    example_usage = '''
    # In your main.py, modify the generate_reports function:
    
    def generate_reports(report_generator, results, output_dir='reports'):
        """Generate and save reports including HTML"""
        print("📄 Generating reports...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate traditional reports
        daily_report = report_generator.generate_daily_report()
        daily_file = os.path.join(output_dir, f"daily_report_{datetime.now().strftime('%Y%m%d')}.md")
        
        with open(daily_file, 'w', encoding='utf-8') as f:
            f.write(daily_report)
        print(f"📝 Daily report saved: {daily_file}")
        
        # Generate HTML report
        html_file = report_generator.save_html_report(output_dir)
        print(f"🌐 HTML report saved: {html_file}")
        
        # Generate other reports...
    '''
    
    return example_usage