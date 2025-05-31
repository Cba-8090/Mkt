
import os
import re
import glob
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
from datetime import datetime, timedelta


class SectorSentimentTrendGenerator:
    """
    Class to generate a trend analysis of sector sentiment between two dates
    by processing Progressive Analysis stock reports or dashboard files.
    """

    def __init__(self, output_dir, start_date, end_date, reports_base_dir=None,
                 dashboard_base_dir=None, date_format="%Y%m%d", generate_allinone=False):
        """
        Initialize with date range and directory information

        Args:
            output_dir (str): Directory to save output files
            start_date (str): Start date in format YYYYMMDD
            end_date (str): End date in format YYYYMMDD
            reports_base_dir (str): Base directory containing dated folders with stock reports
                                   (e.g., /reports/20240501/, /reports/20240502/, etc.)
            dashboard_base_dir (str): Base directory containing dashboard HTML files
                                     (e.g., market_dashboard_20240501.html)
            date_format (str): Format of date strings in filenames
        """
        self.output_dir = output_dir
        self.start_date = datetime.strptime(start_date, date_format)
        self.end_date = datetime.strptime(end_date, date_format)
        self.reports_base_dir = reports_base_dir
        self.dashboard_base_dir = dashboard_base_dir
        self.date_format = date_format
        self.generate_allinone = generate_allinone

        # Ensure at least one data source is provided
        if not reports_base_dir and not dashboard_base_dir:
            raise ValueError("At least one of reports_base_dir or dashboard_base_dir must be provided")

        # Data structures to store results
        self.date_sentiment_data = {}  # Will store sentiment data by date
        self.sentiment_categories = ['LONG', 'ACCUMULATION', 'NEUTRAL', 'DISTRIBUTION', 'SHORT']
        self.sentiment_mapping = {
            'LONG': 2,
            'ACCUMULATION (trending toward LONG)': 1,
            'NEUTRAL': 0,
            'DISTRIBUTION (trending toward SHORT)': -1,
            'SHORT': -2
        }

    def generate_trend_analysis(self):
        """
        Main method to generate the trend analysis

        Returns:
            str: Path to the generated HTML file
        """
        # Process data for each date in the range
        current_date = self.start_date
        while current_date <= self.end_date:
            date_str = current_date.strftime(self.date_format)
            print(f"Processing data for {date_str}...")

            # Try first method (direct from stock reports)
            if self.reports_base_dir:
                success = self._process_stock_reports_for_date(date_str)
                if success:
                    current_date += timedelta(days=1)
                    continue

            # Fall back to second method (extract from dashboard)
            if self.dashboard_base_dir:
                success = self._extract_from_dashboard_for_date(date_str)
                if not success:
                    print(f"WARNING: No data found for {date_str}")

            current_date += timedelta(days=1)

        # Convert collected data to a more suitable format for visualization
        trend_data = self._prepare_trend_data()

        # Generate the single-filter visualization
        html_path = self._generate_visualization(trend_data)

        # If requested, generate the all-in-one visualization with all sectors
        if self.generate_allinone:
            allinone_path = self._generate_allinone_visualization(trend_data)
            return allinone_path

        return html_path

    def _process_stock_reports_for_date(self, date_str):
        """
        Process all stock reports for a specific date to extract sector sentiment data

        Args:
            date_str (str): Date string in the format specified by date_format

        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Find the reports directory for this date
        date_reports_dir = os.path.join(self.reports_base_dir, date_str)
        if not os.path.exists(date_reports_dir):
            date_reports_dir = os.path.join(self.reports_base_dir, f"output_{date_str}")

        if not os.path.exists(date_reports_dir):
            # Try looking for a directory that contains this date
            for dir_name in os.listdir(self.reports_base_dir):
                if date_str in dir_name and os.path.isdir(os.path.join(self.reports_base_dir, dir_name)):
                    date_reports_dir = os.path.join(self.reports_base_dir, dir_name)
                    break

        if not os.path.exists(date_reports_dir):
            print(f"No reports directory found for date {date_str}")
            return False

        # Find all report files
        report_files = glob.glob(os.path.join(date_reports_dir, "*_progressive_analysis.html"))
        if not report_files:
            report_files = glob.glob(os.path.join(date_reports_dir, "*.html"))

        if not report_files:
            print(f"No report files found in {date_reports_dir}")
            return False

        print(f"Found {len(report_files)} report files for {date_str}")

        # Process each report file
        sectors_data = {}

        for file_path in report_files:
            try:
                stock_name = os.path.basename(file_path).split('_')[0]
                stock_data = self._extract_data_from_html(file_path)

                if stock_data and stock_data.get('sector') and stock_data.get('current_sentiment'):
                    sector = stock_data['sector']
                    sentiment = stock_data['current_sentiment']

                    # Initialize sector data if not already present
                    if sector not in sectors_data:
                        sectors_data[sector] = {
                            'stocks': 0,
                            'sentiment_counts': {cat: 0 for cat in self.sentiment_categories}
                        }

                    # Increment stock count
                    sectors_data[sector]['stocks'] += 1

                    # Determine sentiment category
                    if 'LONG' in sentiment.upper() and 'TRENDING' not in sentiment.upper():
                        sectors_data[sector]['sentiment_counts']['LONG'] += 1
                    elif 'ACCUMULATION' in sentiment.upper():
                        sectors_data[sector]['sentiment_counts']['ACCUMULATION'] += 1
                    elif 'NEUTRAL' in sentiment.upper():
                        sectors_data[sector]['sentiment_counts']['NEUTRAL'] += 1
                    elif 'DISTRIBUTION' in sentiment.upper():
                        sectors_data[sector]['sentiment_counts']['DISTRIBUTION'] += 1
                    elif 'SHORT' in sentiment.upper():
                        sectors_data[sector]['sentiment_counts']['SHORT'] += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Calculate percentages
        for sector, data in sectors_data.items():
            total_stocks = data['stocks']
            data['sentiment_percentages'] = {
                cat: (data['sentiment_counts'][cat] / total_stocks * 100) if total_stocks > 0 else 0
                for cat in self.sentiment_categories
            }

        # Store the data for this date
        if sectors_data:
            self.date_sentiment_data[date_str] = sectors_data
            return True
        else:
            print(f"No valid sector data found for {date_str}")
            return False

    def _extract_data_from_html(self, html_file):
        """
        Extract sentiment and sector data from a single stock report HTML file

        Args:
            html_file (str): Path to the HTML file

        Returns:
            dict: Extracted data or None if extraction failed
        """
        try:
            # Read HTML content
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract basic information
            stock_name = os.path.basename(html_file).split('_')[0]

            # Find sector with fallback mechanisms
            sector = "Unknown"
            header_div = soup.select_one('div.header')
            if header_div:
                sector_p = header_div.find('p', string=lambda s: 'Sector:' in s if s else False)
                if sector_p:
                    sector = sector_p.text.replace('Sector:', '').strip()

            # Try alternative approach for sector
            if sector == "Unknown":
                for p in soup.find_all('p'):
                    if 'Sector:' in p.text:
                        sector = p.text.replace('Sector:', '').strip()
                        break

            # Extract current sentiment with fallback mechanisms
            current_sentiment = ""

            # Try approach 1: Find in .summary-box div
            summary_box = soup.select_one('div.summary-box')
            if summary_box:
                for p in summary_box.find_all('p'):
                    strong = p.find('strong')
                    if strong and 'Current View' in strong.text:
                        current_sentiment = p.text.replace(strong.text, '').strip()
                        break

            # Try approach 2: Find each strong tag directly
            if not current_sentiment:
                for strong in soup.find_all('strong'):
                    if 'Current View' in strong.text:
                        parent = strong.parent
                        if parent:
                            parts = parent.text.split(':', 1)
                            if len(parts) > 1:
                                current_sentiment = parts[1].strip()
                                break

            # Try approach 3: Historical data table
            if not current_sentiment:
                data_table = soup.select_one('table.data-table')
                if data_table:
                    rows = data_table.select('tr')
                    if len(rows) > 1:  # Header + at least one data row
                        first_data_row = rows[1]
                        cells = first_data_row.select('td')
                        if len(cells) >= 2:
                            current_sentiment = cells[1].text.strip()

            # Return the extracted data
            if sector and current_sentiment:
                return {
                    'stock': stock_name,
                    'sector': sector,
                    'current_sentiment': current_sentiment
                }
            else:
                print(f"Incomplete data for {html_file}: sector={sector}, sentiment={current_sentiment}")
                return None

        except Exception as e:
            print(f"Error extracting data from {html_file}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_from_dashboard_for_date(self, date_str):
        """
        Extract sector sentiment data from a dashboard HTML file

        Args:
            date_str (str): Date string in the format specified by date_format

        Returns:
            bool: True if extraction was successful, False otherwise
        """
        # Look for the dashboard file
        dashboard_path = os.path.join(self.dashboard_base_dir, f"market_dashboard_{date_str}.html")
        if not os.path.exists(dashboard_path):
            print(f"Dashboard file not found: {dashboard_path}")
            return False

        try:
            # Read the dashboard HTML
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Extract sector heatmap data
            soup = BeautifulSoup(html_content, 'html.parser')

            # First, try to find the JavaScript data
            sector_data = {}
            js_match = re.search(r'const sectorFig = (.+?);', html_content)

            if js_match:
                # Extract JSON data
                try:
                    sector_fig = json.loads(js_match.group(1))

                    # Extract sectors (y-axis)
                    sectors = []
                    for trace in sector_fig.get('data', []):
                        if trace.get('y'):
                            sectors = trace.get('y')
                            break

                    # Extract sentiment categories (x-axis)
                    sentiment_categories = []
                    for trace in sector_fig.get('data', []):
                        if trace.get('x'):
                            sentiment_categories = trace.get('x')
                            break

                    # Extract z-data (percentages)
                    z_data = []
                    for trace in sector_fig.get('data', []):
                        if trace.get('z'):
                            z_data = trace.get('z')
                            break

                    # Construct sector data
                    if sectors and sentiment_categories and z_data and len(sectors) == len(z_data):
                        for i, sector in enumerate(sectors):
                            sector_percentages = {}
                            for j, category in enumerate(sentiment_categories):
                                if j < len(z_data[i]):
                                    sector_percentages[category] = z_data[i][j]

                            # Calculate implied counts
                            total_stocks = 100  # Assume percentages sum to 100
                            sentiment_counts = {
                                cat: int(round(pct * total_stocks / 100))
                                for cat, pct in sector_percentages.items()
                            }

                            sector_data[sector] = {
                                'stocks': total_stocks,
                                'sentiment_counts': sentiment_counts,
                                'sentiment_percentages': sector_percentages
                            }
                except Exception as js_error:
                    print(f"Error parsing JavaScript data: {js_error}")

            # If no data from JavaScript, try parsing the HTML table
            if not sector_data:
                # TODO: Implement HTML table parsing if needed
                pass

            # Store the data for this date
            if sector_data:
                self.date_sentiment_data[date_str] = sector_data
                return True
            else:
                print(f"No sector data found in dashboard for {date_str}")
                return False

        except Exception as e:
            print(f"Error extracting from dashboard {dashboard_path}: {e}")
            return False

    def _prepare_trend_data(self):
        """
        Convert the collected date-sentiment data into a format suitable for trend visualization

        Returns:
            pd.DataFrame: DataFrame with trend data
        """
        # Get all unique sectors
        all_sectors = set()
        for date_data in self.date_sentiment_data.values():
            all_sectors.update(date_data.keys())

        # Prepare a list to store DataFrame rows
        trend_rows = []

        # Process data for each date and sector
        for date_str, date_data in sorted(self.date_sentiment_data.items()):
            for sector in all_sectors:
                sector_data = date_data.get(sector, {})

                # Get sentiment percentages or defaults
                percentages = sector_data.get('sentiment_percentages', {})

                # Create a row for each sentiment category
                for category in self.sentiment_categories:
                    percentage = percentages.get(category, 0)
                    trend_rows.append({
                        'date': date_str,
                        'sector': sector,
                        'sentiment_category': category,
                        'percentage': percentage
                    })

        # Convert to DataFrame
        return pd.DataFrame(trend_rows)

    def _generate_visualization(self, trend_data):
        """
        Generate an HTML visualization of the sector sentiment trends

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            str: Path to the generated HTML file
        """
        # Format dates for display
        trend_data['formatted_date'] = pd.to_datetime(trend_data['date'], format=self.date_format).dt.strftime(
            '%Y-%m-%d')
        # Add shorter date format for chart display
        trend_data['short_date'] = pd.to_datetime(trend_data['date'], format=self.date_format).dt.strftime('%d%m')

        # Create the HTML file
        html_filename = f"sector_sentiment_trend_{self.start_date.strftime(self.date_format)}_{self.end_date.strftime(self.date_format)}.html"
        html_path = os.path.join(self.output_dir, html_filename)

        # Generate the HTML content
        html_content = self._generate_trend_html(trend_data)

        # Write to file
        os.makedirs(self.output_dir, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Trend visualization generated at: {html_path}")
        return html_path

        
    def _generate_trend_html(self, trend_data):
        """
        Generate the HTML content for the trend visualization

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            str: HTML content
        """
        # Get unique sectors and dates for filtering
        unique_sectors = sorted(trend_data['sector'].unique())
        unique_dates = sorted(trend_data['formatted_date'].unique())

        # Create Plotly figure JSON for initial visualization
        # (We'll use JavaScript to update it based on filters)
        initial_fig = self._create_initial_heatmap(trend_data)
        initial_fig_json = json.dumps(initial_fig.to_dict())

        # Create the HTML content with interactive filters
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sector Sentiment Trend Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                :root {{
                    --primary-color: #2C3E50;
                    --secondary-color: #3498DB;
                    --accent-color: #F39C12;
                    --neutral-color: #ECF0F1;
                }}

                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}

                body {{
                    background-color: var(--neutral-color);
                    color: #333;
                    line-height: 1.6;
                    padding: 20px;
                }}

                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                }}

                h1 {{
                    color: var(--primary-color);
                    margin-bottom: 20px;
                    text-align: center;
                }}

                .filter-controls {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f5f5f5;
                    border-radius: 8px;
                }}

                .filter-group {{
                    display: flex;
                    flex-direction: column;
                }}

                .filter-group label {{
                    font-weight: bold;
                    margin-bottom: 8px;
                    color: var(--primary-color);
                }}

                select, input {{
                    padding: 8px 12px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 16px;
                }}

                select[multiple] {{
                    height: 150px;
                }}

                .view-controls {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                    padding: 10px;
                    background-color: #f5f5f5;
                    border-radius: 8px;
                }}

                .view-selector {{
                    display: flex;
                    gap: 15px;
                }}

                .view-selector button {{
                    padding: 8px 16px;
                    background-color: var(--secondary-color);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }}

                .view-selector button.active {{
                    background-color: var(--primary-color);
                }}

                .view-selector button:hover {{
                    background-color: var(--primary-color);
                }}

                .filter-actions {{
                    display: flex;
                    gap: 10px;
                }}

                .filter-button {{
                    padding: 8px 16px;
                    background-color: var(--accent-color);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-weight: bold;
                    transition: background-color 0.3s;
                }}

                .filter-button:hover {{
                    background-color: #E67E22;
                }}

                .visualization {{
                    min-height: 600px;
                }}

                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sector Sentiment Trend Analysis</h1>

                <div class="filter-controls">
                    <div class="filter-group">
                        <label for="sector-select">Filter by Sectors:</label>
                        <select id="sector-select" multiple>
                            <option value="all" selected>All Sectors</option>
                            {self._generate_options(unique_sectors)}
                        </select>
                    </div>

                    <div class="filter-group">
                        <label for="sentiment-select">Filter by Sentiment Categories:</label>
                        <select id="sentiment-select" multiple>
                            <option value="all" selected>All Categories</option>
                            {self._generate_options(self.sentiment_categories)}
                        </select>
                    </div>

                    <div class="filter-group">
                        <label for="date-range">Date Range:</label>
                        <div style="display: flex; gap: 10px;">
                            <select id="start-date">
                                {self._generate_options(unique_dates)}
                            </select>
                            <span style="align-self: center;">to</span>
                            <select id="end-date">
                                {self._generate_options(unique_dates, reverse=True)}
                            </select>
                        </div>
                    </div>

                    <div class="filter-group filter-actions">
                        <label>&nbsp;</label>
                        <button id="apply-filters" class="filter-button">Apply Filters</button>
                    </div>
                </div>

                <div class="view-controls">
                    <div class="view-selector">
                        <button id="view-heatmap" class="active">Heatmap View</button>
                        <button id="view-trend">Trend View</button>
                    </div>
                </div>

                <div id="visualization" class="visualization"></div>

                <div class="footer">
                    <p>Generated by Sector Sentiment Trend Generator | Data Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}</p>
                </div>
            </div>

            <script>
                // Store the full dataset
                const fullData = {trend_data.to_json(orient='records')};

                // Initialize the visualization
                let currentView = 'heatmap';
                let initialFigure = {initial_fig_json};

                // Create the initial visualization
                document.addEventListener('DOMContentLoaded', function() {{
                    Plotly.newPlot('visualization', initialFigure.data, initialFigure.layout);

                    // Add event listeners
                    document.getElementById('apply-filters').addEventListener('click', updateVisualization);
                    document.getElementById('view-heatmap').addEventListener('click', () => changeView('heatmap'));
                    document.getElementById('view-trend').addEventListener('click', () => changeView('trend'));

                    // Set initial end date to the latest date
                    const endDateSelect = document.getElementById('end-date');
                    if (endDateSelect.options.length > 0) {{
                        endDateSelect.selectedIndex = 0;
                    }}
                }});

                // Function to change the visualization view
                function changeView(viewType) {{
                    currentView = viewType;

                    // Update button states
                    document.getElementById('view-heatmap').classList.toggle('active', viewType === 'heatmap');
                    document.getElementById('view-trend').classList.toggle('active', viewType === 'trend');

                    // Update the visualization
                    updateVisualization();
                }}

                // Function to update the visualization based on filters
                function updateVisualization() {{
                    // Get filter values
                    const selectedSectors = getMultiSelectValues('sector-select');
                    const selectedSentiments = getMultiSelectValues('sentiment-select');
                    const startDate = document.getElementById('start-date').value;
                    const endDate = document.getElementById('end-date').value;

                    // Filter the data
                    let filteredData = fullData.filter(row => {{
                        const dateInRange = row.formatted_date >= startDate && row.formatted_date <= endDate;
                        const sectorMatch = selectedSectors.includes('all') || selectedSectors.includes(row.sector);
                        const sentimentMatch = selectedSentiments.includes('all') || 
                                              selectedSentiments.includes(row.sentiment_category);
                        return dateInRange && sectorMatch && sentimentMatch;
                    }});

                    // Create the appropriate visualization
                    if (currentView === 'heatmap') {{
                        createHeatmap(filteredData);
                    }} else {{
                        createTrendVisualization(filteredData);
                    }}
                }}

                // Helper function to get values from a multi-select
                function getMultiSelectValues(selectId) {{
                    const select = document.getElementById(selectId);
                    const result = [];
                    for (let option of select.options) {{
                        if (option.selected) {{
                            result.push(option.value);
                        }}
                    }}
                    return result;
                }}

                // Function to create a heatmap visualization
                function createHeatmap(filteredData) {{
                    // Determine the latest date in the filtered data
                    const uniqueDates = [...new Set(filteredData.map(row => row.formatted_date))].sort();
                    const latestDate = uniqueDates[uniqueDates.length - 1];

                    // Filter to the latest date
                    const latestData = filteredData.filter(row => row.formatted_date === latestDate);

                    // Get unique sectors and sentiment categories
                    const sectors = [...new Set(latestData.map(row => row.sector))].sort();
                    const sentimentCategories = [...new Set(latestData.map(row => row.sentiment_category))];

                    // Prepare data for heatmap
                    const zValues = [];
                    const textValues = [];

                    for (const sector of sectors) {{
                        const sectorData = [];
                        const sectorText = [];

                        for (const category of sentimentCategories) {{
                            const matchingRow = latestData.find(row => 
                                row.sector === sector && row.sentiment_category === category);

                            const value = matchingRow ? matchingRow.percentage : 0;
                            sectorData.push(value);
                            sectorText.push(`${{value.toFixed(1)}}%`);
                        }}

                        zValues.push(sectorData);
                        textValues.push(sectorText);
                    }}

                    // Create heatmap figure
                    const figure = {{
                        data: [{{
                            z: zValues,
                            x: sentimentCategories,
                            y: sectors,
                            type: 'heatmap',
                            colorscale: [
                                [0, '#ffffff'],
                                [0.25, '#c0ecc0'],
                                [0.5, '#7fba7a'],
                                [0.75, '#3f8f3f'],
                                [1, '#1E8449']
                            ],
                            text: textValues,
                            texttemplate: "%{{text}}",
                            showscale: true,
                            colorbar: {{
                                title: "Percentage"
                            }}
                        }}],
                        layout: {{
                            title: `Sector Sentiment Heatmap for ${{latestDate}}`,
                            xaxis: {{
                                title: 'Sentiment Category'
                            }},
                            yaxis: {{
                                title: 'Sector'
                            }},
                            margin: {{l: 150, r: 50, t: 80, b: 80}}
                        }}
                    }};

                    // Plot the figure
                    Plotly.react('visualization', figure.data, figure.layout);
                }}

                // Function to create a trend visualization
                function createTrendVisualization(filteredData) {{
                    // Get unique dates, sectors, and sentiment categories
                    const uniqueDates = [...new Set(filteredData.map(row => row.formatted_date))].sort();
                    const uniqueSectors = [...new Set(filteredData.map(row => row.sector))].sort();
                    const uniqueCategories = [...new Set(filteredData.map(row => row.sentiment_category))];

                    // Prepare traces for each sector and sentiment combination
                    const traces = [];

                    // For each selected sector
                    for (const sector of uniqueSectors) {{
                        // For each sentiment category
                        for (const category of uniqueCategories) {{
                            // Filter data for this sector and category
                            const sectorCategoryData = filteredData.filter(row => 
                                row.sector === sector && row.sentiment_category === category);

                            // Skip if no data
                            if (sectorCategoryData.length === 0) continue;

                            // Sort by date
                            sectorCategoryData.sort((a, b) => a.formatted_date.localeCompare(b.formatted_date));

                            // Extract dates and percentages
                            const dates = sectorCategoryData.map(row => row.formatted_date);
                            const percentages = sectorCategoryData.map(row => row.percentage);

                            // Create trace for this combination
                            traces.push({{
                                x: dates,
                                y: percentages,
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: `${{sector}} - ${{category}}`,
                                line: {{ width: 2 }},
                                hovertemplate: '%{{y:.1f}}%<extra>%{{fullData.name}}</extra>'
                            }});
                        }}
                    }}

                    // Create the figure
                    const figure = {{
                        data: traces,
                        layout: {{
                            title: 'Sector Sentiment Trend Over Time',
                            xaxis: {{
                                title: 'Date',
                                type: 'category'
                            }},
                            yaxis: {{
                                title: 'Percentage (%)',
                                range: [0, 100]
                            }},
                            hovermode: 'closest',
                            legend: {{
                                orientation: 'v',
                                xanchor: 'right',
                                yanchor: 'top',
                                x: 1.1, 
                                y: 1
                            }},
                            margin: {{ l: 60, r: 50, t: 80, b: 80 }}
                        }}
                    }};

                    // Plot the figure
                    Plotly.react('visualization', figure.data, figure.layout);
                }}
            </script>
        </body>
        </html>
        """

        return html

    def _generate_options(self, values, selected_idx=0, reverse=False):
        """
        Generate HTML option tags for a select element

        Args:
            values (list): List of option values
            selected_idx (int): Index of the initially selected option
            reverse (bool): Whether to reverse the order of options

        Returns:
            str: HTML option tags
        """
        if reverse:
            values = list(values)
            values.reverse()

        options = ""
        for i, value in enumerate(values):
            selected = ' selected' if i == selected_idx else ''
            options += f'<option value="{value}"{selected}>{value}</option>\n'

        return options

    def _create_initial_heatmap(self, trend_data):
        """
        Create an initial heatmap visualization for the most recent date

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Get the most recent date
        latest_date = trend_data['formatted_date'].max()
        latest_data = trend_data[trend_data['formatted_date'] == latest_date]

        # Get unique sectors and sentiment categories
        sectors = sorted(latest_data['sector'].unique())
        categories = self.sentiment_categories

        # Prepare data for heatmap
        z_data = []
        text_data = []

        for sector in sectors:
            sector_percentages = []
            sector_text = []

            for category in categories:
                # Get percentage for this sector and category
                row = latest_data[(latest_data['sector'] == sector) &
                                  (latest_data['sentiment_category'] == category)]

                percentage = row['percentage'].values[0] if len(row) > 0 else 0
                sector_percentages.append(percentage)
                sector_text.append(f"{percentage:.1f}%")

            z_data.append(sector_percentages)
            text_data.append(sector_text)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=categories,
            y=sectors,
            colorscale=[
                [0, '#ffffff'],
                [0.25, '#c0ecc0'],
                [0.5, '#7fba7a'],
                [0.75, '#3f8f3f'],
                [1, '#1E8449']
            ],
            text=text_data,
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Percentage")
        ))

        # Update layout
        fig.update_layout(
            title=f'Sector Sentiment Heatmap for {latest_date}',
            xaxis_title='Sentiment Category',
            yaxis_title='Sector',
            margin=dict(l=150, r=50, t=80, b=80)
        )

        return fig

    def _calculate_turnaround_candidates(self, trend_data):
        """
        Identify sectors showing the strongest turnaround potential in BOTH directions
        - Bullish Turnarounds: Moving from bearish to bullish (buy opportunities)
        - Bearish Turnarounds: Moving from bullish to bearish (sell/avoid signals)

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            dict: Information about turnaround candidates with explanations
        """
        # Get unique sectors and dates
        unique_sectors = sorted(trend_data['sector'].unique())
        unique_dates = sorted(trend_data['formatted_date'].unique())

        # Need at least 2 dates to calculate trend changes
        if len(unique_dates) < 2:
            return {"error": "Not enough historical data to identify turnaround candidates"}

        # Take the earliest and latest dates for comparison
        earliest_date = unique_dates[0]
        latest_date = unique_dates[-1]

        # For more accurate analysis, use a midpoint date if enough data
        mid_idx = len(unique_dates) // 2
        mid_date = unique_dates[mid_idx] if len(unique_dates) >= 3 else earliest_date

        # Prepare result structure
        bullish_turnarounds = []  # Bearish to Bullish
        bearish_turnarounds = []  # Bullish to Bearish

        # For each sector, analyze sentiment changes
        for sector in unique_sectors:
            sector_data = {}

            # Get earliest, mid, and latest data points
            early_data = trend_data[(trend_data['sector'] == sector) &
                                    (trend_data['formatted_date'] == earliest_date)]
            mid_data = trend_data[(trend_data['sector'] == sector) &
                                  (trend_data['formatted_date'] == mid_date)]
            late_data = trend_data[(trend_data['sector'] == sector) &
                                   (trend_data['formatted_date'] == latest_date)]

            # Skip if missing data
            if early_data.empty or late_data.empty:
                continue

            # Calculate sentiment values for both points in time
            early_bullish = early_data[early_data['sentiment_category'].isin(['LONG', 'ACCUMULATION'])][
                'percentage'].sum()
            early_neutral = early_data[early_data['sentiment_category'] == 'NEUTRAL']['percentage'].sum()
            early_bearish = early_data[early_data['sentiment_category'].isin(['DISTRIBUTION', 'SHORT'])][
                'percentage'].sum()

            mid_bullish = mid_data[mid_data['sentiment_category'].isin(['LONG', 'ACCUMULATION'])]['percentage'].sum()
            mid_neutral = mid_data[mid_data['sentiment_category'] == 'NEUTRAL']['percentage'].sum()
            mid_bearish = mid_data[mid_data['sentiment_category'].isin(['DISTRIBUTION', 'SHORT'])]['percentage'].sum()

            late_bullish = late_data[late_data['sentiment_category'].isin(['LONG', 'ACCUMULATION'])]['percentage'].sum()
            late_neutral = late_data[late_data['sentiment_category'] == 'NEUTRAL']['percentage'].sum()
            late_bearish = late_data[late_data['sentiment_category'].isin(['DISTRIBUTION', 'SHORT'])][
                'percentage'].sum()

            # Calculate net sentiment score (bullish - bearish)
            early_score = early_bullish - early_bearish
            mid_score = mid_bullish - mid_bearish
            late_score = late_bullish - late_bearish

            # Calculate score changes - positive values indicate improvement towards bullish
            early_to_mid_change = mid_score - early_score
            mid_to_late_change = late_score - mid_score
            overall_change = late_score - early_score

            # Calculate rate of change in bullish and bearish percentages
            bullish_change_rate = late_bullish - early_bullish
            bearish_change_rate = early_bearish - late_bearish  # Inverted so positive is good
            neutral_change_rate = early_neutral - late_neutral  # Reduction in neutral

            # BULLISH TURNAROUND DETECTION (Bearish to Bullish)
            # Only consider sectors that were bearish and are improving
            if early_score < -10 and overall_change > 5:  # Was bearish, now improving
                # Calculate bullish turnaround score
                bullish_turnaround_score = (
                        mid_to_late_change * 2.0 +  # Recent improvement is most important
                        early_to_mid_change * 1.0 +  # Earlier improvement
                        bullish_change_rate * 1.5 +  # Increase in bullish sentiment
                        bearish_change_rate * 1.5 +  # Decrease in bearish sentiment
                        neutral_change_rate * 0.5  # Reduction in neutral (less fence-sitting)
                )

                # Add logical factors for explanation
                factors = []

                if mid_to_late_change > 3:
                    factors.append({
                        "factor": "Strong recent momentum",
                        "description": f"Sentiment improved by {mid_to_late_change:.1f}% in the recent period",
                        "weight": 2.0,
                        "type": "positive"
                    })

                if bullish_change_rate > 10:
                    factors.append({
                        "factor": "Significant bullish growth",
                        "description": f"Bullish sentiment increased from {early_bullish:.1f}% to {late_bullish:.1f}%",
                        "weight": 1.5,
                        "type": "positive"
                    })

                if bearish_change_rate > 10:
                    factors.append({
                        "factor": "Major bearish reduction",
                        "description": f"Bearish sentiment decreased from {early_bearish:.1f}% to {late_bearish:.1f}%",
                        "weight": 1.5,
                        "type": "positive"
                    })

                if late_bullish / max(late_bearish, 1) >= 0.8 and late_bullish / max(late_bearish, 1) < 1.5:
                    factors.append({
                        "factor": "Approaching equilibrium",
                        "description": "The sector is approaching balance between bullish and bearish forces, often a turning point",
                        "weight": 1.0,
                        "type": "neutral"
                    })

                # Save sector turnaround data if it has positive score and at least 2 factors
                if bullish_turnaround_score > 3 and len(factors) >= 2:
                    bullish_turnarounds.append({
                        "sector": sector,
                        "turnaround_score": bullish_turnaround_score,
                        "turnaround_type": "BULLISH",
                        "direction": "Bearish → Bullish",
                        "early_score": early_score,
                        "late_score": late_score,
                        "current_bullish": late_bullish,
                        "current_bearish": late_bearish,
                        "current_neutral": late_neutral,
                        "bullish_change": bullish_change_rate,
                        "bearish_change": bearish_change_rate,
                        "factors": factors
                    })

            # BEARISH TURNAROUND DETECTION (Bullish to Bearish)
            # Only consider sectors that were bullish and are deteriorating
            if early_score > 10 and overall_change < -5:  # Was bullish, now declining
                # Calculate bearish turnaround score (higher = more concerning)
                bearish_turnaround_score = (
                        abs(mid_to_late_change) * 2.0 +  # Recent deterioration
                        abs(early_to_mid_change) * 1.0 +  # Earlier deterioration
                        abs(bullish_change_rate) * 1.5 +  # Decrease in bullish sentiment
                        (late_bearish - early_bearish) * 1.5 +  # Increase in bearish sentiment
                        abs(neutral_change_rate) * 0.5  # Change in neutral sentiment
                )

                # Add logical factors for explanation
                factors = []

                if mid_to_late_change < -3:
                    factors.append({
                        "factor": "Recent momentum deterioration",
                        "description": f"Sentiment declined by {abs(mid_to_late_change):.1f}% in the recent period",
                        "weight": 2.0,
                        "type": "negative"
                    })

                if bullish_change_rate < -10:
                    factors.append({
                        "factor": "Significant bullish decline",
                        "description": f"Bullish sentiment decreased from {early_bullish:.1f}% to {late_bullish:.1f}%",
                        "weight": 1.5,
                        "type": "negative"
                    })

                if (late_bearish - early_bearish) > 10:
                    factors.append({
                        "factor": "Rising bearish sentiment",
                        "description": f"Bearish sentiment increased from {early_bearish:.1f}% to {late_bearish:.1f}%",
                        "weight": 1.5,
                        "type": "negative"
                    })

                if late_bearish / max(late_bullish, 1) > 1.2:
                    factors.append({
                        "factor": "Bearish dominance emerging",
                        "description": f"Bearish sentiment now exceeds bullish by a significant margin (ratio: {late_bearish / max(late_bullish, 1):.2f})",
                        "weight": 1.0,
                        "type": "negative"
                    })

                # Save sector turnaround data if it has significant score and at least 2 factors
                if bearish_turnaround_score > 3 and len(factors) >= 2:
                    bearish_turnarounds.append({
                        "sector": sector,
                        "turnaround_score": bearish_turnaround_score,
                        "turnaround_type": "BEARISH",
                        "direction": "Bullish → Bearish",
                        "early_score": early_score,
                        "late_score": late_score,
                        "current_bullish": late_bullish,
                        "current_bearish": late_bearish,
                        "current_neutral": late_neutral,
                        "bullish_change": bullish_change_rate,
                        "bearish_change": -(late_bearish - early_bearish),  # Make negative for consistency
                        "factors": factors
                    })

        # Sort both lists by turnaround score, highest first
        bullish_turnarounds.sort(key=lambda x: x["turnaround_score"], reverse=True)
        bearish_turnarounds.sort(key=lambda x: x["turnaround_score"], reverse=True)

        # Generate explanations for both types
        result = {
            "bullish_candidates": bullish_turnarounds[:3],  # Top 3 bullish turnarounds
            "bearish_candidates": bearish_turnarounds[:3],  # Top 3 bearish turnarounds
            "explanation": self._generate_enhanced_turnaround_explanation(
                bullish_turnarounds[:3], bearish_turnarounds[:3])
        }

        return result

    def _generate_enhanced_turnaround_explanation(self, bullish_candidates, bearish_candidates):
        """
        Generate a detailed explanation for both bullish and bearish turnaround candidates

        Args:
            bullish_candidates (list): List of bullish turnaround candidates
            bearish_candidates (list): List of bearish turnaround candidates

        Returns:
            str: HTML-formatted explanation
        """
        html = "<h3>Sector Turnaround Analysis</h3>"

        # Bullish Turnarounds Section
        if bullish_candidates:
            html += "<h4 style='color: #1E8449; margin-top: 20px;'>📈 Bullish Turnaround Opportunities</h4>"
            html += "<p style='margin-bottom: 15px;'><em>Sectors showing recovery from bearish sentiment (potential buy opportunities):</em></p>"

            for i, candidate in enumerate(bullish_candidates):
                html += f"<div class='turnaround-candidate' style='border-left: 4px solid #1E8449; margin-bottom: 15px;'>"
                html += f"<h5>{i + 1}. {candidate['sector']} ({candidate['direction']})</h5>"
                html += f"<p><strong>Turnaround Strength:</strong> {candidate['turnaround_score']:.1f}/10</p>"
                html += "<p><strong>Positive indicators:</strong></p>"
                html += "<ul>"

                # Sort factors by weight
                sorted_factors = sorted(candidate['factors'], key=lambda x: x['weight'], reverse=True)
                for factor in sorted_factors:
                    html += f"<li>{factor['description']}</li>"

                html += "</ul>"
                html += "<p><strong>Sentiment shift:</strong> "
                html += f"Net score changed from {candidate['early_score']:.1f}% to {candidate['late_score']:.1f}% "
                html += f"(Current: {candidate['current_bullish']:.1f}% bullish, {candidate['current_bearish']:.1f}% bearish)"
                html += "</p></div>"
        else:
            html += "<h4 style='color: #1E8449; margin-top: 20px;'>📈 Bullish Turnaround Opportunities</h4>"
            html += "<p><em>No significant bullish turnarounds identified in the current data.</em></p>"

        # Bearish Turnarounds Section
        if bearish_candidates:
            html += "<h4 style='color: #C0392B; margin-top: 25px;'>📉 Bearish Turnaround Warnings</h4>"
            html += "<p style='margin-bottom: 15px;'><em>Sectors showing deterioration from bullish sentiment (potential sell/avoid signals):</em></p>"

            for i, candidate in enumerate(bearish_candidates):
                html += f"<div class='turnaround-candidate' style='border-left: 4px solid #C0392B; margin-bottom: 15px;'>"
                html += f"<h5>{i + 1}. {candidate['sector']} ({candidate['direction']})</h5>"
                html += f"<p><strong>Warning Strength:</strong> {candidate['turnaround_score']:.1f}/10</p>"
                html += "<p><strong>Warning signs:</strong></p>"
                html += "<ul>"

                # Sort factors by weight
                sorted_factors = sorted(candidate['factors'], key=lambda x: x['weight'], reverse=True)
                for factor in sorted_factors:
                    html += f"<li>{factor['description']}</li>"

                html += "</ul>"
                html += "<p><strong>Sentiment shift:</strong> "
                html += f"Net score changed from {candidate['early_score']:.1f}% to {candidate['late_score']:.1f}% "
                html += f"(Current: {candidate['current_bullish']:.1f}% bullish, {candidate['current_bearish']:.1f}% bearish)"
                html += "</p></div>"
        else:
            html += "<h4 style='color: #C0392B; margin-top: 25px;'>📉 Bearish Turnaround Warnings</h4>"
            html += "<p><em>No significant bearish turnarounds identified in the current data.</em></p>"

        return html

    def _generate_turnaround_explanation(self, candidates):
        """
        Generate a detailed explanation for the top turnaround candidates

        Args:
            candidates (list): List of top turnaround candidates

        Returns:
            str: HTML-formatted explanation
        """
        if not candidates:
            return "<p>No strong turnaround candidates identified in the current data.</p>"

        # Generate HTML for explanation
        html = "<h3>Top Turnaround Candidates</h3>"

        for i, candidate in enumerate(candidates):
            html += f"<div class='turnaround-candidate'>"
            html += f"<h4>{i + 1}. {candidate['sector']}</h4>"
            html += "<p><strong>Why this sector shows turnaround potential:</strong></p>"
            html += "<ul>"

            # Sort factors by weight
            sorted_factors = sorted(candidate['factors'], key=lambda x: x['weight'], reverse=True)
            for factor in sorted_factors:
                html += f"<li>{factor['description']}</li>"

            # Add current sentiment breakdown
            html += "</ul>"
            html += "<p><strong>Current sentiment breakdown:</strong> "
            html += f"Bullish: {candidate['current_bullish']:.1f}% | "
            html += f"Neutral: {candidate['current_neutral']:.1f}% | "
            html += f"Bearish: {candidate['current_bearish']:.1f}%"
            html += "</p>"
            html += "</div>"

        return html

    def _calculate_top_investment_sectors(self, trend_data):
        """
        Identify the top sectors for investment based on bullish sentiment and other factors

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            dict: Information about top sectors with explanations
        """
        # Get the latest date data
        latest_date = trend_data['formatted_date'].max()
        latest_data = trend_data[trend_data['formatted_date'] == latest_date]

        # Get unique sectors
        unique_sectors = sorted(latest_data['sector'].unique())

        # Prepare result structure
        top_sectors = []

        # For each sector, calculate investment metrics
        for sector in unique_sectors:
            sector_data = latest_data[latest_data['sector'] == sector]

            # Skip if missing data
            if sector_data.empty:
                continue

            # Calculate sentiment values
            bullish = sector_data[sector_data['sentiment_category'].isin(['LONG', 'ACCUMULATION'])]['percentage'].sum()
            neutral = sector_data[sector_data['sentiment_category'] == 'NEUTRAL']['percentage'].sum()
            bearish = sector_data[sector_data['sentiment_category'].isin(['DISTRIBUTION', 'SHORT'])]['percentage'].sum()

            # Calculate bullish-to-bearish ratio (avoid division by zero)
            bull_bear_ratio = bullish / max(bearish, 1.0)

            # Calculate net sentiment score
            net_score = bullish - bearish

            # Calculate a simple investment score
            investment_score = bull_bear_ratio * 2.0 + net_score / 10.0

            # Create factors for explanation
            factors = []

            if bull_bear_ratio > 3.0:
                factors.append({
                    "factor": "Strong bullish-to-bearish ratio",
                    "description": f"Bullish-to-bearish ratio of {bull_bear_ratio:.1f} (higher ratios indicate stronger consensus)",
                    "weight": 2.0
                })
            elif bull_bear_ratio > 1.5:
                factors.append({
                    "factor": "Positive bullish-to-bearish ratio",
                    "description": f"Bullish-to-bearish ratio of {bull_bear_ratio:.1f}",
                    "weight": 1.5
                })

            if bullish > 70:
                factors.append({
                    "factor": "High bullish sentiment",
                    "description": f"Bullish sentiment is very high at {bullish:.1f}%",
                    "weight": 2.0
                })
            elif bullish > 50:
                factors.append({
                    "factor": "Majority bullish sentiment",
                    "description": f"Bullish sentiment is {bullish:.1f}%, showing majority support",
                    "weight": 1.5
                })

            if bearish < 20:
                factors.append({
                    "factor": "Very low bearish sentiment",
                    "description": f"Bearish sentiment is exceptionally low at {bearish:.1f}%",
                    "weight": 1.5
                })
            elif bearish < 30:
                factors.append({
                    "factor": "Low bearish sentiment",
                    "description": f"Bearish sentiment is relatively low at {bearish:.1f}%",
                    "weight": 1.0
                })

            if neutral < 10:
                factors.append({
                    "factor": "Strong conviction",
                    "description": f"Neutral sentiment is very low at {neutral:.1f}%, indicating strong market conviction",
                    "weight": 1.0
                })

            # Add sector data if it has a positive investment score
            if investment_score > 1.0:
                top_sectors.append({
                    "sector": sector,
                    "investment_score": investment_score,
                    "bull_bear_ratio": bull_bear_ratio,
                    "current_bullish": bullish,
                    "current_bearish": bearish,
                    "current_neutral": neutral,
                    "net_score": net_score,
                    "factors": factors
                })

        # Sort by investment score, highest first
        top_sectors.sort(key=lambda x: x["investment_score"], reverse=True)

        # Generate explanations
        result = {
            "top_sectors": top_sectors[:3],  # Top 3 sectors
            "explanation": self._generate_top_sectors_explanation(top_sectors[:3] if top_sectors else [])
        }

        return result

    def _calculate_sectors_to_avoid(self, trend_data):
        """
        Identify sectors to avoid based on bearish sentiment and other negative factors

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            dict: Information about sectors to avoid with explanations
        """
        # Get the latest date data
        latest_date = trend_data['formatted_date'].max()
        latest_data = trend_data[trend_data['formatted_date'] == latest_date]

        # If we have enough data, also get the previous date for trend
        unique_dates = sorted(trend_data['formatted_date'].unique())
        if len(unique_dates) >= 2:
            prev_date = unique_dates[-2]
            prev_data = trend_data[trend_data['formatted_date'] == prev_date]
        else:
            prev_data = pd.DataFrame()

        # Get unique sectors
        unique_sectors = sorted(latest_data['sector'].unique())

        # Prepare result structure
        sectors_to_avoid = []

        # For each sector, calculate avoidance metrics
        for sector in unique_sectors:
            # Latest data
            sector_latest = latest_data[latest_data['sector'] == sector]

            # Previous data (if available)
            if not prev_data.empty:
                sector_prev = prev_data[prev_data['sector'] == sector]
            else:
                sector_prev = pd.DataFrame()

            # Skip if missing latest data
            if sector_latest.empty:
                continue

            # Calculate latest sentiment values
            latest_bullish = sector_latest[sector_latest['sentiment_category'].isin(['LONG', 'ACCUMULATION'])][
                'percentage'].sum()
            latest_neutral = sector_latest[sector_latest['sentiment_category'] == 'NEUTRAL']['percentage'].sum()
            latest_bearish = sector_latest[sector_latest['sentiment_category'].isin(['DISTRIBUTION', 'SHORT'])][
                'percentage'].sum()

            # Calculate bearish-to-bullish ratio (avoid division by zero)
            bear_bull_ratio = latest_bearish / max(latest_bullish, 1.0)

            # Calculate net sentiment score (negative is worse)
            net_score = latest_bullish - latest_bearish

            # Calculate trend if previous data available
            sentiment_worsening = False
            if not sector_prev.empty:
                prev_bullish = sector_prev[sector_prev['sentiment_category'].isin(['LONG', 'ACCUMULATION'])][
                    'percentage'].sum()
                prev_bearish = sector_prev[sector_prev['sentiment_category'].isin(['DISTRIBUTION', 'SHORT'])][
                    'percentage'].sum()

                # Check if sentiment is worsening
                prev_net = prev_bullish - prev_bearish
                sentiment_worsening = (net_score < prev_net)

            # Calculate avoidance score - higher means more reason to avoid
            avoidance_score = bear_bull_ratio * 1.5 - net_score / 10.0
            if sentiment_worsening:
                avoidance_score += 1.0

            # Create factors for explanation
            factors = []

            if bear_bull_ratio > 3.0:
                factors.append({
                    "factor": "Strong bearish-to-bullish ratio",
                    "description": f"Bearish-to-bullish ratio of {bear_bull_ratio:.1f} (higher ratios indicate stronger negative consensus)",
                    "weight": 2.0
                })
            elif bear_bull_ratio > 1.5:
                factors.append({
                    "factor": "Negative sentiment balance",
                    "description": f"Bearish-to-bullish ratio of {bear_bull_ratio:.1f}",
                    "weight": 1.5
                })

            if latest_bearish > 70:
                factors.append({
                    "factor": "High bearish sentiment",
                    "description": f"Bearish sentiment is very high at {latest_bearish:.1f}%",
                    "weight": 2.0
                })
            elif latest_bearish > 50:
                factors.append({
                    "factor": "Majority bearish sentiment",
                    "description": f"Bearish sentiment is {latest_bearish:.1f}%, showing majority negative outlook",
                    "weight": 1.5
                })

            if latest_bullish < 20:
                factors.append({
                    "factor": "Very low bullish sentiment",
                    "description": f"Bullish sentiment is exceptionally low at {latest_bullish:.1f}%",
                    "weight": 1.5
                })

            if sentiment_worsening:
                factors.append({
                    "factor": "Deteriorating sentiment",
                    "description": "Sentiment has been worsening in recent data",
                    "weight": 1.0
                })

            # Add sector data if it has a substantial avoidance score
            if avoidance_score > 1.0 and len(factors) >= 1:
                sectors_to_avoid.append({
                    "sector": sector,
                    "avoidance_score": avoidance_score,
                    "bear_bull_ratio": bear_bull_ratio,
                    "current_bullish": latest_bullish,
                    "current_bearish": latest_bearish,
                    "current_neutral": latest_neutral,
                    "net_score": net_score,
                    "factors": factors
                })

        # Sort by avoidance score, highest first
        sectors_to_avoid.sort(key=lambda x: x["avoidance_score"], reverse=True)

        # Generate explanations
        result = {
            "avoid_sectors": sectors_to_avoid[:3],  # Top 3 sectors to avoid
            "explanation": self._generate_avoid_sectors_explanation(sectors_to_avoid[:3] if sectors_to_avoid else [])
        }

        return result

    def _generate_top_sectors_explanation(self, top_sectors):
        """
        Generate an HTML-formatted explanation for the top investment sectors

        Args:
            top_sectors (list): List of top sectors for investment

        Returns:
            str: HTML-formatted explanation
        """
        if not top_sectors:
            return "<p>No standout investment sectors identified in the current data.</p>"

        # Generate HTML for explanation
        html = "<h3>Top Sectors for Investment</h3>"

        for i, sector in enumerate(top_sectors):
            html += f"<div class='top-sector'>"
            html += f"<h4>{i + 1}. {sector['sector']}</h4>"
            html += f"<p><strong>Bullish-to-Bearish Ratio:</strong> {sector['bull_bear_ratio']:.2f}</p>"
            html += "<p><strong>Why this sector is attractive:</strong></p>"
            html += "<ul>"

            # Sort factors by weight
            sorted_factors = sorted(sector['factors'], key=lambda x: x['weight'], reverse=True)
            for factor in sorted_factors:
                html += f"<li>{factor['description']}</li>"

            # Add current sentiment breakdown
            html += "</ul>"
            html += "<p><strong>Current sentiment breakdown:</strong> "
            html += f"Bullish: {sector['current_bullish']:.1f}% | "
            html += f"Neutral: {sector['current_neutral']:.1f}% | "
            html += f"Bearish: {sector['current_bearish']:.1f}%"
            html += "</p>"
            html += "</div>"

        return html

    def _generate_avoid_sectors_explanation(self, avoid_sectors):
        """
        Generate an HTML-formatted explanation for sectors to avoid

        Args:
            avoid_sectors (list): List of sectors to avoid

        Returns:
            str: HTML-formatted explanation
        """
        if not avoid_sectors:
            return "<p>No clear sectors to avoid identified in the current data.</p>"

        # Generate HTML for explanation
        html = "<h3>Sectors to Approach with Caution</h3>"

        for i, sector in enumerate(avoid_sectors):
            html += f"<div class='avoid-sector'>"
            html += f"<h4>{i + 1}. {sector['sector']}</h4>"
            html += f"<p><strong>Bearish-to-Bullish Ratio:</strong> {sector['bear_bull_ratio']:.2f}</p>"
            html += "<p><strong>Warning signs:</strong></p>"
            html += "<ul>"

            # Sort factors by weight
            sorted_factors = sorted(sector['factors'], key=lambda x: x['weight'], reverse=True)
            for factor in sorted_factors:
                html += f"<li>{factor['description']}</li>"

            # Add current sentiment breakdown
            html += "</ul>"
            html += "<p><strong>Current sentiment breakdown:</strong> "
            html += f"Bullish: {sector['current_bullish']:.1f}% | "
            html += f"Neutral: {sector['current_neutral']:.1f}% | "
            html += f"Bearish: {sector['current_bearish']:.1f}%"
            html += "</p>"
            html += "</div>"

        return html

    def _calculate_market_trend(self, trend_data):
        """
        Calculate overall market trend based on sector sentiment data

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            dict: Market trend data and summary
        """
        # Convert date column to datetime if it's not already
        if trend_data['date'].dtype == 'object':
            trend_data['date'] = pd.to_datetime(trend_data['date'], format=self.date_format)

        # Get unique dates in sorted order
        dates = sorted(trend_data['date'].unique())

        # Prepare result structure
        market_trend = {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'bullish_percentage': [],
            'neutral_percentage': [],
            'bearish_percentage': [],
            'sentiment_score': []
        }

        # For each date, calculate the average sentiment across all sectors
        for date in dates:
            date_data = trend_data[trend_data['date'] == date]

            # Group by sector and sentiment category, and sum the percentages
            sector_sentiment = date_data.groupby(['sector', 'sentiment_category'])['percentage'].sum().unstack().fillna(
                0)

            # Calculate bullish, neutral, and bearish percentages for each sector
            sector_sentiment['bullish'] = sector_sentiment.get('LONG', 0) + sector_sentiment.get('ACCUMULATION', 0)
            sector_sentiment['neutral'] = sector_sentiment.get('NEUTRAL', 0)
            sector_sentiment['bearish'] = sector_sentiment.get('DISTRIBUTION', 0) + sector_sentiment.get('SHORT', 0)

            # Calculate sentiment score for each sector
            sector_sentiment['score'] = (
                                                sector_sentiment.get('LONG', 0) * 2 +
                                                sector_sentiment.get('ACCUMULATION', 0) * 1 +
                                                sector_sentiment.get('NEUTRAL', 0) * 0 +
                                                sector_sentiment.get('DISTRIBUTION', 0) * -1 +
                                                sector_sentiment.get('SHORT', 0) * -2
                                        ) / 100  # Normalize by dividing by 100

            # Calculate average percentages across all sectors
            avg_bullish = sector_sentiment['bullish'].mean()
            avg_neutral = sector_sentiment['neutral'].mean()
            avg_bearish = sector_sentiment['bearish'].mean()
            avg_score = sector_sentiment['score'].mean()

            # Add to result
            market_trend['bullish_percentage'].append(avg_bullish)
            market_trend['neutral_percentage'].append(avg_neutral)
            market_trend['bearish_percentage'].append(avg_bearish)
            market_trend['sentiment_score'].append(avg_score)

        # Generate market trend summary
        summary = self._generate_market_summary(market_trend)
        market_trend['summary'] = summary

        return market_trend


    def _generate_market_summary(self, market_trend):
        """
        Generate a summary of the market trend based on calculations

        Args:
            market_trend (dict): Market trend data

        Returns:
            str: Market trend summary
        """
        # Get the most recent data points
        latest_score = market_trend['sentiment_score'][-1]
        latest_bullish = market_trend['bullish_percentage'][-1]
        latest_bearish = market_trend['bearish_percentage'][-1]

        # Check if we have at least 3 data points to identify a trend
        if len(market_trend['sentiment_score']) >= 3:
            recent_scores = market_trend['sentiment_score'][-3:]

            # Define trend direction
            if recent_scores[-1] > recent_scores[-2] > recent_scores[-3]:
                trend_direction = "BULLISH"
            elif recent_scores[-1] < recent_scores[-2] < recent_scores[-3]:
                trend_direction = "BEARISH"
            elif recent_scores[-1] > recent_scores[-3]:
                trend_direction = "IMPROVING"
            elif recent_scores[-1] < recent_scores[-3]:
                trend_direction = "DETERIORATING"
            else:
                trend_direction = "SIDEWAYS"

            # Generate summary text
            if latest_score > 0.5:
                strength = "STRONGLY BULLISH"
            elif latest_score > 0:
                strength = "MODERATELY BULLISH"
            elif latest_score > -0.5:
                strength = "MODERATELY BEARISH"
            else:
                strength = "STRONGLY BEARISH"

            summary = f"MARKET TREND: {strength} AND {trend_direction}"

            # Add additional analysis
            if latest_bullish > 60:
                summary += f" | HIGH BULLISH SENTIMENT ({latest_bullish:.1f}%)"
            elif latest_bearish > 60:
                summary += f" | HIGH BEARISH SENTIMENT ({latest_bearish:.1f}%)"

            return summary
        else:
            # Not enough data points for trend analysis
            if latest_score > 0.5:
                return "MARKET TREND: STRONGLY BULLISH (LIMITED DATA)"
            elif latest_score > 0:
                return "MARKET TREND: MODERATELY BULLISH (LIMITED DATA)"
            elif latest_score > -0.5:
                return "MARKET TREND: MODERATELY BEARISH (LIMITED DATA)"
            else:
                return "MARKET TREND: STRONGLY BEARISH (LIMITED DATA)"

    def _generate_allinone_visualization(self, trend_data):
        """
        Generate a single HTML page with charts for all sectors

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data

        Returns:
            str: Path to the generated HTML file
        """
        # Format dates for display
        trend_data['formatted_date'] = pd.to_datetime(trend_data['date'], format=self.date_format).dt.strftime(
            '%Y-%m-%d')
        # Add shorter date format for chart display
        trend_data['short_date'] = pd.to_datetime(trend_data['date'], format=self.date_format).dt.strftime('%d%m')

        # Get unique sectors
        unique_sectors = sorted(trend_data['sector'].unique())

        # Calculate overall market trend
        market_trend = self._calculate_market_trend(trend_data)

        # Create the HTML file
        html_filename = f"sector_sentiment_allinone_{self.start_date.strftime(self.date_format)}_{self.end_date.strftime(self.date_format)}.html"
        html_path = os.path.join(self.output_dir, html_filename)

        # Generate the HTML content
        html_content = self._generate_allinone_html(trend_data, unique_sectors, market_trend)

        # Write to file
        os.makedirs(self.output_dir, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"All-in-one visualization generated at: {html_path}")
        return html_path

    def _generate_allinone_html(self, trend_data, unique_sectors, market_trend):
        """
        Generate HTML content for the all-in-one visualization with all sector charts

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data
            unique_sectors (list): List of unique sectors
            market_trend (dict): Market trend data including summary

        Returns:
            str: HTML content
        """
        # Calculate analysis sections
        turnaround_analysis = self._calculate_turnaround_candidates(trend_data)
        top_sectors_analysis = self._calculate_top_investment_sectors(trend_data)
        avoid_sectors_analysis = self._calculate_sectors_to_avoid(trend_data)

        # Create market trend chart JSON
        market_trend_fig = self._create_market_trend_chart(market_trend)
        market_trend_json = json.dumps(market_trend_fig.to_dict())

        # Generate sector chart JSONs
        sector_charts = []
        for sector in unique_sectors:
            # Create sector chart
            sector_fig = self._create_sector_trend_chart(trend_data, sector)
            sector_charts.append({
                'sector': sector,
                'chart_json': json.dumps(sector_fig.to_dict())
            })

        # HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sector Sentiment Trend Analysis - All Sectors</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                :root {{
                    --primary-color: #2C3E50;
                    --secondary-color: #3498DB;
                    --accent-color: #F39C12;
                    --neutral-color: #ECF0F1;
                    --bullish-color: #1E8449;
                    --bearish-color: #C0392B;
                    --neutral-color-alt: #F7DC6F;
                }}

                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}

                body {{
                    background-color: var(--neutral-color);
                    color: #333;
                    line-height: 1.6;
                    padding: 20px;
                }}

                .container {{
                    max-width: 1600px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                }}

                h1 {{
                    color: var(--primary-color);
                    margin-bottom: 20px;
                    text-align: center;
                }}

                .summary-box {{
                    background-color: #4a69bd;
                    color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    text-align: center;
                }}

                .summary-text {{
                    font-size: 24px;
                    font-weight: bold;
                }}

                .date-filter {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    text-align: right;
                }}

                .market-chart {{
                    height: 400px;
                    margin-bottom: 40px;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}

                /* Investment Recommendations Styling */
                .recommendations-container {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 30px;
                }}

                .top-sectors-box {{
                    background-color: #d5f4e6;
                    border-left: 5px solid var(--bullish-color);
                    padding: 20px;
                    border-radius: 8px;
                }}

                .avoid-sectors-box {{
                    background-color: #fdeaea;
                    border-left: 5px solid var(--bearish-color);
                    padding: 20px;
                    border-radius: 8px;
                }}

                .top-sectors-box h3, .avoid-sectors-box h3 {{
                    margin-bottom: 15px;
                    color: var(--primary-color);
                }}

                .top-sector, .avoid-sector {{
                    margin-bottom: 15px;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}

                .top-sector h4 {{
                    color: var(--bullish-color);
                    margin-bottom: 5px;
                }}

                .avoid-sector h4 {{
                    color: var(--bearish-color);
                    margin-bottom: 5px;
                }}

                /* Turnaround Candidates Styling */
                .turnaround-box {{
                    background-color: #f8f9fa;
                    border: 2px solid #F39C12;
                    padding: 25px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                
                .turnaround-box h3 {{
                    margin-bottom: 20px;
                    color: var(--primary-color);
                    text-align: center;
                    font-size: 24px;
                }}
                
                .turnaround-box h4 {{
                    margin: 20px 0 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 18px;
                }}
                
                .turnaround-box h4:first-of-type {{
                    background-color: #d5f4e6;
                }}
                
                .turnaround-box h4:last-of-type {{
                    background-color: #fdeaea;
                }}
                
                .turnaround-candidate {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    position: relative;
                }}
                
                .turnaround-candidate h5 {{
                    color: var(--primary-color);
                    margin-bottom: 10px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                
                .turnaround-candidate ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                
                .turnaround-candidate li {{
                    margin-bottom: 8px;
                    line-height: 1.4;
                }}
                
                .turnaround-candidate p {{
                    margin: 8px 0;
                }}
                
                /* Responsive adjustments */
                @media (max-width: 768px) {{
                    .turnaround-candidate {{
                        padding: 12px;
                    }}
                    
                    .turnaround-box {{
                        padding: 15px;
                    }}
                }}

                .sector-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(700px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}

                .sector-chart {{
                    height: 350px;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}

                .chart-title {{
                    font-size: 18px;
                    margin-bottom: 10px;
                    color: var(--primary-color);
                    font-weight: bold;
                }}

                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                    padding: 20px;
                    border-top: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>SECTOR PHASE SENTIMENT ANALYSIS</h1>

                <div class="date-filter">
                    <strong>Date Filter:</strong> {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
                </div>

                <div class="summary-box">
                    <div class="summary-text">{market_trend['summary']}</div>
                </div>

                <!-- Market Trend Chart -->
                <div class="chart-title">Overall Market Sentiment Trend</div>
                <div id="market-chart" class="market-chart"></div>

                <!-- Investment Recommendations Section -->
                <div class="recommendations-container">
                    <div class="top-sectors-box">
                        {top_sectors_analysis['explanation']}
                    </div>
                    <div class="avoid-sectors-box">
                        {avoid_sectors_analysis['explanation']}
                    </div>
                </div>

                <!-- Turnaround Candidates Section -->
                <div class="turnaround-box">
                    {turnaround_analysis['explanation']}
                </div>

                <!-- Sector Charts Grid -->
                <div class="sector-grid">
        """

        # Add sector chart placeholders
        for i, sector_chart in enumerate(sector_charts):
            sector = sector_chart['sector']
            html += f"""
                    <div>
                        <div class="chart-title">{sector} - Sentiment Trend Over Time</div>
                        <div id="sector-chart-{i}" class="sector-chart"></div>
                    </div>
            """

        # Complete HTML
        html += f"""
                </div>

                <div class="footer">
                    <p>Generated by Sector Sentiment Trend Generator | Data Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}</p>
                </div>
            </div>

            <script>
                // Create the market trend chart
                const marketTrendFig = {market_trend_json};
                Plotly.newPlot('market-chart', marketTrendFig.data, marketTrendFig.layout);

                // Create all sector charts
        """

        # Add JavaScript to create each sector chart
        for i, sector_chart in enumerate(sector_charts):
            html += f"""
                // Create chart for {sector_chart['sector']}
                const sectorFig{i} = {sector_chart['chart_json']};
                Plotly.newPlot('sector-chart-{i}', sectorFig{i}.data, sectorFig{i}.layout);
            """

        # Complete script and HTML
        html += """
                // Add resize handler
                window.addEventListener('resize', function() {
                    Plotly.Plots.resize('market-chart');
        """

        # Add resize handlers for each sector chart
        for i in range(len(sector_charts)):
            html += f"""
                    Plotly.Plots.resize('sector-chart-{i}');
            """

        # Complete the script and HTML
        html += """
                });
            </script>
        </body>
        </html>
        """

        return html

    def _create_market_trend_chart(self, market_trend):
        """
        Create a chart showing overall market sentiment trend

        Args:
            market_trend (dict): Market trend data

        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Convert dates to shorter format for display
        short_dates = [datetime.strptime(date, '%Y-%m-%d').strftime('%d%m') for date in market_trend['dates']]

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add sentiment score line
        fig.add_trace(
            go.Scatter(
                x=short_dates,
                y=market_trend['sentiment_score'],
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='black', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        # Add bullish percentage
        fig.add_trace(
            go.Scatter(
                x=short_dates,
                y=market_trend['bullish_percentage'],
                mode='lines+markers',
                name='Bullish %',
                line=dict(color='#1E8449', width=2),
                marker=dict(size=6)
            ),
            secondary_y=False
        )

        # Add neutral percentage
        fig.add_trace(
            go.Scatter(
                x=short_dates,
                y=market_trend['neutral_percentage'],
                mode='lines+markers',
                name='Neutral %',
                line=dict(color='#F7DC6F', width=2),
                marker=dict(size=6)
            ),
            secondary_y=False
        )

        # Add bearish percentage
        fig.add_trace(
            go.Scatter(
                x=short_dates,
                y=market_trend['bearish_percentage'],
                mode='lines+markers',
                name='Bearish %',
                line=dict(color='#C0392B', width=2),
                marker=dict(size=6)
            ),
            secondary_y=False
        )

        # Update layout
        fig.update_layout(
            title='Market Sentiment Trend',
            xaxis=dict(
                title='Date',
                tickmode='array',
                tickvals=short_dates,
                ticktext=short_dates
            ),
            yaxis=dict(
                title='Percentage (%)',
                range=[0, 100]
            ),
            yaxis2=dict(
                title='Sentiment Score',
                titlefont=dict(color='black'),
                tickfont=dict(color='black'),
                range=[-2, 2],
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            hovermode='closest',
            template='plotly_white',
            margin=dict(l=60, r=60, t=80, b=80, pad=4)
        )

        # Add zero line for sentiment score
        fig.add_shape(
            type='line',
            xref='paper',
            yref='y2',
            x0=0,
            y0=0,
            x1=1,
            y1=0,
            line=dict(color='gray', width=1, dash='dash')
        )

        return fig

    def _create_sector_trend_chart(self, trend_data, sector):
        """
        Create a chart showing sentiment trend for a specific sector

        Args:
            trend_data (pd.DataFrame): DataFrame with trend data
            sector (str): Sector name

        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Filter data for this sector
        sector_data = trend_data[trend_data['sector'] == sector]

        # Get unique dates
        unique_dates = sorted(sector_data['formatted_date'].unique())
        short_dates = [datetime.strptime(date, '%Y-%m-%d').strftime('%d%m') for date in unique_dates]

        # Create figure
        fig = go.Figure()

        # Add trace for each sentiment category
        for category in self.sentiment_categories:
            # Filter for this category
            category_data = sector_data[sector_data['sentiment_category'] == category]

            # Ensure data for all dates (fill missing with 0)
            percentages = []
            for date in unique_dates:
                row = category_data[category_data['formatted_date'] == date]
                percentages.append(row['percentage'].values[0] if len(row) > 0 else 0)

            # Determine color based on category
            if category == 'LONG':
                color = '#1E8449'  # Dark green
            elif category == 'ACCUMULATION':
                color = '#82E0AA'  # Light green
            elif category == 'NEUTRAL':
                color = '#F7DC6F'  # Yellow
            elif category == 'DISTRIBUTION':
                color = '#F5B041'  # Orange
            elif category == 'SHORT':
                color = '#C0392B'  # Red
            else:
                color = '#3498DB'  # Blue (fallback)

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=short_dates,
                    y=percentages,
                    mode='lines+markers',
                    name=f'{category}',
                    line=dict(color=color, width=2),
                    marker=dict(size=6)
                )
            )

        # Update layout
        fig.update_layout(
            title=f'{sector}',
            xaxis=dict(
                title='Date',
                tickmode='array',
                tickvals=short_dates,
                ticktext=short_dates
            ),
            yaxis=dict(
                title='Percentage (%)',
                range=[0, 100]
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            hovermode='closest',
            template='plotly_white',
            margin=dict(l=40, r=40, t=60, b=80, pad=4)
        )

        return fig

    




def main():
    """
    Main function to run the sector sentiment trend generator
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate sector sentiment trend analysis')
    parser.add_argument('--reports-dir', type=str, default=None,
                        help='Base directory containing folders with stock reports')
    parser.add_argument('--dashboard-dir', type=str, default=None,
                        help='Directory containing market_dashboard_YYYYMMDD.html files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for output files')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date in YYYYMMDD format')
    parser.add_argument('--date-format', type=str, default="%Y%m%d",
                        help='Format of date strings (default: YYYYMMDD)')
    parser.add_argument('--generate-allinone', action='store_true',
                        help='Generate a single page with all sector charts')
    
    args = parser.parse_args()

    # Validate at least one source directory is provided
    if not args.reports_dir and not args.dashboard_dir:
        parser.error("At least one of --reports-dir or --dashboard-dir must be provided")

    # Create trend generator
    trend_generator = SectorSentimentTrendGenerator(
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        reports_base_dir=args.reports_dir,
        dashboard_base_dir=args.dashboard_dir,
        date_format=args.date_format,
        generate_allinone=args.generate_allinone
    )

    # Generate the trend analysis
    output_path = trend_generator.generate_trend_analysis()

    print(f"Sector sentiment trend analysis generated at: {output_path}")


if __name__ == "__main__":
    main()