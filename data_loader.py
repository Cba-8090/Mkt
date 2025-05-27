import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
import os
import time
from typing import Dict, Optional, Tuple

from database import HYGDatabase


class DataLoader:
    """Handles CSV loading and daily data updates using FRED CSV downloads only"""

    def __init__(self, db: HYGDatabase, fred_api_key: str = None):
        self.db = db
        # Note: fred_api_key parameter kept for compatibility but not used
        self.logger = logging.getLogger(__name__)

        # File paths for historical CSV data
        self.csv_files = {
            'hyg_spread': 'data/BAMLH0A0HYM2.csv',
            'hy_yield': 'data/BAMLH0A2HYBEY.csv',
            'treasury_10y': 'data/DGS10.csv'
        }

        # FRED CSV download URLs for daily updates
        self.fred_csv_urls = {
            'hyg_spread': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2',
            'hy_yield': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A2HYBEY',
            'treasury_10y': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10'
        }

        # Data directory
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)

        self.logger.info("DataLoader initialized - using FRED CSV downloads only")

    def load_csv_data(self) -> bool:
        """Load historical data from CSV files"""
        try:
            self.logger.info("Loading historical data from CSV files...")

            # Load each CSV file
            datasets = {}
            for data_type, file_path in self.csv_files.items():
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)

                    # Standardize column names (first = date, second = value)
                    if len(df.columns) >= 2:
                        df.columns = ['date', 'value'] + list(df.columns[2:])
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                        # Handle missing values (represented as '.' in FRED data)
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df = df.dropna(subset=['value'])  # Remove missing values

                        datasets[data_type] = df.set_index('date')['value'].to_dict()
                        self.logger.info(f"Loaded {len(df)} records from {file_path}")
                    else:
                        self.logger.warning(f"Invalid CSV format in {file_path}")
                        datasets[data_type] = {}
                else:
                    self.logger.warning(f"CSV file not found: {file_path}")
                    datasets[data_type] = {}

            # Combine datasets and insert into database
            success = self._merge_and_insert_data(datasets, data_source='CSV_HISTORICAL')

            if success:
                summary = self.db.get_data_summary()
                self.logger.info(f"CSV loading complete. Total records: {summary.get('total_records', 0)}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to load CSV data: {e}")
            return False

    def update_daily_data(self) -> bool:
        """Fetch and update latest data using FRED CSV downloads"""
        try:
            self.logger.info("Updating daily market data using FRED CSV downloads...")

            # Download latest CSV data for each series
            downloaded_data = {}

            for series_name, url in self.fred_csv_urls.items():
                self.logger.info(f"Downloading {series_name}...")

                success, data = self._download_and_process_csv(series_name, url)
                if success and data:
                    downloaded_data[series_name] = data
                    self.logger.info(f"Successfully downloaded {series_name}: {len(data)} records")
                else:
                    self.logger.warning(f"Failed to download {series_name}")

            if not downloaded_data:
                self.logger.error("No data downloaded successfully")
                return False

            # Insert new data into database
            return self._process_daily_updates(downloaded_data)

        except Exception as e:
            self.logger.error(f"Daily data update failed: {e}")
            return False

    def _download_and_process_csv(self, series_name: str, url: str, max_retries: int = 3) -> Tuple[bool, Dict]:
        """Download CSV from FRED and process it"""
        try:
            for attempt in range(max_retries):
                try:
                    # Add date parameters to get last 30 days
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

                    params = {
                        'cosd': start_date,
                        'coed': end_date
                    }

                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()

                    # Parse CSV content
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))

                    # Validate and clean data
                    if len(df.columns) < 2 or len(df) == 0:
                        raise ValueError(f"Invalid CSV format for {series_name}")

                    # Standardize columns
                    df.columns = ['date', 'value'] + list(df.columns[2:]) if len(df.columns) > 2 else ['date', 'value']

                    # Clean data
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna(subset=['value'])

                    if len(df) == 0:
                        raise ValueError(f"No valid data after cleaning for {series_name}")

                    # Convert to dictionary
                    data_dict = df.set_index('date')['value'].to_dict()

                    # Validate data ranges
                    if self._validate_data_ranges(series_name, data_dict):
                        return True, data_dict
                    else:
                        self.logger.warning(f"Data validation failed for {series_name}")
                        return False, {}

                except Exception as e:
                    self.logger.warning(f"Download attempt {attempt + 1} failed for {series_name}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry

            return False, {}

        except Exception as e:
            self.logger.error(f"Failed to download {series_name}: {e}")
            return False, {}

    def _validate_data_ranges(self, series_name: str, data_dict: Dict) -> bool:
        """Validate that data values are in reasonable ranges"""
        try:
            if not data_dict:
                return False

            values = list(data_dict.values())
            min_val = min(values)
            max_val = max(values)

            # Define reasonable ranges for each series
            ranges = {
                'hyg_spread': (0.5, 25.0),  # 0.5% to 25%
                'hy_yield': (2.0, 20.0),  # 2% to 20%
                'treasury_10y': (0.0, 15.0)  # 0% to 15%
            }

            if series_name in ranges:
                min_expected, max_expected = ranges[series_name]

                if min_val < min_expected or max_val > max_expected:
                    self.logger.warning(f"Values out of range for {series_name}: {min_val:.2f}% to {max_val:.2f}%")
                    # Still return True but log warning - data might be valid during crisis

            return True

        except Exception as e:
            self.logger.error(f"Data validation error for {series_name}: {e}")
            return False

    def _process_daily_updates(self, downloaded_data: Dict) -> bool:
        """Process downloaded data and update database"""
        try:
            # Get latest database date to avoid duplicates
            latest_db_date = self._get_latest_database_date()

            # Combine all data by date
            combined_data = {}

            for series_name, data_dict in downloaded_data.items():
                for date_str, value in data_dict.items():
                    # Only process new data
                    if not latest_db_date or date_str > latest_db_date:
                        if date_str not in combined_data:
                            combined_data[date_str] = {}
                        combined_data[date_str][series_name] = value

            if not combined_data:
                self.logger.info("No new data to insert")
                return True

            # Insert data into database
            inserted_count = 0

            for date_str in sorted(combined_data.keys()):
                values = combined_data[date_str]

                success = self.db.insert_market_data(
                    date=date_str,
                    hyg_spread=values.get('hyg_spread'),
                    hy_yield=values.get('hy_yield'),
                    treasury_10y=values.get('treasury_10y'),
                    data_source='CSV_DAILY_UPDATE'
                )

                if success:
                    inserted_count += 1
                    self.logger.debug(f"Inserted data for {date_str}")

            self.logger.info(f"Successfully inserted {inserted_count} new records")
            return inserted_count > 0

        except Exception as e:
            self.logger.error(f"Failed to process daily updates: {e}")
            return False

    def _get_latest_database_date(self) -> Optional[str]:
        """Get the latest date in the database"""
        try:
            latest_data = self.db.get_latest_data()
            return latest_data['date'] if latest_data else None
        except Exception:
            return None

    def _merge_and_insert_data(self, datasets: Dict[str, Dict], data_source: str = 'CSV') -> bool:
        """Merge datasets by date and insert into database"""
        try:
            # Get all unique dates
            all_dates = set()
            for dataset in datasets.values():
                all_dates.update(dataset.keys())

            if not all_dates:
                self.logger.warning("No dates found in datasets")
                return False

            # Insert data for each date
            inserted_count = 0
            total_dates = len(all_dates)

            for i, date in enumerate(sorted(all_dates)):
                if i % 1000 == 0:  # Progress logging
                    self.logger.info(f"Processing dates: {i}/{total_dates}")

                hyg_spread = datasets.get('hyg_spread', {}).get(date)
                hy_yield = datasets.get('hy_yield', {}).get(date)
                treasury_10y = datasets.get('treasury_10y', {}).get(date)

                # Only insert if we have at least one value
                if any([hyg_spread, hy_yield, treasury_10y]):
                    if self.db.insert_market_data(
                            date=date,
                            hyg_spread=hyg_spread,
                            hy_yield=hy_yield,
                            treasury_10y=treasury_10y,
                            data_source=data_source
                    ):
                        inserted_count += 1

            self.logger.info(f"Inserted {inserted_count} records into database")
            return inserted_count > 0

        except Exception as e:
            self.logger.error(f"Failed to merge and insert data: {e}")
            return False

    def validate_data_integrity(self) -> Dict[str, any]:
        """Validate data integrity and consistency"""
        try:
            validation_report = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'data_quality_score': 1.0
            }

            # Get recent data for validation
            recent_data = self.db.get_historical_data(30)

            if not recent_data:
                validation_report['is_valid'] = False
                validation_report['issues'].append("No recent data available")
                return validation_report

            # Check for missing values
            missing_spread_count = sum(1 for d in recent_data if d['hyg_spread'] is None)
            missing_hy_yield_count = sum(1 for d in recent_data if d['hy_yield'] is None)
            missing_treasury_count = sum(1 for d in recent_data if d['treasury_10y'] is None)

            total_records = len(recent_data)

            if missing_spread_count > total_records * 0.2:  # More than 20% missing
                validation_report['issues'].append(
                    f"High missing data rate for HYG spread: {missing_spread_count}/{total_records}")
                validation_report['data_quality_score'] *= 0.7

            # Check for data consistency (calculated spread vs reported spread)
            inconsistency_count = 0
            for data in recent_data:
                if all([data['hyg_spread'], data['hy_yield'], data['treasury_10y']]):
                    calculated = data['hy_yield'] - data['treasury_10y']
                    reported = data['hyg_spread']

                    if abs(calculated - reported) > 0.5:  # More than 50bps difference
                        inconsistency_count += 1

            if inconsistency_count > 0:
                validation_report['warnings'].append(
                    f"Spread calculation inconsistencies: {inconsistency_count} records")
                validation_report['data_quality_score'] *= 0.9

            # Check for extreme outliers
            spreads = [d['hyg_spread'] for d in recent_data if d['hyg_spread'] is not None]
            if spreads:
                min_spread = min(spreads)
                max_spread = max(spreads)

                if min_spread < 0 or max_spread > 30:
                    validation_report['warnings'].append(
                        f"Extreme spread values detected: {min_spread:.2f}% to {max_spread:.2f}%")
                    validation_report['data_quality_score'] *= 0.8

            # Data freshness check
            latest_data = self.db.get_latest_data()
            if latest_data:
                latest_date = datetime.strptime(latest_data['date'], '%Y-%m-%d').date()
                days_old = (datetime.now().date() - latest_date).days

                if days_old > 5:
                    validation_report['warnings'].append(f"Data is {days_old} days old")
                    validation_report['data_quality_score'] *= 0.9

            return validation_report

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {e}"],
                'warnings': [],
                'data_quality_score': 0.0
            }

    def get_data_status(self) -> Dict[str, any]:
        """Get comprehensive data status report"""
        try:
            summary = self.db.get_data_summary()
            latest_data = self.db.get_latest_data()
            validation = self.validate_data_integrity()

            status = {
                'database_summary': summary,
                'latest_data': latest_data,
                'validation_report': validation,
                'data_sources_available': {
                    'csv_files': {name: os.path.exists(path) for name, path in self.csv_files.items()},
                    'fred_csv_urls': True,  # Always available
                    'fred_api': False,  # Not used
                    'yahoo_finance': False  # Not used
                },
                'last_updated': datetime.now().isoformat()
            }

            return status

        except Exception as e:
            self.logger.error(f"Failed to get data status: {e}")
            return {'error': str(e)}

    def backfill_missing_data(self, start_date: str = None, end_date: str = None) -> bool:
        """Backfill missing data using FRED CSV downloads"""
        try:
            self.logger.info("Starting backfill process using FRED CSV downloads...")

            if not start_date:
                # Default to last 90 days
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Get existing data dates
            existing_data = self.db.get_historical_data(365)  # Get more data for backfill check
            existing_dates = {d['date'] for d in existing_data}

            self.logger.info(f"Backfilling data from {start_date} to {end_date}")
            self.logger.info(f"Found {len(existing_dates)} existing dates in database")

            # Download data for the backfill period
            downloaded_data = {}

            for series_name, url in self.fred_csv_urls.items():
                self.logger.info(f"Downloading {series_name} for backfill...")

                try:
                    params = {
                        'cosd': start_date,
                        'coed': end_date
                    }

                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()

                    # Parse CSV
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))

                    if len(df.columns) >= 2 and len(df) > 0:
                        df.columns = ['date', 'value'] + list(df.columns[2:]) if len(df.columns) > 2 else ['date',
                                                                                                           'value']
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df = df.dropna(subset=['value'])

                        # Only keep dates not in existing data
                        new_dates_df = df[~df['date'].isin(existing_dates)]

                        if len(new_dates_df) > 0:
                            downloaded_data[series_name] = new_dates_df.set_index('date')['value'].to_dict()
                            self.logger.info(f"Found {len(new_dates_df)} new dates for {series_name}")
                        else:
                            self.logger.info(f"No new dates found for {series_name}")

                except Exception as e:
                    self.logger.warning(f"Failed to download {series_name} for backfill: {e}")

            if not downloaded_data:
                self.logger.info("No new data found for backfill")
                return True

            # Process backfill data
            return self._process_daily_updates(downloaded_data)

        except Exception as e:
            self.logger.error(f"Backfill process failed: {e}")
            return False