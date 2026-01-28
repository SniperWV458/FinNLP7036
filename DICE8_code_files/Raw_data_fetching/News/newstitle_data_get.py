"""
Financial Asset Data Monthly Download (2012-2024)
Download 14 assets from the table, maximum 20 records per month
"""

from gdeltdoc import GdeltDoc, Filters
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import random
import json


class FinancialAssetsDownloader:
    """Financial asset data downloader"""

    def __init__(self):
        self.gd = GdeltDoc()

        # Asset configuration - from the table
        self.assets_config = {
            "GSPC": {
                "name": "S&P 500 Index",
                "keywords": ["SP500", "SPX", "Standard Poors 500", "S and P 500"],
                "ticker": "$SPX",  # StockTwits format code
                "type": "E"  # Equity/Index
            },
            "IXIC": {
                "name": "NASDAQ Composite",
                "keywords": ["NASDAQ", "NASDAQ Composite", "IXIC"],
                "ticker": "$QQQ",  # Using QQQ as proxy
                "type": "E"
            },
            "DJI": {
                "name": "Dow Jones Industrial Average",
                "keywords": ["Dow Jones", "DJIA", "Dow 30"],
                "ticker": "$DIA",
                "type": "E"
            },
            "GOLD": {
                "name": "Gold",
                "keywords": ["gold price", "gold market", "gold bullion"],
                "type": "C"  # Commodity
            },
            "SILVER": {
                "name": "Silver",
                "keywords": ["silver price", "silver market", "silver bullion"],
                "type": "C"
            },
            "OIL": {
                "name": "WTI Crude Oil Futures",
                "keywords": ["crude oil", "oil price", "WTI", "oil futures"],
                "type": "C"
            }
        }

        # Download parameters
        self.start_year = 2012
        self.end_year = 2024
        self.max_records_per_month = 20  # Maximum 20 records per month

        # Output directory : Needs to be changed！！！！！！
        self.base_output_dir = "/Users/gjn/Desktop/financial_assets_data"

        # Delay settings
        self.min_delay = 2.0
        self.max_delay = 4.0

    def get_month_date_range(self, year, month):
        """Get start and end dates for a month"""
        start_date = f"{year}-{month:02d}-01"

        if month == 12:
            end_date = f"{year}-12-31"
        else:
            end_date = f"{year}-{month:02d}-{(datetime(year, month + 1, 1) - timedelta(days=1)).day:02d}"

        return start_date, end_date

    def download_asset_data(self, asset_id, asset_config):
        """Download data for a single asset"""
        asset_name = asset_config["name"]
        keywords = asset_config["keywords"]

        print(f"\n{'=' * 60}")
        print(f"Starting download: {asset_id} - {asset_name}")
        print(f"{'=' * 60}")

        # Create asset output directory
        asset_dir = os.path.join(self.base_output_dir, asset_id)
        os.makedirs(asset_dir, exist_ok=True)

        # Store all monthly data
        all_data = []

        for year in range(self.start_year, self.end_year + 1):
            print(f"\n{year}:")

            for month in range(1, 13):
                print(f"  {year}-{month:02d}...", end=" ")

                try:
                    # Get date range
                    start_date, end_date = self.get_month_date_range(year, month)

                    # Create filters
                    f = Filters(
                        start_date=start_date,
                        end_date=end_date,
                        num_records=self.max_records_per_month,
                        keyword=keywords,
                        language="English"
                    )

                    # Execute search
                    articles_df = self.gd.article_search(f)

                    if not articles_df.empty:
                        # Add asset information and time info
                        articles_df['asset_id'] = asset_id
                        articles_df['asset_name'] = asset_name
                        articles_df['asset_type'] = asset_config["type"]
                        articles_df['year'] = year
                        articles_df['month'] = month
                        articles_df['download_date'] = datetime.now().strftime("%Y-%m-%d")

                        # Add to total data
                        all_data.append(articles_df)

                        # Save monthly file
                        month_csv = os.path.join(asset_dir, f"{asset_id}_{year}_{month:02d}.csv")
                        articles_df.to_csv(month_csv, index=False, encoding="utf-8-sig")

                        print(f"✓ {len(articles_df)} records")

                    else:
                        print("⚠ 0 records")

                except Exception as e:
                    print(f"✗ Error: {str(e)[:50]}")

                # Random delay
                time.sleep(random.uniform(self.min_delay, self.max_delay))

        # Save merged data for this asset
        if all_data:
            # Merge all monthly data
            asset_df = pd.concat(all_data, ignore_index=True)

            # Sort by time
            if 'date' in asset_df.columns:
                asset_df = asset_df.sort_values('date')
            elif 'datetime' in asset_df.columns:
                asset_df = asset_df.sort_values('datetime')

            # Save merged CSV
            asset_csv = os.path.join(asset_dir, f"{asset_id}_2017-2024_all.csv")
            asset_df.to_csv(asset_csv, index=False, encoding="utf-8-sig")

            # Statistics
            total_records = len(asset_df)
            print(f"\n{'=' * 40}")
            print(f"{asset_name} download completed:")
            print(f"  Total records: {total_records}")
            print(f"  Time range: {self.start_year}-2024")
            print(f"  File location: {asset_csv}")
            print(f"{'=' * 40}")

            return {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "type": asset_config["type"],
                "total_records": total_records,
                "file_path": asset_csv
            }
        else:
            print(f"\n{asset_name} no data retrieved")
            return {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "type": asset_config["type"],
                "total_records": 0,
                "file_path": None
            }

    def create_summary_report(self, download_stats):
        """Create download summary report"""
        summary_data = []

        for stats in download_stats:
            summary_data.append({
                "Asset_ID": stats["asset_id"],
                "Asset_Name": stats["asset_name"],
                "Type": stats["type"],
                "Total_Records": stats["total_records"],
                "Status": "Success" if stats["total_records"] > 0 else "No Data",
                "File_Path": stats["file_path"] or "N/A"
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        summary_csv = os.path.join(self.base_output_dir, "download_summary.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        # Display summary
        print(f"\n{'=' * 80}")
        print(" Download Summary Report")
        print(f"{'=' * 80}")
        print(f"{'Asset_ID':<12} | {'Asset_Name':<30} | {'Type':<6} | {'Records':<10} | {'Status':<10}")
        print(f"{'-' * 80}")

        total_records = 0
        for _, row in summary_df.iterrows():
            print(f"{row['Asset_ID']:<12} | {row['Asset_Name'][:28]:<30} | "
                  f"{row['Type']:<6} | {row['Total_Records']:<10} | {row['Status']:<10}")
            total_records += row['Total_Records']

        print(f"{'-' * 80}")
        print(f"{'Total':<12} | {'':<30} | {'':<6} | {total_records:<10} |")
        print(f"{'=' * 80}")

        # Save configuration information
        config_info = {
            "download_parameters": {
                "start_year": self.start_year,
                "end_year": self.end_year,
                "max_records_per_month": self.max_records_per_month,
                "total_assets": len(self.assets_config)
            },
            "assets": self.assets_config,
            "summary": summary_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        config_json = os.path.join(self.base_output_dir, "download_config.json")
        with open(config_json, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)

        return summary_df

    def run(self):
        """Run download program"""

        # Create output directory
        os.makedirs(self.base_output_dir, exist_ok=True)

        print(f"{'=' * 80}")
        print(" Financial Asset Data Download System")
        print(f"{'=' * 80}")
        print(f"Number of assets: {len(self.assets_config)}")
        print(f"Time range: {self.start_year} Jan - {self.end_year} Dec")
        print(f"Maximum per month: {self.max_records_per_month} records")
        print(f"Output directory: {self.base_output_dir}")
        print(f"{'=' * 80}")

        # Download statistics
        download_stats = []

        # Download each asset
        for i, (asset_id, asset_config) in enumerate(self.assets_config.items(), 1):
            print(f"\n\n[Progress {i}/{len(self.assets_config)}]")

            stats = self.download_asset_data(asset_id, asset_config)
            download_stats.append(stats)

            # Delay between assets
            if i < len(self.assets_config):
                delay_time = random.uniform(8, 15)
                print(f"\n⏳ Waiting {delay_time:.1f} seconds before downloading next asset...")
                time.sleep(delay_time)

        # Create summary report
        self.create_summary_report(download_stats)

        print(f"\n{'=' * 80}")
        print("🎉 Download task completed!")
        print(f"{'=' * 80}")
        print(f"All files saved to: {self.base_output_dir}")
        print(f"Each asset includes: Monthly CSV files + Merged CSV file")
        print(f"{'=' * 80}")

        # Automatically open folder
        try:
            import subprocess
            subprocess.run(["open", self.base_output_dir])
            print("✅ Output directory opened automatically")
        except:
            print("💡 Tip: Please manually open the folder to view files")


# Main program
if __name__ == "__main__":
    # Use the full version
    downloader = FinancialAssetsDownloader()
    downloader.run()

    # Or use the simplified version
    # simple_download()
