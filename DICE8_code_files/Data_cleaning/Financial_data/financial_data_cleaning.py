def clean_stock_data(input_csv, output_csv):
    try:
        # Read the CSV, assuming Date is the index
        df = pd.read_csv(input_csv, index_col='Date', parse_dates=True)

        # Apply backward fill to handle missing data at start/end
        df = df.bfill()

        # Apply linear interpolation to fill small gaps
        df = df.interpolate(method='linear', limit_direction='both')

        # Save the cleaned data to a new CSV
        df.to_csv(output_csv)
        print(f"Cleaned data saved to {output_csv}")

        # Report any remaining NaN values
        if df.isna().any().any():
            print("Warning: Some NaN values remain after cleaning.")
            print(df.isna().sum())

    except FileNotFoundError:
        print(f"Error: Input file {input_csv} not found.")
    except Exception as e:
        print(f"Error processing data: {e}")

# Example usage
if __name__ == "__main__":
    clean_stock_data("stock_closing_prices.csv", "clean_data.csv")

"""2. Normalize"""

def process_stock_data(input_csv, normalized_output_csv, log_output_csv):
    try:
        # Read the CSV, assuming Date is the index
        df = pd.read_csv(input_csv, index_col='Date', parse_dates=True)

        # Check if DataFrame is empty
        if df.empty:
            print("Error: Input CSV is empty.")
            return

        # Initialize DataFrames for normalized and log evolution data
        normalized_df = df.copy()
        log_evolution_df = df.copy()

        # Process each ticker (column)
        for ticker in df.columns:
            # Get the first non-null value for the ticker
            first_valid = df[ticker].dropna().iloc[0] if df[ticker].dropna().size > 0 else None

            if first_valid is not None and first_valid > 0:
                # Normalize: divide by the first non-null value (set first point to 1)
                normalized_df[ticker] = df[ticker] / first_valid

                # Log evolution: ln(price / first_valid)
                log_evolution_df[ticker] = np.log(df[ticker] / first_valid)
            else:
                # If no valid first value or first value is zero, set to NaN
                print(f"Warning: No valid first value for {ticker} or first value is zero.")
                normalized_df[ticker] = np.nan
                log_evolution_df[ticker] = np.nan

        # Save the normalized data
        normalized_df.to_csv(normalized_output_csv)
        print(f"Normalized data saved to {normalized_output_csv}")

        # Save the log evolution data
        log_evolution_df.to_csv(log_output_csv)
        print(f"Log evolution data saved to {log_output_csv}")

        # Report any columns with all NaN values
        if normalized_df.isna().all().any():
            print("Warning: Some tickers have all NaN in normalized data:",
                  normalized_df.columns[normalized_df.isna().all()].tolist())
        if log_evolution_df.isna().all().any():
            print("Warning: Some tickers have all NaN in log evolution data:",
                  log_evolution_df.columns[log_evolution_df.isna().all()].tolist())

    except FileNotFoundError:
        print(f"Error: Input file {input_csv} not found.")
    except Exception as e:
        print(f"Error processing data: {e}")

# Example usage
if __name__ == "__main__":
    process_stock_data("clean_data.csv", "normalized_data.csv", "log_evolution_data.csv")

def simulate_equal_weights_portfolio(input_csv, output_csv, initial_value=1):
    try:
        # Read the CSV, assuming Date is the index
        df = pd.read_csv(input_csv, index_col='Date', parse_dates=True)

        # Check if DataFrame is empty
        if df.empty:
            print("Error: Input CSV is empty.")
            return

        # Drop columns with all NaN values (e.g., failed tickers)
        df = df.dropna(axis=1, how='all')
        if df.empty:
            print("Error: No valid ticker data after dropping NaN columns.")
            return

        # Number of tickers
        n_tickers = len(df.columns)
        if n_tickers == 0:
            print("Error: No valid tickers found.")
            return

        # Initial allocation: equal weight for each ticker
        weight = 1.0 / n_tickers
        initial_allocation = initial_value * weight

        # Calculate number of shares for each ticker (based on first valid price)
        first_valid_prices = df.apply(lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else np.nan)
        if first_valid_prices.isna().any():
            print("Warning: Some tickers have no valid prices:",
                  first_valid_prices.index[first_valid_prices.isna()].tolist())
            df = df.loc[:, ~first_valid_prices.isna()]
            first_valid_prices = first_valid_prices.dropna()
            n_tickers = len(df.columns)
            weight = 1.0 / n_tickers
            initial_allocation = initial_value * weight

        shares = initial_allocation / first_valid_prices

        # Calculate portfolio value: sum of (shares * price) for each ticker
        portfolio_values = (df * shares).sum(axis=1)

        # Calculate log evolution: ln(portfolio_value / initial_value)
        log_evolution = np.log(portfolio_values / initial_value)

        # Create output DataFrame
        output_df = pd.DataFrame({
            'Portfolio_Value': portfolio_values,
            'Log_Evolution': log_evolution
        })

        # Save to CSV
        output_df.to_csv(output_csv)
        print(f"Portfolio evolution saved to {output_csv}")

        # Report any NaN values in the output
        if output_df.isna().any().any():
            print("Warning: Some NaN values in portfolio evolution.")
            print(output_df.isna().sum())

    except FileNotFoundError:
        print(f"Error: Input file {input_csv} not found.")
    except Exception as e:
        print(f"Error processing data: {e}")

# Example usage
if __name__ == "__main__":
    simulate_equal_weights_portfolio("clean_data.csv", "portfolio_evolution.csv")

"""3. Separate by month"""

import pandas as pd

def split_normalized_by_month(input_csv, output_dir="."):
    try:
        # Read the CSV, assuming Date is the index
        df = pd.read_csv(input_csv, index_col='Date', parse_dates=True)

        # Check if DataFrame is empty
        if df.empty:
            print("Error: Input CSV is empty.")
            return

        # Group data by year and month
        grouped = df.groupby([df.index.year, df.index.month])

        # Iterate through each year-month group
        for (year, month), group_data in grouped:
            # Format the output filename (e.g., 2003_01_closing.csv)
            output_file = f"{output_dir}/{year}_{month:02d}_closing.csv"

            # Save the group data to a CSV
            group_data.to_csv(output_file)
            print(f"Saved {output_file} with {len(group_data)} rows")

            # Warn if the group has NaN values
            if group_data.isna().any().any():
                print(f"Warning: {output_file} contains NaN values.")
                nan_columns = group_data.columns[group_data.isna().any()].tolist()
                print(f"Columns with NaN: {nan_columns}")

        print("All monthly CSVs generated successfully.")

    except FileNotFoundError:
        print(f"Error: Input file {input_csv} not found.")
    except Exception as e:
        print(f"Error processing data: {e}")

# Example usage
if __name__ == "__main__":
    split_normalized_by_month("normalized_data.csv")

"""4. Compute metrics for each"""

import pandas as pd
import numpy as np
import os
from glob import glob

def calculate_monthly_metrics(input_dir=".", output_dir="."):
    """
    Generate monthly observation and correlation CSVs from normalized closing data.

    For each {yyyy}_{mm}_closing.csv, creates:
    - {yyyy}_{mm}_obs.csv: First close, last close, volatility, Sharpe, Sortino, Calmar, MDD
    - {yyyy}_{mm}_corr.csv: Correlation matrix of daily log returns

    Parameters:
    input_dir (str): Directory containing input CSVs (default: current directory)
    output_dir (str): Directory to save output CSVs (default: current directory)

    Returns:
    None: Saves observation and correlation CSVs to output_dir
    """
    try:
        # Find all monthly closing CSVs
        closing_files = glob(f"{input_dir}/*_closing.csv")
        if not closing_files:
            print(f"Error: No *_closing.csv files found in {input_dir}.")
            return

        for file in closing_files:
            # Extract year and month from filename (e.g., 2003_01_closing.csv)
            filename = os.path.basename(file)
            year_month = filename.replace("_closing.csv", "")
            year, month = map(int, year_month.split("_"))

            # Read the monthly CSV
            df = pd.read_csv(file, index_col='Date', parse_dates=True)

            if df.empty:
                print(f"Warning: {filename} is empty. Skipping.")
                continue

            # Drop columns with all NaN values
            df = df.dropna(axis=1, how='all')
            if df.empty:
                print(f"Warning: {filename} has no valid data after dropping NaN columns. Skipping.")
                continue

            # Calculate daily log returns: ln(price_t / price_{t-1})
            log_returns = np.log(df / df.shift(1)).dropna()

            # Initialize metrics DataFrame
            metrics = pd.DataFrame(index=df.columns)

            # Month First Close: First non-null normalized price
            metrics['Month_First_Close'] = df.apply(lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else np.nan)

            # Month Last Close: Last non-null normalized price
            metrics['Month_Last_Close'] = df.apply(lambda x: x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan)

            # Volatility: Annualized standard deviation of daily log returns
            metrics['Volatility'] = log_returns.std() * np.sqrt(252)

            # Sharpe Ratio: Annualized mean log return / volatility (risk-free rate = 0)
            metrics['Sharpe_Ratio'] = (log_returns.mean() * 252) / metrics['Volatility']

            # Sortino Ratio: Annualized mean log return / downside volatility
            downside_returns = log_returns.where(log_returns < 0, 0)
            downside_vol = downside_returns.std() * np.sqrt(252)
            metrics['Sortino_Ratio'] = (log_returns.mean() * 252) / downside_vol

            # Maximum Drawdown (MDD): Max peak-to-trough decline
            def calculate_mdd(series):
                cumulative = series / series.iloc[0]  # Normalize to start at 1
                peak = cumulative.cummax()
                drawdown = (peak - cumulative) / peak
                return drawdown.max() if drawdown.size > 0 else np.nan

            metrics['MDD'] = df.apply(calculate_mdd)

            # Calmar Ratio: Annualized return / MDD
            monthly_return = (metrics['Month_Last_Close'] / metrics['Month_First_Close']) - 1
            annualized_return = (1 + monthly_return) ** (12 / 1) - 1  # Annualize monthly return
            metrics['Calmar_Ratio'] = annualized_return / metrics['MDD']

            # Replace infinities with NaN (e.g., zero volatility or MDD)
            metrics = metrics.replace([np.inf, -np.inf], np.nan)

            # Save observation CSV
            obs_file = f"{output_dir}/{year_month}_obs.csv"
            metrics.to_csv(obs_file)
            print(f"Saved {obs_file} with metrics for {len(metrics)} tickers")

            # Warn about NaN values in metrics
            if metrics.isna().any().any():
                print(f"Warning: {obs_file} contains NaN values.")
                nan_columns = metrics.columns[metrics.isna().any()].tolist()
                print(f"Metrics with NaN: {nan_columns}")

            # Correlation matrix of daily log returns
            if not log_returns.empty:
                corr_matrix = log_returns.corr()
                corr_file = f"{output_dir}/{year_month}_corr.csv"
                corr_matrix.to_csv(corr_file)
                print(f"Saved {corr_file} with correlation matrix")

                # Warn about NaN values in correlation matrix
                if corr_matrix.isna().any().any():
                    print(f"Warning: {corr_file} contains NaN values.")
                    nan_columns = corr_matrix.columns[corr_matrix.isna().any()].tolist()
                    print(f"Columns with NaN: {nan_columns}")
            else:
                print(f"Warning: No valid log returns for {year_month}. Skipping correlation matrix.")

        print("All observation and correlation CSVs generated successfully.")

    except Exception as e:
        print(f"Error processing files: {e}")

# Example usage
if __name__ == "__main__":
    calculate_monthly_metrics()

import pandas as pd
import os
from glob import glob

def replace_nan_with_zeros(input_dir="."):
    try:
        # List of specific files to process
        specific_files = [
            "normalized_data.csv",
            "log_evolution_data.csv",
            "portfolio_evolution.csv"
        ]

        # Add monthly files using glob
        monthly_patterns = [
            "*_closing.csv",
            "*_obs.csv",
            "*_corr.csv"
        ]

        # Collect all files to process
        all_files = []
        for file in specific_files:
            file_path = os.path.join(input_dir, file)
            if os.path.exists(file_path):
                all_files.append(file_path)

        for pattern in monthly_patterns:
            all_files.extend(glob(os.path.join(input_dir, pattern)))

        if not all_files:
            print(f"Error: No matching CSV files found in {input_dir}.")
            return

        # Process each file
        for file in all_files:
            try:
                # Determine index column based on file type
                filename = os.path.basename(file)
                if filename.endswith(("_obs.csv", "_corr.csv")):
                    # _obs.csv and _corr.csv use first column as index (tickers)
                    df = pd.read_csv(file, index_col=0)
                else:
                    # Other CSVs use Date as index
                    df = pd.read_csv(file, index_col='Date', parse_dates=True)

                # Replace NaN with 0
                df = df.fillna(0)

                # Save back to the original file
                df.to_csv(file)
                print(f"Processed {filename}: Replaced NaN with 0s")

                # Verify no NaNs remain
                if df.isna().any().any():
                    print(f"Warning: {filename} still contains NaN values after processing.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        print("All specified CSVs processed successfully.")

    except Exception as e:
        print(f"Error accessing files: {e}")

# Example usage
if __name__ == "__main__":
    replace_nan_with_zeros()

"""Create obs vectors"""

import pandas as pd
import numpy as np
import os
from glob import glob

def combine_monthly_metrics(input_dir=".", output_dir="."):
    """
    Combine monthly observation and correlation data into a single-column CSV.

    For each {yyyy}_{mm}_obs.csv and {yyyy}_{mm}_corr.csv, creates:
    - {yyyy}_{mm}_combined.csv: Single column with metrics and flattened correlations

    Parameters:
    input_dir (str): Directory containing input CSVs (default: current directory)
    output_dir (str): Directory to save output CSVs (default: current directory)

    Returns:
    None: Saves combined CSVs to output_dir
    """
    try:
        # Find all observation CSVs
        obs_files = glob(f"{input_dir}/*_obs.csv")
        if not obs_files:
            print(f"Error: No *_obs.csv files found in {input_dir}.")
            return

        for obs_file in obs_files:
            # Extract year and month from filename (e.g., 2003_01_obs.csv)
            filename = os.path.basename(obs_file)
            year_month = filename.replace("_obs.csv", "")

            # Corresponding correlation file
            corr_file = f"{input_dir}/{year_month}_corr.csv"

            # Check if correlation file exists
            if not os.path.exists(corr_file):
                print(f"Warning: {corr_file} not found. Skipping {year_month}.")
                continue

            try:
                # Read observation CSV (tickers as index, metrics as columns)
                obs_df = pd.read_csv(obs_file, index_col=0)

                # Read correlation CSV (tickers as both index and columns)
                corr_df = pd.read_csv(corr_file, index_col=0)

                # Validate DataFrames
                if obs_df.empty:
                    print(f"Warning: {filename} is empty. Skipping {year_month}.")
                    continue
                if corr_df.empty:
                    print(f"Warning: {corr_file} is empty. Skipping {year_month}.")
                    continue

                # Initialize list for combined data
                combined_data = []

                # Process observation metrics
                for ticker in obs_df.index:
                    for metric in obs_df.columns:
                        value = obs_df.at[ticker, metric]
                        row_label = f"{ticker}_{metric}"
                        combined_data.append((row_label, value))

                # Flatten correlation matrix (upper triangle, exclude diagonal)
                tickers = corr_df.index
                for i in range(len(tickers)):
                    for j in range(i + 1, len(tickers)):  # Upper triangle
                        ticker1, ticker2 = tickers[i], tickers[j]
                        value = corr_df.at[ticker1, ticker2]
                        row_label = f"{ticker1}_{ticker2}_Correlation"
                        combined_data.append((row_label, value))

                # Create single-column DataFrame
                combined_df = pd.DataFrame(
                    [x[1] for x in combined_data],
                    index=[x[0] for x in combined_data],
                    columns=['Value']
                )

                # Save to combined CSV
                output_file = f"{output_dir}/{year_month}_combined.csv"
                combined_df.to_csv(output_file)
                print(f"Saved {output_file} with {len(combined_df)} rows")

                # Warn about any unexpected values (e.g., NaNs, though replaced with 0s)
                if combined_df['Value'].eq(0).any():
                    print(f"Note: {output_file} contains zero values (from prior NaN replacement).")

            except Exception as e:
                print(f"Error processing {year_month}: {e}")

        print("All combined CSVs generated successfully.")

    except Exception as e:
        print(f"Error accessing files: {e}")

# Example usage
if __name__ == "__main__":
    combine_monthly_metrics()

import os
import shutil
from glob import glob

def organize_csv_files(input_dir=".", output_dir="."):
    """
    Organize CSV files into three folders: combined, price, and usage.

    Folders:
    - combined: {yyyy}_{mm}_combined.csv
    - price: clean_data.csv, normalized_data.csv, log_evolution_data.csv, portfolio_evolution.csv
    - usage: {yyyy}_{mm}_closing.csv, {yyyy}_{mm}_obs.csv, {yyyy}_{mm}_corr.csv

    Parameters:
    input_dir (str): Directory containing the CSV files (default: current directory)
    output_dir (str): Directory to create folders and move files (default: current directory)

    Returns:
    None: Moves CSV files to appropriate folders
    """
    try:
        # Define folder names and their corresponding files
        folders = {
            "combined": glob(os.path.join(input_dir, "*_combined.csv")),
            "price": [
                os.path.join(input_dir, f) for f in [
                    "clean_data.csv",
                    "normalized_data.csv",
                    "log_evolution_data.csv",
                    "portfolio_evolution.csv",
                    "stock_closing_prices.csv"
                ] if os.path.exists(os.path.join(input_dir, f))
            ],
            "usage": (
                glob(os.path.join(input_dir, "*_closing.csv")) +
                glob(os.path.join(input_dir, "*_obs.csv")) +
                glob(os.path.join(input_dir, "*_corr.csv"))
            )
        }

        # Check if any files were found
        total_files = sum(len(files) for files in folders.values())
        if total_files == 0:
            print(f"Error: No matching CSV files found in {input_dir}.")
            return

        # Create folders if they don't exist
        for folder in folders:
            folder_path = os.path.join(output_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created/Verified folder: {folder_path}")

        # Move files to their respective folders
        for folder, files in folders.items():
            target_dir = os.path.join(output_dir, folder)
            for file in files:
                filename = os.path.basename(file)
                target_path = os.path.join(target_dir, filename)

                # Skip if file already exists in target to avoid overwriting
                if os.path.exists(target_path):
                    print(f"Skipped {filename}: Already exists in {target_dir}")
                    continue

                try:
                    shutil.move(file, target_path)
                    print(f"Moved {filename} to {target_dir}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

        print("All CSV files organized successfully.")

    except Exception as e:
        print(f"Error organizing files: {e}")

# Example usage
if __name__ == "__main__":
    organize_csv_files()
