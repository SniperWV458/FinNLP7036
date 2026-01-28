import os
import pandas as pd

INPUT_CSV = os.path.join("sentiment_analysis_cleaned", "combined_unified_matrix.csv")
OUTPUT_CSV = os.path.join("sentiment_analysis_cleaned", "combined_unified_matrix_padded.csv")
OUTPUT_METRICS_DIR = os.path.join("observation", "metrics")
START_YEAR = 2012
END_YEAR = 2024
NEUTRAL_VALUE = 0.0


def pad_neutral(
        input_path: str,
        output_path: str,
        start_year: int,
        end_year: int,
        neutral_value: float,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if "Year" not in df.columns or "Month" not in df.columns:
        raise ValueError("Expected 'Year' and 'Month' columns in the input CSV.")

    target_mask = (df["Year"] >= start_year) & (df["Year"] <= end_year)
    fill_columns = [col for col in df.columns if col not in ("Year", "Month")]
    df.loc[target_mask, fill_columns] = df.loc[target_mask, fill_columns].fillna(neutral_value)

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved neutral-padded CSV to {output_path}")
    return df


def write_metrics_files(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data_cols = [col for col in df.columns if col not in ("Year", "Month")]
    for _, row in df.iterrows():
        year = int(row["Year"])
        month = int(row["Month"])
        output_file = os.path.join(output_dir, f"{year}_{month:02d}_combined.csv")
        pd.DataFrame(row[data_cols].tolist()).to_csv(
            output_file,
            index=False,
            header=False,
            encoding="utf-8",
        )
    print(f"Wrote observation files to {output_dir}")


def main() -> None:
    padded_df = pad_neutral(
        input_path=INPUT_CSV,
        output_path=OUTPUT_CSV,
        start_year=START_YEAR,
        end_year=END_YEAR,
        neutral_value=NEUTRAL_VALUE,
    )
    write_metrics_files(padded_df, OUTPUT_METRICS_DIR)


if __name__ == "__main__":
    main()
