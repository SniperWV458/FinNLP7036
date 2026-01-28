import os
import pandas as pd
from transformers import pipeline

NEWS_CLEANED_DIR = os.path.join("data", "textual_data", "news_cleaned")
STOCKWITS_CLEANED_DIR = os.path.join("data", "textual_data", "stockwits_cleaned_data")
OUTPUT_BASE_DIR = "sentiment_analysis_cleaned"

# Set True to combine news + stocktwits into a single summary/matrix.
# Set False to store summaries separately by source.
COMBINE_SOURCES = True

sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")

EQUIVALENT_TICKERS = {
    "SPX": "GSPC",
    "COMPQ": "IXIC",
    "DIA": "DJI",
    "GLD": "GOLD",
    "SLV": "SILVER",
    "USO": "OIL",
}


def normalize_query_group(group_name: str) -> str:
    return EQUIVALENT_TICKERS.get(group_name, group_name)


def extract_group_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.startswith("cleaned_"):
        stem = stem[len("cleaned_"):]
    return stem.split("_")[0]


def add_sentiment_labels(df: pd.DataFrame, text_col: str) -> pd.DataFrame | None:
    if text_col not in df.columns:
        return None

    sentiment_labels = []
    sentiment_scores = []

    for text in df[text_col]:
        if isinstance(text, str) and text.strip():
            truncated_text = text[:500]
            sentiment = sentiment_pipeline(truncated_text)[0]
            sentiment_labels.append(sentiment["label"])
            sentiment_scores.append(sentiment["score"])
        else:
            sentiment_labels.append("Error")
            sentiment_scores.append(0.0)

    df["Sentiment_Label"] = sentiment_labels
    df["Sentiment_Score"] = sentiment_scores
    return df


def calculate_average_score(df: pd.DataFrame) -> float | None:
    if "Sentiment_Label" not in df.columns or "Sentiment_Score" not in df.columns:
        return None
    positive_scores = df[df["Sentiment_Label"] == "positive"]["Sentiment_Score"]
    negative_scores = df[df["Sentiment_Label"] == "negative"]["Sentiment_Score"]
    num_positive = len(positive_scores)
    num_negative = len(negative_scores)
    if num_positive + num_negative == 0:
        return None
    return (positive_scores.sum() - negative_scores.sum()) / (num_positive + num_negative)


def build_monthly_summary(df: pd.DataFrame, source_name: str | None = None) -> pd.DataFrame:
    summary_rows = []
    for (year, month, group_name), group_df in df.groupby(["Year", "Month", "Query Group"]):
        avg_score = calculate_average_score(group_df)
        if avg_score is not None:
            summary_rows.append([year, month, group_name, avg_score])

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["Year", "Month", "Query Group", "Average Score"],
    )
    if source_name:
        summary_df["Source"] = source_name
    return summary_df


def process_source(
        source_name: str,
        input_dir: str,
        text_col: str,
        year_col: str,
        month_col: str,
) -> list[pd.DataFrame]:
    output_dir = os.path.join(OUTPUT_BASE_DIR, source_name)
    os.makedirs(output_dir, exist_ok=True)
    labeled_dfs = []

    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path)

        if year_col not in df.columns or month_col not in df.columns:
            print(f"Skipping {file_name} (missing year/month columns)")
            continue

        if text_col not in df.columns:
            print(f"Skipping {file_name} (missing text column: {text_col})")
            continue

        group_value = None
        if "asset_id" in df.columns:
            asset_vals = df["asset_id"].dropna()
            if not asset_vals.empty:
                group_value = str(asset_vals.iloc[0])
        if not group_value:
            group_value = extract_group_from_filename(file_name)

        df = df.rename(columns={year_col: "Year", month_col: "Month"})
        df["Query Group"] = normalize_query_group(group_value)
        df["Source"] = source_name

        df = add_sentiment_labels(df, text_col)
        if df is None:
            print(f"Skipping {file_name} (sentiment labeling failed)")
            continue

        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False, encoding="utf-8")
        labeled_dfs.append(df)
        print(f"Updated sentiment for {output_path}")

    if labeled_dfs:
        summary_df = build_monthly_summary(
            pd.concat(labeled_dfs, ignore_index=True),
            source_name=source_name,
        )
        summary_path = os.path.join(output_dir, "monthly_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        matrix = summary_df.pivot_table(
            index=["Year", "Month"],
            columns="Query Group",
            values="Average Score",
        )
        matrix.to_csv(os.path.join(output_dir, "unified_matrix.csv"), encoding="utf-8")
        print(f"Created summaries for {source_name}: {summary_path}")
    else:
        print(f"No labeled data found for {source_name}")

    return labeled_dfs


def main() -> None:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    all_labeled = []
    all_labeled.extend(
        process_source(
            source_name="news_cleaned",
            input_dir=NEWS_CLEANED_DIR,
            text_col="title",
            year_col="year",
            month_col="month",
        )
    )
    all_labeled.extend(
        process_source(
            source_name="stockwits_cleaned_data",
            input_dir=STOCKWITS_CLEANED_DIR,
            text_col="Text",
            year_col="Year",
            month_col="Month",
        )
    )

    if COMBINE_SOURCES and all_labeled:
        combined_df = pd.concat(all_labeled, ignore_index=True)
        combined_summary = build_monthly_summary(combined_df)
        combined_summary_path = os.path.join(OUTPUT_BASE_DIR, "combined_monthly_summary.csv")
        combined_summary.to_csv(combined_summary_path, index=False, encoding="utf-8")
        combined_matrix = combined_summary.pivot_table(
            index=["Year", "Month"],
            columns="Query Group",
            values="Average Score",
        )
        combined_matrix.to_csv(
            os.path.join(OUTPUT_BASE_DIR, "combined_unified_matrix.csv"),
            encoding="utf-8",
        )
        print(f"Created combined summaries: {combined_summary_path}")
    elif COMBINE_SOURCES:
        print("No labeled data available to combine.")


if __name__ == "__main__":
    main()
