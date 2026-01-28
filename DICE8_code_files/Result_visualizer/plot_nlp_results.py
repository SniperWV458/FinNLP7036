import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# --- Configuration ---
TICKERS = ["^GSPC", "^IXIC", "^DJI", "GC=F", "SI=F", "CL=F"]
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2024-11-01"
SEEDS = [1, 2, 3, 4, 5]

# Directories
BACKTEST_DIR_NLP = "backtest_results_nlp_6assets"
BACKTEST_DIR_METRICS = "backtest_results_6assets"
PRICE_DIR = "metrics_used_6assets/price"
OUTPUT_DIR = "analysis_plots_nlp"


def load_price_data():
    """Load clean price data"""
    price_file = os.path.join(PRICE_DIR, "clean_data.csv")
    df = pd.read_csv(price_file, index_col="Date", parse_dates=True)
    # Filter to backtest period
    df = df.loc[BACKTEST_START:BACKTEST_END]
    return df


def calculate_equal_weight_performance(prices_df):
    """Calculate equal weight portfolio performance"""
    monthly_prices = prices_df.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    # Equal weight across all 6 assets
    equal_weight_returns = monthly_returns.mean(axis=1)

    results = []
    for i in range(len(equal_weight_returns)):
        month = equal_weight_returns.index[i].strftime("%Y-%m")
        roi = equal_weight_returns.iloc[i]
        results.append({
            "month": month,
            "roi": roi,
            "strategy": "Equal Weight"
        })

    return pd.DataFrame(results)


def calculate_sp500_performance(prices_df):
    """Calculate S&P 500 only performance"""
    sp500_prices = prices_df["^GSPC"]
    monthly_prices = sp500_prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    results = []
    for i in range(len(monthly_returns)):
        month = monthly_returns.index[i].strftime("%Y-%m")
        roi = monthly_returns.iloc[i]
        results.append({
            "month": month,
            "roi": roi,
            "strategy": "S&P 500"
        })

    return pd.DataFrame(results)


def build_monthly_returns(prices_df):
    """Build monthly returns indexed by YYYY-MM"""
    monthly_prices = prices_df.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    monthly_returns.index = monthly_returns.index.strftime("%Y-%m")
    return monthly_returns


def rebase_backtest_results(backtest_df, prices_df):
    """Recalculate ROI using price data rebased to backtest start."""
    monthly_returns = build_monthly_returns(prices_df)
    weight_cols = [f"weight_{ticker}" for ticker in TICKERS]

    missing_cols = [col for col in weight_cols if col not in backtest_df.columns]
    if missing_cols:
        raise ValueError(f"Missing weight columns: {missing_cols}")

    rets = []
    months = []
    for _, row in backtest_df.iterrows():
        month = row["month"]
        if month not in monthly_returns.index:
            continue
        weights = row[weight_cols].values.astype(float)
        asset_returns = monthly_returns.loc[month, TICKERS].values.astype(float)
        rets.append(float(np.dot(weights, asset_returns)))
        months.append(month)

    rebased_df = backtest_df[backtest_df["month"].isin(months)].copy()
    rebased_df["roi"] = rets
    return rebased_df


def load_backtest_results(backtest_dir, label):
    """Load all backtest results from a specific directory"""
    all_results = []

    model_types = ["ppo", "sac", "ddpg", "td3"]

    for model_type in model_types:
        model_dir = os.path.join(backtest_dir, model_type)
        if not os.path.exists(model_dir):
            print(f"Directory {model_dir} not found. Skipping.")
            continue

        for seed in SEEDS:
            file_path = os.path.join(model_dir, f"seed_{seed}.csv")
            if not os.path.exists(file_path):
                print(f"File {file_path} not found. Skipping.")
                continue

            df = pd.read_csv(file_path)
            df["model"] = model_type.upper()
            df["seed"] = seed
            df["strategy"] = f"{label} {model_type.upper()}_seed{seed}"
            all_results.append(df)

    if not all_results:
        raise ValueError("No backtest results found!")

    return pd.concat(all_results, ignore_index=True)


def aggregate_model_results(backtest_df):
    """Aggregate results by model type (average across seeds)"""
    aggregated = []

    for model in backtest_df["model"].unique():
        model_data = backtest_df[backtest_df["model"] == model]

        # Group by month and average across seeds
        monthly_avg = model_data.groupby("month").agg({
            "roi": "mean",
            "volatility": "mean",
            "mdd": "mean",
            "reward": "mean"
        }).reset_index()

        monthly_avg["strategy"] = f"{model} (avg)"
        aggregated.append(monthly_avg)

    return pd.concat(aggregated, ignore_index=True)


def calculate_cumulative_returns(df, roi_col="roi"):
    """Calculate cumulative returns from monthly returns"""
    df = df.sort_values("month")
    df["cumulative_return"] = (1 + df[roi_col]).cumprod() - 1
    return df


def calculate_performance_metrics(df, roi_col="roi"):
    """Calculate performance metrics for a strategy"""
    returns = df[roi_col].values

    # Total return
    total_return = (1 + returns).prod() - 1

    # Annualized return (assuming monthly data)
    n_months = len(returns)
    annualized_return = (1 + total_return) ** (12 / n_months) - 1 if n_months > 0 else 0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(12)

    # Sharpe ratio (assuming 0 risk-free rate)
    sharpe = annualized_return / volatility if volatility > 0 else 0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = drawdown.max()

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
    sortino = annualized_return / downside_std if downside_std > 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


def plot_cumulative_returns(backtest_df, equal_weight_df, sp500_df, output_dir, label):
    """Plot cumulative returns comparison"""
    plt.figure(figsize=(16, 10))

    # Aggregate model results
    agg_df = aggregate_model_results(backtest_df)

    # Calculate cumulative returns for each strategy
    for strategy in agg_df["strategy"].unique():
        strategy_data = agg_df[agg_df["strategy"] == strategy].copy()
        strategy_data = calculate_cumulative_returns(strategy_data)
        plt.plot(strategy_data["month"], strategy_data["cumulative_return"] * 100,
                 label=strategy, linewidth=2, marker='o', markersize=3)

    # Equal weight
    eq_df = equal_weight_df.copy()
    eq_df = calculate_cumulative_returns(eq_df)
    plt.plot(eq_df["month"], eq_df["cumulative_return"] * 100,
             label="Equal Weight", linewidth=2.5, linestyle="--", color="black", marker='s', markersize=3)

    # S&P 500
    sp_df = sp500_df.copy()
    sp_df = calculate_cumulative_returns(sp_df)
    plt.plot(sp_df["month"], sp_df["cumulative_return"] * 100,
             label="S&P 500", linewidth=2.5, linestyle=":", color="red", marker='^', markersize=3)

    plt.xlabel("Month", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Return (%)", fontsize=12, fontweight='bold')
    plt.title(f"Cumulative Returns: {label.upper()} RL Models vs Benchmarks", fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{label}_cumulative_returns.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_monthly_returns(backtest_df, equal_weight_df, sp500_df, output_dir, label):
    """Plot monthly returns comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    agg_df = aggregate_model_results(backtest_df)
    models = sorted(agg_df["strategy"].unique())

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Model data
        model_data = agg_df[agg_df["strategy"] == model].sort_values("month")

        # Merge with benchmarks
        eq_data = equal_weight_df[["month", "roi"]].copy()
        eq_data = eq_data.rename(columns={"roi": "equal_weight_roi"})

        sp_data = sp500_df[["month", "roi"]].copy()
        sp_data = sp_data.rename(columns={"roi": "sp500_roi"})

        merged = model_data.merge(eq_data, on="month", how="left")
        merged = merged.merge(sp_data, on="month", how="left")

        x = np.arange(len(merged))
        width = 0.25

        ax.bar(x - width, merged["roi"] * 100, width, label=model, alpha=0.8)
        ax.bar(x, merged["equal_weight_roi"] * 100, width, label="Equal Weight", alpha=0.8)
        ax.bar(x + width, merged["sp500_roi"] * 100, width, label="S&P 500", alpha=0.8)

        ax.set_xlabel("Month", fontsize=10, fontweight='bold')
        ax.set_ylabel("Monthly Return (%)", fontsize=10, fontweight='bold')
        ax.set_title(f"{model} - Monthly Returns", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Set x-axis labels (show every 3rd month)
        tick_positions = x[::3]
        tick_labels = merged["month"].values[::3]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{label}_monthly_returns_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_performance_metrics_table(backtest_df, equal_weight_df, sp500_df, output_dir, label):
    """Create performance metrics comparison table"""
    agg_df = aggregate_model_results(backtest_df)

    metrics_list = []

    # Calculate metrics for each model
    for strategy in sorted(agg_df["strategy"].unique()):
        strategy_data = agg_df[agg_df["strategy"] == strategy]
        metrics = calculate_performance_metrics(strategy_data)
        metrics["Strategy"] = strategy
        metrics_list.append(metrics)

    # Equal weight
    eq_metrics = calculate_performance_metrics(equal_weight_df)
    eq_metrics["Strategy"] = "Equal Weight"
    metrics_list.append(eq_metrics)

    # S&P 500
    sp_metrics = calculate_performance_metrics(sp500_df)
    sp_metrics["Strategy"] = "S&P 500"
    metrics_list.append(sp_metrics)

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[["Strategy", "Total Return", "Annualized Return", "Volatility",
                             "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate"]]

    # Save to CSV
    csv_file = os.path.join(output_dir, f"{label}_performance_metrics.csv")
    metrics_df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")

    # Create visual table
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    # Format values for display
    display_df = metrics_df.copy()
    display_df["Total Return"] = display_df["Total Return"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Annualized Return"] = display_df["Annualized Return"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Volatility"] = display_df["Volatility"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Sortino Ratio"] = display_df["Sortino Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Win Rate"] = display_df["Win Rate"].apply(lambda x: f"{x * 100:.1f}%")

    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        if i % 2 == 0:
            for j in range(len(display_df.columns)):
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.title(f"Performance Metrics Comparison ({label.upper()})", fontsize=14, fontweight='bold', pad=20)

    output_file = os.path.join(output_dir, f"{label}_performance_metrics_table.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    return metrics_df


def plot_sharpe_ratio_comparison(backtest_df, equal_weight_df, sp500_df, output_dir, label):
    """Plot Sharpe ratio comparison"""
    agg_df = aggregate_model_results(backtest_df)

    sharpe_ratios = []
    strategies = []

    # Models
    for strategy in sorted(agg_df["strategy"].unique()):
        strategy_data = agg_df[agg_df["strategy"] == strategy]
        metrics = calculate_performance_metrics(strategy_data)
        sharpe_ratios.append(metrics["Sharpe Ratio"])
        strategies.append(strategy)

    # Benchmarks
    eq_metrics = calculate_performance_metrics(equal_weight_df)
    sharpe_ratios.append(eq_metrics["Sharpe Ratio"])
    strategies.append("Equal Weight")

    sp_metrics = calculate_performance_metrics(sp500_df)
    sharpe_ratios.append(sp_metrics["Sharpe Ratio"])
    strategies.append("S&P 500")

    # Plot
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#000000', '#8B0000']
    bars = plt.bar(strategies, sharpe_ratios, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, sharpe_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel("Strategy", fontsize=12, fontweight='bold')
    plt.ylabel("Sharpe Ratio", fontsize=12, fontweight='bold')
    plt.title(f"Sharpe Ratio Comparison ({label.upper()})", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{label}_sharpe_ratio_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_volatility_vs_return(backtest_df, equal_weight_df, sp500_df, output_dir, label):
    """Plot risk-return scatter"""
    agg_df = aggregate_model_results(backtest_df)

    data_points = []

    # Models
    for strategy in sorted(agg_df["strategy"].unique()):
        strategy_data = agg_df[agg_df["strategy"] == strategy]
        metrics = calculate_performance_metrics(strategy_data)
        data_points.append({
            "Strategy": strategy,
            "Return": metrics["Annualized Return"] * 100,
            "Volatility": metrics["Volatility"] * 100,
            "Type": "RL Model"
        })

    # Benchmarks
    eq_metrics = calculate_performance_metrics(equal_weight_df)
    data_points.append({
        "Strategy": "Equal Weight",
        "Return": eq_metrics["Annualized Return"] * 100,
        "Volatility": eq_metrics["Volatility"] * 100,
        "Type": "Benchmark"
    })

    sp_metrics = calculate_performance_metrics(sp500_df)
    data_points.append({
        "Strategy": "S&P 500",
        "Return": sp_metrics["Annualized Return"] * 100,
        "Volatility": sp_metrics["Volatility"] * 100,
        "Type": "Benchmark"
    })

    df = pd.DataFrame(data_points)

    # Plot
    plt.figure(figsize=(12, 8))

    for type_name, marker, size in [("RL Model", 'o', 100), ("Benchmark", 's', 150)]:
        subset = df[df["Type"] == type_name]
        plt.scatter(subset["Volatility"], subset["Return"],
                    s=size, alpha=0.7, label=type_name, marker=marker, edgecolors='black', linewidths=1.5)

    # Add labels
    for _, row in df.iterrows():
        plt.annotate(row["Strategy"],
                     (row["Volatility"], row["Return"]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.xlabel("Volatility (% per year)", fontsize=12, fontweight='bold')
    plt.ylabel("Annualized Return (%)", fontsize=12, fontweight='bold')
    plt.title(f"Risk-Return Profile ({label.upper()})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{label}_risk_return_scatter.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_drawdown_analysis(backtest_df, equal_weight_df, sp500_df, output_dir, label):
    """Plot drawdown comparison"""
    agg_df = aggregate_model_results(backtest_df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    all_strategies = []

    # Models
    for strategy in sorted(agg_df["strategy"].unique()):
        all_strategies.append((strategy, agg_df[agg_df["strategy"] == strategy], "roi"))

    # Benchmarks
    all_strategies.append(("Equal Weight", equal_weight_df, "roi"))
    all_strategies.append(("S&P 500", sp500_df, "roi"))

    for idx, (strategy_name, strategy_data, roi_col) in enumerate(all_strategies):
        if idx >= len(axes):
            break

        ax = axes[idx]
        strategy_data = strategy_data.sort_values("month")

        # Calculate drawdown
        returns = strategy_data[roi_col].values
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max * 100

        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        ax.plot(drawdown, color='darkred', linewidth=2)

        ax.set_xlabel("Month", fontsize=9, fontweight='bold')
        ax.set_ylabel("Drawdown (%)", fontsize=9, fontweight='bold')
        ax.set_title(f"{strategy_name} - Drawdown", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # Add max drawdown annotation
        max_dd = drawdown.max()
        ax.text(0.5, 0.95, f'Max DD: {max_dd:.2f}%',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{label}_drawdown_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_seed_variability(backtest_df, output_dir, label):
    """Plot variability across different seeds for each model"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    models = sorted(backtest_df["model"].unique())

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = backtest_df[backtest_df["model"] == model]

        # Calculate cumulative returns for each seed
        for seed in SEEDS:
            seed_data = model_data[model_data["seed"] == seed].sort_values("month")
            seed_data = calculate_cumulative_returns(seed_data)
            ax.plot(seed_data["month"], seed_data["cumulative_return"] * 100,
                    label=f"Seed {seed}", alpha=0.7, linewidth=1.5, marker='o', markersize=2)

        # Calculate and plot average
        avg_data = model_data.groupby("month")["roi"].mean().reset_index()
        avg_data = calculate_cumulative_returns(avg_data)
        ax.plot(avg_data["month"], avg_data["cumulative_return"] * 100,
                label="Average", color='black', linewidth=3, linestyle='--')

        ax.set_xlabel("Month", fontsize=10, fontweight='bold')
        ax.set_ylabel("Cumulative Return (%)", fontsize=10, fontweight='bold')
        ax.set_title(f"{model} - Seed Variability", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Reduce x-axis label density
        n_ticks = len(avg_data)
        tick_spacing = max(1, n_ticks // 10)
        ax.set_xticks(ax.get_xticks()[::tick_spacing])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{label}_seed_variability.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main analysis pipeline"""
    print("=" * 60)
    print("RL Backtest Analysis and Visualization (NLP + Metrics)")
    print("=" * 60)

    # Create base output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load data
    print("Loading price data...")
    prices_df = load_price_data()

    print("Calculating equal weight portfolio performance...")
    equal_weight_df = calculate_equal_weight_performance(prices_df)

    print("Calculating S&P 500 performance...")
    sp500_df = calculate_sp500_performance(prices_df)

    def run_analysis(label, backtest_dir):
        output_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"{label.upper()} BACKTEST ANALYSIS")
        print("=" * 60)

        print("Loading backtest results...")
        backtest_df = load_backtest_results(backtest_dir, label)
        backtest_df = rebase_backtest_results(backtest_df, prices_df)
        print(f"Loaded {len(backtest_df)} backtest records\n")

        print("Generating visualizations...")
        print("1. Cumulative returns plot...")
        plot_cumulative_returns(backtest_df, equal_weight_df, sp500_df, output_dir, label)

        print("2. Monthly returns comparison...")
        plot_monthly_returns(backtest_df, equal_weight_df, sp500_df, output_dir, label)

        print("3. Performance metrics table...")
        metrics_df = plot_performance_metrics_table(backtest_df, equal_weight_df, sp500_df, output_dir, label)

        print("4. Sharpe ratio comparison...")
        plot_sharpe_ratio_comparison(backtest_df, equal_weight_df, sp500_df, output_dir, label)

        print("5. Risk-return scatter plot...")
        plot_volatility_vs_return(backtest_df, equal_weight_df, sp500_df, output_dir, label)

        print("6. Drawdown analysis...")
        plot_drawdown_analysis(backtest_df, equal_weight_df, sp500_df, output_dir, label)

        print("7. Seed variability analysis...")
        plot_seed_variability(backtest_df, output_dir, label)

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"All plots saved to: {output_dir}")
        print("=" * 60)

        # Print summary statistics
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(metrics_df.to_string(index=False))
        print("=" * 60)

    run_analysis("nlp", BACKTEST_DIR_NLP)
    run_analysis("metrics", BACKTEST_DIR_METRICS)


if __name__ == "__main__":
    main()
