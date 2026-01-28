import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# --- Configuration ---
TICKERS = ["^GSPC", "^IXIC", "^DJI", "GC=F", "SI=F", "CL=F"]
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2024-11-01"

# Directories
SUPER_DIR = "SUPER_6assets"
META_METRICS_DIR = "Meta_6assets/Metrics_obs"
META_NLP_DIR = "Meta_6assets/NLP_obs"
PRICE_DIR = "metrics_used_6assets/price"
OUTPUT_DIR = "analysis_plots_super"


def load_price_data():
    """Load clean price data"""
    price_file = os.path.join(PRICE_DIR, "clean_data.csv")
    df = pd.read_csv(price_file, index_col="Date", parse_dates=True)
    df = df.loc[BACKTEST_START:BACKTEST_END]
    return df


def calculate_equal_weight_performance(prices_df):
    """Calculate equal weight portfolio performance"""
    monthly_prices = prices_df.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    equal_weight_returns = monthly_returns.mean(axis=1)

    results = []
    for i in range(len(equal_weight_returns)):
        month = equal_weight_returns.index[i].strftime("%Y-%m")
        roi = equal_weight_returns.iloc[i]
        results.append({"month": month, "roi": roi, "strategy": "Equal Weight"})

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
        results.append({"month": month, "roi": roi, "strategy": "S&P 500"})

    return pd.DataFrame(results)


def load_all_results():
    """Load all agent results"""
    results = {}

    # Super Meta-Agent
    super_file = os.path.join(SUPER_DIR, "backtest_super_meta_seed_1.csv")
    if os.path.exists(super_file):
        df = pd.read_csv(super_file)
        df["strategy"] = "Super Meta-Agent"
        results["Super Meta"] = df
        print(f"Loaded Super Meta-Agent: {len(df)} rows")
    else:
        print(f"Warning: {super_file} not found")

    # Metrics Meta-Agent
    metrics_file = os.path.join(META_METRICS_DIR, "backtest_meta_seed_1.csv")
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        df["strategy"] = "Meta-Agent (Metrics)"
        results["Metrics Meta"] = df
        print(f"Loaded Metrics Meta-Agent: {len(df)} rows")
    else:
        print(f"Warning: {metrics_file} not found")

    # NLP Meta-Agent
    nlp_file = os.path.join(META_NLP_DIR, "backtest_meta_seed_1.csv")
    if os.path.exists(nlp_file):
        df = pd.read_csv(nlp_file)
        df["strategy"] = "Meta-Agent (NLP)"
        results["NLP Meta"] = df
        print(f"Loaded NLP Meta-Agent: {len(df)} rows")
    else:
        print(f"Warning: {nlp_file} not found")

    if not results:
        raise ValueError("No results found!")

    return results


def build_monthly_returns(prices_df):
    """Build monthly returns indexed by YYYY-MM"""
    monthly_prices = prices_df.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    monthly_returns.index = monthly_returns.index.strftime("%Y-%m")
    return monthly_returns


def rebase_results_with_prices(results, prices_df):
    """Recalculate ROI using price data rebased to backtest start."""
    monthly_returns = build_monthly_returns(prices_df)
    rebased_results = {}
    weight_cols = [f"weight_{ticker}" for ticker in TICKERS]

    for name, df in results.items():
        df = df.copy()
        missing_cols = [col for col in weight_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing weight columns for {name}: {missing_cols}")

        rets = []
        months = []
        for _, row in df.iterrows():
            month = row["month"]
            if month not in monthly_returns.index:
                continue
            weights = row[weight_cols].values.astype(float)
            asset_returns = monthly_returns.loc[month, TICKERS].values.astype(float)
            rets.append(float(np.dot(weights, asset_returns)))
            months.append(month)

        df = df[df["month"].isin(months)].copy()
        df["roi"] = rets
        rebased_results[name] = df

    return rebased_results


def calculate_cumulative_returns(df, roi_col="roi"):
    """Calculate cumulative returns from monthly returns"""
    df = df.sort_values("month")
    df["cumulative_return"] = (1 + df[roi_col]).cumprod() - 1
    return df


def add_baseline_point(df, start_date):
    """Add a baseline point at backtest start date."""
    df = df.sort_values("month").copy()
    df["plot_date"] = pd.to_datetime(df["month"] + "-01") + pd.offsets.MonthEnd(0)
    baseline = pd.DataFrame({
        "month": [pd.to_datetime(start_date).strftime("%Y-%m")],
        "cumulative_return": [0.0],
        "plot_date": [pd.to_datetime(start_date)]
    })
    return pd.concat([baseline, df], ignore_index=True)


def calculate_performance_metrics(df, roi_col="roi"):
    """Calculate performance metrics for a strategy"""
    returns = df[roi_col].values

    total_return = (1 + returns).prod() - 1
    n_months = len(returns)
    annualized_return = (1 + total_return) ** (12 / n_months) - 1 if n_months > 0 else 0
    volatility = returns.std() * np.sqrt(12)
    sharpe = annualized_return / volatility if volatility > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = drawdown.max()

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
    sortino = annualized_return / downside_std if downside_std > 0 else 0

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


def plot_cumulative_returns_all(all_results, equal_weight_df, sp500_df, output_dir):
    """Plot cumulative returns comparison for all strategies"""
    plt.figure(figsize=(16, 10))

    # Define colors for each strategy
    colors = {
        'Super Meta': '#e74c3c',  # Red (highlight)
        'Metrics Meta': '#1f77b4',  # Blue
        'NLP Meta': '#ff7f0e',  # Orange
    }

    # Plot agents in order
    order = ['Metrics Meta', 'NLP Meta', 'Super Meta']
    for name in order:
        if name in all_results:
            df = all_results[name].copy()
            df = calculate_cumulative_returns(df)
            df_plot = add_baseline_point(df, BACKTEST_START)
            linewidth = 3.5 if name == 'Super Meta' else 2.5
            plt.plot(df_plot["plot_date"], df_plot["cumulative_return"] * 100,
                     label=df["strategy"].iloc[0], linewidth=linewidth,
                     marker='o', markersize=5 if name == 'Super Meta' else 4,
                     color=colors.get(name, None), alpha=0.9)

    # Equal weight
    eq_df = equal_weight_df.copy()
    eq_df = calculate_cumulative_returns(eq_df)
    eq_plot = add_baseline_point(eq_df, BACKTEST_START)
    plt.plot(eq_plot["plot_date"], eq_plot["cumulative_return"] * 100,
             label="Equal Weight", linewidth=2.5, linestyle="--",
             color="black", marker='s', markersize=4)

    # S&P 500
    sp_df = sp500_df.copy()
    sp_df = calculate_cumulative_returns(sp_df)
    sp_plot = add_baseline_point(sp_df, BACKTEST_START)
    plt.plot(sp_plot["plot_date"], sp_plot["cumulative_return"] * 100,
             label="S&P 500", linewidth=2.5, linestyle=":",
             color="darkred", marker='^', markersize=4)

    plt.xlabel("Month", fontsize=13, fontweight='bold')
    plt.ylabel("Cumulative Return (%)", fontsize=13, fontweight='bold')
    plt.title("Cumulative Returns: Super Meta-Agent vs All Strategies", fontsize=15, fontweight='bold')
    plt.legend(loc="best", fontsize=12, framealpha=0.9)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, "super_cumulative_returns_all.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_performance_metrics_table(all_results, equal_weight_df, sp500_df, output_dir):
    """Create comprehensive performance metrics comparison table"""
    metrics_list = []

    # Calculate metrics for each agent (in specific order)
    order = ['Super Meta', 'Metrics Meta', 'NLP Meta']
    for name in order:
        if name in all_results:
            df = all_results[name]
            metrics = calculate_performance_metrics(df)
            metrics["Strategy"] = df["strategy"].iloc[0]
            metrics_list.append(metrics)

    # Benchmarks
    eq_metrics = calculate_performance_metrics(equal_weight_df)
    eq_metrics["Strategy"] = "Equal Weight"
    metrics_list.append(eq_metrics)

    sp_metrics = calculate_performance_metrics(sp500_df)
    sp_metrics["Strategy"] = "S&P 500"
    metrics_list.append(sp_metrics)

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[["Strategy", "Total Return", "Annualized Return", "Volatility",
                             "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate"]]

    # Save to CSV
    csv_file = os.path.join(output_dir, "super_performance_metrics.csv")
    metrics_df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")

    # Create visual table
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis('tight')
    ax.axis('off')

    # Format values for display
    display_df = metrics_df.copy()
    display_df["Total Return"] = display_df["Total Return"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Annualized Return"] = display_df["Annualized Return"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Volatility"] = display_df["Volatility"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Sortino Ratio"] = display_df["Sortino Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Calmar Ratio"] = display_df["Calmar Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Win Rate"] = display_df["Win Rate"].apply(lambda x: f"{x * 100:.1f}%")

    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#2E75B6')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight Super Meta-Agent
    for j in range(len(display_df.columns)):
        table[(1, j)].set_facecolor('#F8CECC')  # Light red
        table[(1, j)].set_text_props(weight='bold')

    # Highlight other meta-agents
    for i in range(2, len(order) + 1):
        for j in range(len(display_df.columns)):
            table[(i, j)].set_facecolor('#D6EAF8')  # Light blue

    # Highlight benchmarks
    for i in range(len(order) + 1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            table[(i, j)].set_facecolor('#FCF3CF')  # Light yellow

    plt.title("Performance Metrics: Super Meta-Agent vs All Strategies", fontsize=15, fontweight='bold', pad=20)

    output_file = os.path.join(output_dir, "super_performance_metrics_table.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    return metrics_df


def plot_sharpe_ratio_comparison(all_results, equal_weight_df, sp500_df, output_dir):
    """Plot Sharpe ratio comparison"""
    sharpe_ratios = []
    strategies = []
    colors_list = []

    # Agents in order
    order = ['Super Meta', 'Metrics Meta', 'NLP Meta']
    color_map = {'Super Meta': '#e74c3c', 'Metrics Meta': '#1f77b4', 'NLP Meta': '#ff7f0e'}

    for name in order:
        if name in all_results:
            df = all_results[name]
            metrics = calculate_performance_metrics(df)
            sharpe_ratios.append(metrics["Sharpe Ratio"])
            strategies.append(df["strategy"].iloc[0])
            colors_list.append(color_map[name])

    # Benchmarks
    eq_metrics = calculate_performance_metrics(equal_weight_df)
    sharpe_ratios.append(eq_metrics["Sharpe Ratio"])
    strategies.append("Equal Weight")
    colors_list.append('#000000')

    sp_metrics = calculate_performance_metrics(sp500_df)
    sharpe_ratios.append(sp_metrics["Sharpe Ratio"])
    strategies.append("S&P 500")
    colors_list.append('#8B0000')

    # Plot
    plt.figure(figsize=(14, 7))
    bars = plt.bar(strategies, sharpe_ratios, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, value in zip(bars, sharpe_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.3f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xlabel("Strategy", fontsize=13, fontweight='bold')
    plt.ylabel("Sharpe Ratio", fontsize=13, fontweight='bold')
    plt.title("Sharpe Ratio Comparison: Super Meta-Agent vs All Strategies", fontsize=15, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    output_file = os.path.join(output_dir, "super_sharpe_ratio_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_risk_return_scatter(all_results, equal_weight_df, sp500_df, output_dir):
    """Plot risk-return scatter"""
    data_points = []

    # Agents
    order = ['Super Meta', 'Metrics Meta', 'NLP Meta']
    for name in order:
        if name in all_results:
            df = all_results[name]
            metrics = calculate_performance_metrics(df)
            data_points.append({
                "Strategy": df["strategy"].iloc[0],
                "Return": metrics["Annualized Return"] * 100,
                "Volatility": metrics["Volatility"] * 100,
                "Type": "Super Meta" if name == "Super Meta" else "Meta-Agent"
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

    df_plot = pd.DataFrame(data_points)

    # Plot
    plt.figure(figsize=(13, 9))

    for type_name, marker, size, color in [
        ("Super Meta", '*', 400, '#e74c3c'),
        ("Meta-Agent", 'D', 200, '#3498db'),
        ("Benchmark", 's', 150, '#95a5a6')
    ]:
        subset = df_plot[df_plot["Type"] == type_name]
        plt.scatter(subset["Volatility"], subset["Return"],
                    s=size, alpha=0.8, label=type_name, marker=marker,
                    edgecolors='black', linewidths=2, color=color)

    # Add labels
    for _, row in df_plot.iterrows():
        fontsize = 11 if row["Type"] == "Super Meta" else 10
        fontweight = 'bold' if row["Type"] == "Super Meta" else 'normal'
        plt.annotate(row["Strategy"],
                     (row["Volatility"], row["Return"]),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=fontsize, fontweight=fontweight,
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='yellow' if row["Type"] == "Super Meta" else 'lightgray',
                               alpha=0.6, edgecolor='black'))

    plt.xlabel("Volatility (% per year)", fontsize=13, fontweight='bold')
    plt.ylabel("Annualized Return (%)", fontsize=13, fontweight='bold')
    plt.title("Risk-Return Profile: Super Meta-Agent vs All Strategies", fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, "super_risk_return_scatter.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_portfolio_weights_comparison(all_results, output_dir):
    """Plot average portfolio weights for all agents"""
    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 4 * len(all_results)))

    if len(all_results) == 1:
        axes = [axes]

    order = ['Super Meta', 'Metrics Meta', 'NLP Meta']

    for idx, name in enumerate(order):
        if name in all_results:
            ax = axes[idx]
            df = all_results[name]

            # Extract weights
            weight_cols = [col for col in df.columns if 'weight_' in col]
            avg_weights = df[weight_cols].mean()

            # Create labels
            labels = [col.replace('weight_', '') for col in weight_cols]

            # Plot
            colors_palette = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            bars = ax.bar(labels, avg_weights, color=colors_palette, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Add value labels
            for bar, value in zip(bars, avg_weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{value:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xlabel("Asset", fontsize=11, fontweight='bold')
            ax.set_ylabel("Average Weight", fontsize=11, fontweight='bold')
            ax.set_title(f"{df['strategy'].iloc[0]} - Average Portfolio Weights",
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(avg_weights) * 1.2)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "super_portfolio_weights_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_hierarchical_structure(output_dir):
    """Create a visual representation of the hierarchical agent structure"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, "Hierarchical Agent Structure", ha='center', fontsize=18, fontweight='bold')

    # Level 1: Base RL Agents (4 agents × 2 types = 8 total)
    level1_y = 7
    agents = ['PPO', 'SAC', 'DDPG', 'TD3']

    # Metrics agents
    ax.text(2.5, level1_y + 1, "Metrics-based Agents", ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    for i, agent in enumerate(agents):
        x = 1 + i * 0.8
        ax.add_patch(plt.Rectangle((x - 0.3, level1_y - 0.3), 0.6, 0.6,
                                   facecolor='#3498db', edgecolor='black', linewidth=2))
        ax.text(x, level1_y, agent, ha='center', va='center', fontsize=9, fontweight='bold')

    # NLP agents
    ax.text(7.5, level1_y + 1, "NLP-based Agents", ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    for i, agent in enumerate(agents):
        x = 6 + i * 0.8
        ax.add_patch(plt.Rectangle((x - 0.3, level1_y - 0.3), 0.6, 0.6,
                                   facecolor='#f39c12', edgecolor='black', linewidth=2))
        ax.text(x, level1_y, agent, ha='center', va='center', fontsize=9, fontweight='bold')

    # Level 2: Meta-Agents
    level2_y = 4.5

    # Metrics Meta
    ax.add_patch(plt.Rectangle((1.5, level2_y - 0.5), 2, 1,
                               facecolor='#2980b9', edgecolor='black', linewidth=3))
    ax.text(2.5, level2_y, "Meta-Agent\n(Metrics)", ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # NLP Meta
    ax.add_patch(plt.Rectangle((6.5, level2_y - 0.5), 2, 1,
                               facecolor='#e67e22', edgecolor='black', linewidth=3))
    ax.text(7.5, level2_y, "Meta-Agent\n(NLP)", ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Level 3: Super Meta-Agent
    level3_y = 2
    ax.add_patch(plt.Rectangle((3.5, level3_y - 0.6), 3, 1.2,
                               facecolor='#c0392b', edgecolor='black', linewidth=4))
    ax.text(5, level3_y, "SUPER META-AGENT", ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

    # Draw arrows
    # Metrics agents to Metrics Meta
    for i in range(4):
        x_start = 1 + i * 0.8
        ax.annotate('', xy=(2.5, level2_y + 0.5), xytext=(x_start, level1_y - 0.3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2980b9', alpha=0.6))

    # NLP agents to NLP Meta
    for i in range(4):
        x_start = 6 + i * 0.8
        ax.annotate('', xy=(7.5, level2_y + 0.5), xytext=(x_start, level1_y - 0.3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#e67e22', alpha=0.6))

    # Meta agents to Super Meta
    ax.annotate('', xy=(4.5, level3_y + 0.6), xytext=(2.5, level2_y - 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='#c0392b', alpha=0.8))
    ax.annotate('', xy=(5.5, level3_y + 0.6), xytext=(7.5, level2_y - 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='#c0392b', alpha=0.8))

    # Final output
    ax.text(5, 0.5, "Final Portfolio Allocation", ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.annotate('', xy=(5, 0.8), xytext=(5, level3_y - 0.6),
                arrowprops=dict(arrowstyle='->', lw=3, color='green', alpha=0.8))

    plt.tight_layout()
    output_file = os.path.join(output_dir, "hierarchical_structure.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("SUPER META-AGENT ANALYSIS AND VISUALIZATION")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load data
    print("Loading price data...")
    prices_df = load_price_data()

    print("Calculating equal weight portfolio performance...")
    equal_weight_df = calculate_equal_weight_performance(prices_df)

    print("Calculating S&P 500 performance...")
    sp500_df = calculate_sp500_performance(prices_df)

    print("Loading all agent results...")
    all_results = load_all_results()
    all_results = rebase_results_with_prices(all_results, prices_df)

    print("\nGenerating visualizations...")

    print("1. Cumulative returns plot...")
    plot_cumulative_returns_all(all_results, equal_weight_df, sp500_df, OUTPUT_DIR)

    print("2. Performance metrics table...")
    metrics_df = plot_performance_metrics_table(all_results, equal_weight_df, sp500_df, OUTPUT_DIR)

    print("3. Sharpe ratio comparison...")
    plot_sharpe_ratio_comparison(all_results, equal_weight_df, sp500_df, OUTPUT_DIR)

    print("4. Risk-return scatter plot...")
    plot_risk_return_scatter(all_results, equal_weight_df, sp500_df, OUTPUT_DIR)

    print("5. Portfolio weights comparison...")
    plot_portfolio_weights_comparison(all_results, OUTPUT_DIR)

    print("6. Hierarchical structure diagram...")
    plot_hierarchical_structure(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 80)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
