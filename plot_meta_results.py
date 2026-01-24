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

# Directories
META_METRICS_DIR = "Meta_6assets/Metrics_obs"
META_NLP_DIR = "Meta_6assets/NLP_obs"
PRICE_DIR = "metrics_used_6assets/price"
OUTPUT_DIR = "analysis_plots_meta"


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


def load_meta_results():
    """Load meta-agent backtest results"""
    results = {}
    
    # Load Metrics Meta-Agent
    metrics_file = os.path.join(META_METRICS_DIR, "backtest_meta_seed_1.csv")
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        df["strategy"] = "Meta-Agent (Metrics)"
        results["Metrics Meta"] = df
        print(f"Loaded Metrics Meta-Agent: {len(df)} rows")
    else:
        print(f"Warning: {metrics_file} not found")
    
    # Load NLP Meta-Agent
    nlp_file = os.path.join(META_NLP_DIR, "backtest_meta_seed_1.csv")
    if os.path.exists(nlp_file):
        df = pd.read_csv(nlp_file)
        df["strategy"] = "Meta-Agent (NLP)"
        results["NLP Meta"] = df
        print(f"Loaded NLP Meta-Agent: {len(df)} rows")
    else:
        print(f"Warning: {nlp_file} not found")
    
    if not results:
        raise ValueError("No meta-agent results found!")
    
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
    
    # Calmar ratio
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


def plot_cumulative_returns(meta_results, equal_weight_df, sp500_df, output_dir):
    """Plot cumulative returns comparison"""
    plt.figure(figsize=(16, 10))
    
    # Meta-agents
    colors = {'Metrics Meta': '#1f77b4', 'NLP Meta': '#ff7f0e'}
    for name, df in meta_results.items():
        df_copy = df.copy()
        df_copy = calculate_cumulative_returns(df_copy)
        df_plot = add_baseline_point(df_copy, BACKTEST_START)
        plt.plot(df_plot["plot_date"], df_plot["cumulative_return"] * 100, 
                label=df_copy["strategy"].iloc[0], linewidth=2.5, 
                marker='o', markersize=4, color=colors.get(name, None))
    
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
            color="red", marker='^', markersize=4)
    
    plt.xlabel("Month", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Return (%)", fontsize=12, fontweight='bold')
    plt.title("Cumulative Returns: Meta-Agents vs Benchmarks", fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "meta_cumulative_returns.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_monthly_returns(meta_results, equal_weight_df, sp500_df, output_dir):
    """Plot monthly returns comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    
    # Combine all strategies
    all_strategies = []
    for name, df in meta_results.items():
        all_strategies.append((df["strategy"].iloc[0], df[["month", "roi"]].copy()))
    all_strategies.append(("Equal Weight", equal_weight_df[["month", "roi"]].copy()))
    all_strategies.append(("S&P 500", sp500_df[["month", "roi"]].copy()))
    
    for idx, (strategy_name, strategy_data) in enumerate(all_strategies[:2]):
        ax = axes[idx]
        
        # Merge with benchmarks
        eq_data = equal_weight_df[["month", "roi"]].copy()
        eq_data = eq_data.rename(columns={"roi": "equal_weight_roi"})
        
        sp_data = sp500_df[["month", "roi"]].copy()
        sp_data = sp_data.rename(columns={"roi": "sp500_roi"})
        
        merged = strategy_data.merge(eq_data, on="month", how="left")
        merged = merged.merge(sp_data, on="month", how="left")
        merged = merged.sort_values("month")
        
        x = np.arange(len(merged))
        width = 0.25
        
        ax.bar(x - width, merged["roi"] * 100, width, label=strategy_name, alpha=0.8)
        ax.bar(x, merged["equal_weight_roi"] * 100, width, label="Equal Weight", alpha=0.8)
        ax.bar(x + width, merged["sp500_roi"] * 100, width, label="S&P 500", alpha=0.8)
        
        ax.set_xlabel("Month", fontsize=11, fontweight='bold')
        ax.set_ylabel("Monthly Return (%)", fontsize=11, fontweight='bold')
        ax.set_title(f"{strategy_name} - Monthly Returns", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Set x-axis labels (show every 3rd month)
        tick_positions = x[::3]
        tick_labels = merged["month"].values[::3]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "meta_monthly_returns_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_performance_metrics_table(meta_results, equal_weight_df, sp500_df, output_dir):
    """Create performance metrics comparison table"""
    metrics_list = []
    
    # Calculate metrics for each meta-agent
    for name, df in meta_results.items():
        metrics = calculate_performance_metrics(df)
        metrics["Strategy"] = df["strategy"].iloc[0]
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
                              "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate"]]
    
    # Save to CSV
    csv_file = os.path.join(output_dir, "meta_performance_metrics.csv")
    metrics_df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Format values for display
    display_df = metrics_df.copy()
    display_df["Total Return"] = display_df["Total Return"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Annualized Return"] = display_df["Annualized Return"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Volatility"] = display_df["Volatility"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Sortino Ratio"] = display_df["Sortino Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Calmar Ratio"] = display_df["Calmar Ratio"].apply(lambda x: f"{x:.3f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Win Rate"] = display_df["Win Rate"].apply(lambda x: f"{x*100:.1f}%")
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#2E75B6')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight meta-agents
    for i in range(1, len(metrics_list) - 1):  # Meta-agents
        for j in range(len(display_df.columns)):
            table[(i, j)].set_facecolor('#D6EAF8')
    
    # Highlight benchmarks
    for i in range(len(metrics_list) - 1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            table[(i, j)].set_facecolor('#FCF3CF')
    
    plt.title("Meta-Agent Performance Metrics Comparison", fontsize=14, fontweight='bold', pad=20)
    
    output_file = os.path.join(output_dir, "meta_performance_metrics_table.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return metrics_df


def plot_sharpe_ratio_comparison(meta_results, equal_weight_df, sp500_df, output_dir):
    """Plot Sharpe ratio comparison"""
    sharpe_ratios = []
    strategies = []
    
    # Meta-agents
    for name, df in meta_results.items():
        metrics = calculate_performance_metrics(df)
        sharpe_ratios.append(metrics["Sharpe Ratio"])
        strategies.append(df["strategy"].iloc[0])
    
    # Benchmarks
    eq_metrics = calculate_performance_metrics(equal_weight_df)
    sharpe_ratios.append(eq_metrics["Sharpe Ratio"])
    strategies.append("Equal Weight")
    
    sp_metrics = calculate_performance_metrics(sp500_df)
    sharpe_ratios.append(sp_metrics["Sharpe Ratio"])
    strategies.append("S&P 500")
    
    # Plot
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#000000', '#8B0000']
    bars = plt.bar(strategies, sharpe_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, sharpe_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel("Strategy", fontsize=12, fontweight='bold')
    plt.ylabel("Sharpe Ratio", fontsize=12, fontweight='bold')
    plt.title("Sharpe Ratio Comparison: Meta-Agents vs Benchmarks", fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "meta_sharpe_ratio_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_volatility_vs_return(meta_results, equal_weight_df, sp500_df, output_dir):
    """Plot risk-return scatter"""
    data_points = []
    
    # Meta-agents
    for name, df in meta_results.items():
        metrics = calculate_performance_metrics(df)
        data_points.append({
            "Strategy": df["strategy"].iloc[0],
            "Return": metrics["Annualized Return"] * 100,
            "Volatility": metrics["Volatility"] * 100,
            "Type": "Meta-Agent"
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
    plt.figure(figsize=(12, 8))
    
    for type_name, marker, size, color in [
        ("Meta-Agent", 'D', 200, '#3498db'), 
        ("Benchmark", 's', 150, '#e74c3c')
    ]:
        subset = df_plot[df_plot["Type"] == type_name]
        plt.scatter(subset["Volatility"], subset["Return"], 
                   s=size, alpha=0.7, label=type_name, marker=marker, 
                   edgecolors='black', linewidths=2, color=color)
    
    # Add labels
    for _, row in df_plot.iterrows():
        plt.annotate(row["Strategy"], 
                    (row["Volatility"], row["Return"]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.4, edgecolor='black'))
    
    plt.xlabel("Volatility (% per year)", fontsize=12, fontweight='bold')
    plt.ylabel("Annualized Return (%)", fontsize=12, fontweight='bold')
    plt.title("Risk-Return Profile: Meta-Agents vs Benchmarks", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "meta_risk_return_scatter.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_drawdown_analysis(meta_results, equal_weight_df, sp500_df, output_dir):
    """Plot drawdown comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    all_strategies = []
    
    # Meta-agents
    for name, df in meta_results.items():
        all_strategies.append((df["strategy"].iloc[0], df, "roi"))
    
    # Benchmarks
    all_strategies.append(("Equal Weight", equal_weight_df, "roi"))
    all_strategies.append(("S&P 500", sp500_df, "roi"))
    
    for idx, (strategy_name, strategy_data, roi_col) in enumerate(all_strategies):
        ax = axes[idx]
        strategy_data = strategy_data.sort_values("month")
        
        # Calculate drawdown
        returns = strategy_data[roi_col].values
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max * 100
        
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        ax.plot(drawdown, color='darkred', linewidth=2.5)
        
        ax.set_xlabel("Month", fontsize=10, fontweight='bold')
        ax.set_ylabel("Drawdown (%)", fontsize=10, fontweight='bold')
        ax.set_title(f"{strategy_name} - Drawdown Analysis", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        # Add max drawdown annotation
        max_dd = drawdown.max()
        ax.text(0.5, 0.95, f'Max DD: {max_dd:.2f}%', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black'),
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "meta_drawdown_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_portfolio_weights_over_time(meta_results, output_dir):
    """Plot how portfolio weights change over time for meta-agents"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    weight_cols = [col for col in meta_results[list(meta_results.keys())[0]].columns if 'weight_' in col]
    
    for idx, (name, df) in enumerate(meta_results.items()):
        ax = axes[idx]
        
        # Extract weights
        weights_data = df[weight_cols].copy()
        weights_data.index = df['month']
        
        # Create stacked area plot
        ax.stackplot(range(len(weights_data)), 
                    *[weights_data[col].values for col in weight_cols],
                    labels=[col.replace('weight_', '') for col in weight_cols],
                    alpha=0.8)
        
        ax.set_xlabel("Month", fontsize=11, fontweight='bold')
        ax.set_ylabel("Portfolio Weight", fontsize=11, fontweight='bold')
        ax.set_title(f"{df['strategy'].iloc[0]} - Portfolio Allocation Over Time", 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis labels
        tick_spacing = max(1, len(weights_data) // 10)
        tick_positions = range(0, len(weights_data), tick_spacing)
        tick_labels = weights_data.index[::tick_spacing]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "meta_portfolio_weights_over_time.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_rolling_sharpe_ratio(meta_results, equal_weight_df, sp500_df, output_dir, window=12):
    """Plot rolling Sharpe ratio over time"""
    plt.figure(figsize=(16, 8))
    
    def calculate_rolling_sharpe(df, window=12):
        """Calculate rolling Sharpe ratio"""
        df = df.sort_values("month")
        rolling_sharpe = []
        months = []
        
        for i in range(window, len(df)):
            returns = df["roi"].iloc[i-window:i].values
            if len(returns) >= window:
                mean_return = returns.mean() * 12
                std_return = returns.std() * np.sqrt(12)
                sharpe = mean_return / std_return if std_return > 0 else 0
                rolling_sharpe.append(sharpe)
                months.append(df["month"].iloc[i])
        
        return pd.DataFrame({"month": months, "sharpe": rolling_sharpe})
    
    # Meta-agents
    for name, df in meta_results.items():
        rolling_data = calculate_rolling_sharpe(df, window)
        plt.plot(rolling_data["month"], rolling_data["sharpe"], 
                label=df["strategy"].iloc[0], linewidth=2.5, marker='o', markersize=3)
    
    # Benchmarks
    eq_rolling = calculate_rolling_sharpe(equal_weight_df, window)
    plt.plot(eq_rolling["month"], eq_rolling["sharpe"], 
            label="Equal Weight", linewidth=2.5, linestyle="--", marker='s', markersize=3)
    
    sp_rolling = calculate_rolling_sharpe(sp500_df, window)
    plt.plot(sp_rolling["month"], sp_rolling["sharpe"], 
            label="S&P 500", linewidth=2.5, linestyle=":", marker='^', markersize=3)
    
    plt.xlabel("Month", fontsize=12, fontweight='bold')
    plt.ylabel(f"Rolling {window}-Month Sharpe Ratio", fontsize=12, fontweight='bold')
    plt.title(f"Rolling {window}-Month Sharpe Ratio: Meta-Agents vs Benchmarks", 
             fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "meta_rolling_sharpe_ratio.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("META-AGENT BACKTEST ANALYSIS AND VISUALIZATION")
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
    
    print("Loading meta-agent results...")
    meta_results = load_meta_results()
    meta_results = rebase_results_with_prices(meta_results, prices_df)
    
    print("\nGenerating visualizations...")
    
    print("1. Cumulative returns plot...")
    plot_cumulative_returns(meta_results, equal_weight_df, sp500_df, OUTPUT_DIR)
    
    print("2. Monthly returns comparison...")
    plot_monthly_returns(meta_results, equal_weight_df, sp500_df, OUTPUT_DIR)
    
    print("3. Performance metrics table...")
    metrics_df = plot_performance_metrics_table(meta_results, equal_weight_df, sp500_df, OUTPUT_DIR)
    
    print("4. Sharpe ratio comparison...")
    plot_sharpe_ratio_comparison(meta_results, equal_weight_df, sp500_df, OUTPUT_DIR)
    
    print("5. Risk-return scatter plot...")
    plot_volatility_vs_return(meta_results, equal_weight_df, sp500_df, OUTPUT_DIR)
    
    print("6. Drawdown analysis...")
    plot_drawdown_analysis(meta_results, equal_weight_df, sp500_df, OUTPUT_DIR)
    
    print("7. Portfolio weights over time...")
    plot_portfolio_weights_over_time(meta_results, OUTPUT_DIR)
    
    print("8. Rolling Sharpe ratio...")
    plot_rolling_sharpe_ratio(meta_results, equal_weight_df, sp500_df, OUTPUT_DIR)
    
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
