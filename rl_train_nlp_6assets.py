import os
import random
import time
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


# --- Configuration ---
TICKERS = ["^GSPC", "^IXIC", "^DJI", "GC=F", "SI=F", "CL=F"]
START_DATE = "2012-01-01"
END_DATE = "2024-12-31"

TRAIN_START = "2012-01-01"
TRAIN_END = "2019-12-31"
BACKTEST_START = "2020-01-01"
BACKTEST_END = "2024-11-01"

SEEDS = [1, 2, 3, 4, 5]
TOTAL_TIMESTEPS = 10000

# Calculate expected observation size: 6 assets * 7 metrics + 6*5/2 correlations = 42 + 15 = 57
EXPECTED_OBS_SIZE = 57


# --- Step 1: Download and Clean Price Data ---
def fetch_data(tickers, start, end):
    """Fetch data with rate limit handling (matching alpha.py)"""
    retry = True
    delay = 60
    data = None
    while retry:
        try:
            data = yf.download(tickers, start=start, end=end, actions=False)
            if len(tickers) > 1:
                data = data['Close']
            else:
                data = data[['Close']]
            retry = False
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit. Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Error fetching data: {e}")
                retry = False
                data = pd.DataFrame()
    return data


def clean_stock_data(df):
    """Clean stock data by applying backward fill and linear interpolation (matching alpha.py)"""
    df = df.bfill()
    df = df.interpolate(method='linear', limit_direction='both')
    return df


def process_stock_data(df):
    """Process cleaned stock data to create normalized and log evolution (matching alpha.py)"""
    normalized_df = df.copy()
    log_evolution_df = df.copy()
    
    for ticker in df.columns:
        first_valid = df[ticker].dropna().iloc[0] if df[ticker].dropna().size > 0 else None
        if first_valid is not None and first_valid > 0:
            normalized_df[ticker] = df[ticker] / first_valid
            log_evolution_df[ticker] = np.log(df[ticker] / first_valid)
        else:
            print(f"Warning: No valid first value for {ticker}")
            normalized_df[ticker] = np.nan
            log_evolution_df[ticker] = np.nan
    
    return normalized_df, log_evolution_df


# --- Step 2: Calculate Monthly Metrics and Correlations ---
def calculate_monthly_metrics(normalized_df):
    """
    Generate monthly observation and correlation data (matching alpha.py structure).
    Returns dict of {year_month: observation_vector}
    """
    observations = {}
    
    # Group by year and month
    grouped = normalized_df.groupby([normalized_df.index.year, normalized_df.index.month])
    
    for (year, month), group_data in grouped:
        year_month = f"{year}_{month:02d}"
        
        # Drop columns with all NaN
        df = group_data.dropna(axis=1, how='all')
        if df.empty:
            continue
        
        # Calculate daily log returns
        log_returns = np.log(df / df.shift(1)).dropna()
        if log_returns.empty:
            continue
        
        # Initialize metrics DataFrame
        metrics = pd.DataFrame(index=df.columns)
        
        # Month First Close
        metrics['Month_First_Close'] = df.apply(lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else np.nan)
        
        # Month Last Close
        metrics['Month_Last_Close'] = df.apply(lambda x: x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan)
        
        # Volatility
        metrics['Volatility'] = log_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        metrics['Sharpe_Ratio'] = (log_returns.mean() * 252) / metrics['Volatility']
        
        # Sortino Ratio
        downside_returns = log_returns.where(log_returns < 0, 0)
        downside_vol = downside_returns.std() * np.sqrt(252)
        metrics['Sortino_Ratio'] = (log_returns.mean() * 252) / downside_vol
        
        # Maximum Drawdown
        def calculate_mdd(series):
            cumulative = series / series.iloc[0]
            peak = cumulative.cummax()
            drawdown = (peak - cumulative) / peak
            return drawdown.max() if drawdown.size > 0 else np.nan
        
        metrics['MDD'] = df.apply(calculate_mdd)
        
        # Calmar Ratio
        monthly_return = (metrics['Month_Last_Close'] / metrics['Month_First_Close']) - 1
        annualized_return = (1 + monthly_return) ** 12 - 1
        metrics['Calmar_Ratio'] = annualized_return / metrics['MDD']
        
        # Replace infinities with NaN
        metrics = metrics.replace([np.inf, -np.inf], np.nan)
        
        # Replace NaN with 0 (matching alpha.py)
        metrics = metrics.fillna(0)
        
        # Flatten metrics into single column
        combined_data = []
        for ticker in metrics.index:
            for metric in metrics.columns:
                value = metrics.at[ticker, metric]
                combined_data.append(value)
        
        # Calculate correlation matrix
        if not log_returns.empty:
            corr_matrix = log_returns.corr()
            corr_matrix = corr_matrix.fillna(0)
            
            # Flatten correlation matrix (upper triangle)
            tickers_list = corr_matrix.index.tolist()
            for i in range(len(tickers_list)):
                for j in range(i + 1, len(tickers_list)):
                    value = corr_matrix.iloc[i, j]
                    combined_data.append(value)
        
        observations[year_month] = np.array(combined_data, dtype=np.float32)
    
    return observations


# --- Step 3: Create Organized Folder Structure (matching alpha.py) ---
def save_organized_data(clean_df, normalized_df, log_evolution_df, observations):
    """Save data in organized folder structure matching alpha.py"""
    
    # Create metrics_used/price folder
    price_dir = os.path.join("metrics_used_6assets", "price")
    os.makedirs(price_dir, exist_ok=True)
    
    clean_df.to_csv(os.path.join(price_dir, "clean_data.csv"))
    normalized_df.to_csv(os.path.join(price_dir, "normalized_data.csv"))
    log_evolution_df.to_csv(os.path.join(price_dir, "log_evolution_data.csv"))
    
    # Create observation/metrics folder
    obs_metrics_dir = os.path.join("organized_6assets", "observation", "metrics")
    os.makedirs(obs_metrics_dir, exist_ok=True)
    
    # Save observations as single-column CSVs (no header, no index)
    for year_month, obs_vector in observations.items():
        output_file = os.path.join(obs_metrics_dir, f"{year_month}_combined.csv")
        pd.DataFrame(obs_vector).to_csv(output_file, index=False, header=False)
    
    print(f"Saved {len(observations)} observation files to {obs_metrics_dir}")


# --- Step 4: Custom Portfolio Environment (matching alpha.py) ---
class CustomPortfolioEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio management (matching alpha.py structure).
    
    - Observation: 57-dimensional vector from observation/metrics/{yyyy}_{mm}_combined.csv
    - Action: 6-dimensional weight allocations for 6 assets (sum to 1)
    - Reward: 2 * ROI - 0.7 * volatility - 0.5 * MDD
    """
    def __init__(self, price_dir="metrics_used_6assets/price", metrics_dir="organized_6assets/observation/metrics"):
        super().__init__()
        
        self.tickers = TICKERS
        
        # Load daily price data
        price_file = os.path.join(price_dir, "clean_data.csv")
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price file {price_file} not found.")
        self.daily_prices = pd.read_csv(price_file, index_col="Date", parse_dates=True)
        
        # Load monthly observations
        self.metrics_dir = metrics_dir
        months = pd.date_range(start=TRAIN_START, end=TRAIN_END, freq="ME")
        self.observations = []
        self.month_files = []
        
        for month in months:
            file_path = os.path.join(metrics_dir, f"{month.year}_{month.month:02d}_combined.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=None)
                    if len(df) == EXPECTED_OBS_SIZE:
                        self.observations.append(df.iloc[:, 0].values.astype(np.float32))
                        self.month_files.append(file_path)
                    else:
                        print(f"Warning: {file_path} has {len(df)} rows, expected {EXPECTED_OBS_SIZE}. Skipping.")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        if not self.observations:
            raise ValueError("No valid observation files found.")
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(EXPECTED_OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(TICKERS),), dtype=np.float32)
        
        # Get monthly last trading days
        self.monthly_last_days = self.daily_prices.resample("ME").last().index.tolist()
        self.monthly_last_days = [d for d in self.monthly_last_days
                                if d >= pd.Timestamp(TRAIN_START) and d <= pd.Timestamp(END_DATE)]
        
        # Align observations with price data
        self.total_steps = min(len(self.observations), len(self.monthly_last_days) - 1)
        if self.total_steps < 1:
            raise ValueError("Insufficient data for training.")
        
        self.current_step = 0
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        obs = self.observations[0]
        info = {}
        return obs, info
    
    def step(self, action):
        if self.current_step >= self.total_steps:
            raise ValueError("Episode is over")
        
        # Normalize action to sum to 1
        action_sum = np.sum(action)
        if action_sum > 0:
            weights = action / action_sum
        else:
            weights = np.ones(len(TICKERS)) / len(TICKERS)
        
        # Get start and end dates for the month
        start_date = self.monthly_last_days[self.current_step]
        end_date = self.monthly_last_days[self.current_step + 1]
        
        # Get daily prices for the period
        try:
            daily_prices = self.daily_prices.loc[start_date:end_date, self.tickers]
        except KeyError as e:
            print(f"Warning: Missing price data for period {start_date} to {end_date}. Using zeros.")
            daily_prices = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns=self.tickers)
        
        # Compute portfolio values
        daily_portfolio_values = np.sum(daily_prices.values * weights, axis=1)
        
        # Compute ROI
        if daily_portfolio_values[0] > 0:
            roi = (daily_portfolio_values[-1] / daily_portfolio_values[0]) - 1
        else:
            roi = 0
        
        # Compute daily returns
        daily_returns = daily_portfolio_values[1:] / daily_portfolio_values[:-1] - 1 if len(daily_portfolio_values) > 1 else np.array([0])
        
        # Compute volatility
        volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
        
        # Compute MDD
        cummax = np.maximum.accumulate(daily_portfolio_values)
        drawdown = (cummax - daily_portfolio_values) / cummax if cummax[0] > 0 else np.zeros_like(daily_portfolio_values)
        mdd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Compute reward
        reward = 2 * roi - 0.7 * volatility - 0.5 * mdd
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.total_steps
        truncated = False
        
        # Get next observation
        obs = self.observations[self.current_step] if not done else self.observations[-1]
        info = {"roi": roi, "volatility": volatility, "mdd": mdd}
        
        return obs, reward, done, truncated, info
    
    def render(self):
        pass


# --- Step 5: Backtesting Environment (matching alpha.py) ---
class CustomPortfolioEnvBacktest(gym.Env):
    """
    Custom Gymnasium environment for portfolio management backtesting.
    
    - Observation: 57-dimensional vector
    - Action: 6-dimensional weight allocations
    - Reward: 2 * ROI - 0.7 * volatility - 0.5 * MDD
    """
    def __init__(self, price_dir="metrics_used_6assets/price", metrics_dir="organized_6assets/observation/metrics"):
        super().__init__()
        
        self.tickers = TICKERS
        
        # Load daily price data
        price_file = os.path.join(price_dir, "clean_data.csv")
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price file {price_file} not found.")
        self.daily_prices = pd.read_csv(price_file, index_col="Date", parse_dates=True)
        
        # Load monthly observations into a dictionary
        months = pd.date_range(start=TRAIN_START, end=END_DATE, freq="ME")
        self.observations = {}
        
        for month in months:
            file_path = os.path.join(metrics_dir, f"{month.year}_{month.month:02d}_combined.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=None)
                    if len(df) == EXPECTED_OBS_SIZE:
                        self.observations[month.strftime("%Y-%m")] = df.iloc[:, 0].values.astype(np.float32)
                    else:
                        print(f"Warning: {file_path} has {len(df)} rows, expected {EXPECTED_OBS_SIZE}. Skipping.")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        # Get monthly last trading days
        self.monthly_last_days = self.daily_prices.resample("ME").last().index.tolist()
        self.month_list = [d.strftime("%Y-%m") for d in self.monthly_last_days]
        
        # Define backtest period
        self.backtest_months = pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq="ME").strftime("%Y-%m").tolist()
        for month in self.backtest_months:
            if month not in self.observations:
                raise ValueError(f"Observation file for {month} not found.")
        
        self.total_steps = len(self.backtest_months) - 1
        
        # Define spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(EXPECTED_OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(TICKERS),), dtype=np.float32)
        
        self.current_step = 0
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        obs = self.observations[self.backtest_months[0]]
        info = {}
        return obs, info
    
    def step(self, action):
        if self.current_step >= self.total_steps:
            raise ValueError("Episode is over")
        
        # Normalize action
        action_sum = np.sum(action)
        if action_sum > 0:
            weights = action / action_sum
        else:
            weights = np.ones(len(TICKERS)) / len(TICKERS)
        
        # Get allocation month
        allocation_month = self.backtest_months[self.current_step + 1]
        
        # Get start and end dates
        start_date = pd.to_datetime(allocation_month + "-01")
        end_date = start_date + pd.offsets.MonthEnd(0)
        
        # Get daily prices
        try:
            daily_prices = self.daily_prices.loc[start_date:end_date, self.tickers]
        except KeyError:
            print(f"Warning: Missing price data for {start_date} to {end_date}. Using zeros.")
            daily_prices = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns=self.tickers)
        
        # Compute portfolio values
        daily_portfolio_values = np.sum(daily_prices.values * weights, axis=1)
        
        # Compute ROI
        if daily_portfolio_values[0] > 0:
            roi = (daily_portfolio_values[-1] / daily_portfolio_values[0]) - 1
        else:
            roi = 0
        
        # Compute daily returns
        daily_returns = daily_portfolio_values[1:] / daily_portfolio_values[:-1] - 1 if len(daily_portfolio_values) > 1 else np.array([0])
        
        # Compute volatility
        volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
        
        # Compute MDD
        cummax = np.maximum.accumulate(daily_portfolio_values)
        drawdown = (cummax - daily_portfolio_values) / cummax if cummax[0] > 0 else np.zeros_like(daily_portfolio_values)
        mdd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Compute reward
        reward = 2 * roi - 0.7 * volatility - 0.5 * mdd
        
        # Advance step
        self.current_step += 1
        
        # Check done
        done = self.current_step >= self.total_steps
        truncated = False
        
        # Get next observation
        if not done:
            obs = self.observations[self.backtest_months[self.current_step]]
        else:
            obs = np.zeros(EXPECTED_OBS_SIZE)
        
        # Info
        info = {
            "allocation_month": allocation_month,
            "roi": roi,
            "volatility": volatility,
            "mdd": mdd
        }
        
        return obs, reward, done, truncated, info
    
    def render(self):
        pass


# --- Step 6: Training Function (matching alpha.py) ---
def train_rl_agents(price_dir="metrics_used_6assets/price", 
                   metrics_dir="organized_6assets/observation/metrics",
                   output_dir="results_6assets", 
                   seeds=SEEDS, 
                   total_timesteps=TOTAL_TIMESTEPS):
    """
    Train PPO, SAC, DDPG, TD3 models with specified seeds (matching alpha.py).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created/Verified output directory: {output_dir}")
    
    evaluation_results = []
    
    models = {
        "ppo": PPO,
        "sac": SAC,
        "ddpg": DDPG,
        "td3": TD3
    }
    
    def make_env():
        return CustomPortfolioEnv(price_dir=price_dir, metrics_dir=metrics_dir)
    
    for seed in seeds:
        print(f"\nTraining with seed {seed}")
        
        np.random.seed(seed)
        random.seed(seed)
        
        env = make_vec_env(make_env, n_envs=1, seed=seed)
        
        for model_name, model_class in models.items():
            print(f"Training {model_name.upper()}...")
            
            try:
                model = model_class(
                    policy="MlpPolicy",
                    env=env,
                    verbose=0,
                    seed=seed
                )
                
                model.learn(total_timesteps=total_timesteps, progress_bar=True)
                
                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
                
                model_path = os.path.join(output_dir, f"{model_name}_seed_{seed}.zip")
                model.save(model_path)
                print(f"Saved model: {model_path}")
                
                evaluation_results.append({
                    "model": model_name,
                    "seed": seed,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward
                })
                
            except Exception as e:
                print(f"Error training {model_name} with seed {seed}: {e}")
        
        env.reset()
    
    eval_df = pd.DataFrame(evaluation_results)
    eval_path = os.path.join(output_dir, "evaluation.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved evaluation results: {eval_path}")


# --- Step 7: Backtesting Function (matching alpha.py) ---
def backtest_model(model, env, output_file):
    """Backtest a trained RL model (matching alpha.py)"""
    obs, _ = env.reset()
    done = False
    records = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        if np.sum(action) > 0:
            weights = action / np.sum(action)
        else:
            weights = np.ones(len(TICKERS)) / len(TICKERS)
        
        record = {"month": env.backtest_months[env.current_step + 1]}
        for i, ticker in enumerate(TICKERS):
            record[f"weight_{ticker}"] = weights[i]
        
        obs, reward, done, truncated, info = env.step(weights)
        
        record.update({
            "roi": info["roi"],
            "volatility": info["volatility"],
            "mdd": info["mdd"],
            "reward": reward
        })
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Saved backtest results to {output_file}")


def run_backtests(price_dir="metrics_used_6assets/price",
                 metrics_dir="organized_6assets/observation/metrics",
                 results_dir="results_6assets",
                 output_dir="backtest_results_6assets",
                 seeds=SEEDS):
    """Run backtests for all models and seeds (matching alpha.py)"""
    model_classes = {
        "ppo": PPO,
        "sac": SAC,
        "ddpg": DDPG,
        "td3": TD3
    }
    
    for model_type in model_classes:
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        for seed in seeds:
            model_path = os.path.join(results_dir, f"{model_type}_seed_{seed}.zip")
            if not os.path.exists(model_path):
                print(f"Model {model_path} not found. Skipping.")
                continue
            
            try:
                model = model_classes[model_type].load(model_path)
                env = CustomPortfolioEnvBacktest(
                    price_dir=price_dir,
                    metrics_dir=metrics_dir
                )
                output_file = os.path.join(model_dir, f"seed_{seed}.csv")
                backtest_model(model, env, output_file)
            except Exception as e:
                print(f"Error backtesting {model_type} seed {seed}: {e}")


# --- Step 8: NLP Environment (matching alpha.py NLP section) ---
class CustomPortfolioEnvNLP(gym.Env):
    """
    Custom Gymnasium environment for portfolio optimization with NLP vectors (matching alpha.py).
    Uses NLP sentiment observations from observation/metrics
    """
    def __init__(self, price_dir="metrics_used_6assets/price", nlp_metrics_dir="observation/metrics"):
        super().__init__()
        self.tickers = TICKERS
        
        # Load daily price data
        price_file = os.path.join(price_dir, "clean_data.csv")
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price file {price_file} not found.")
        self.daily_prices = pd.read_csv(price_file, index_col="Date", parse_dates=True)
        
        # Load NLP observation vectors (6 sentiment scores for 6 assets)
        months = pd.date_range(start=TRAIN_START, end=TRAIN_END, freq="ME")
        self.observations = []
        self.obs_months = []
        
        for month in months:
            file_path = os.path.join(nlp_metrics_dir, f"{month.year}_{month.month:02d}_combined.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=None)
                    if len(df) == len(TICKERS):  # Expecting 6 sentiment scores
                        self.observations.append(df.iloc[:, 0].values.astype(np.float32))
                        self.obs_months.append(month.strftime("%Y-%m"))
                    else:
                        print(f"Warning: {file_path} has {len(df)} rows, expected {len(TICKERS)}. Skipping.")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        if not self.observations:
            raise ValueError("No valid NLP observation files found.")
        
        self.total_steps = len(self.observations) - 1
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(TICKERS),), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(TICKERS),), dtype=np.float32)
        
        self.current_step = 0
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        obs = self.observations[0]
        return obs, {}
    
    def step(self, action):
        if self.current_step >= self.total_steps:
            raise ValueError("Episode has ended.")
        
        # Normalize weights to sum to 1
        action_sum = np.sum(action)
        weights = action / action_sum if action_sum > 0 else np.ones(len(TICKERS)) / len(TICKERS)
        
        # Determine allocation month (next month)
        allocation_month = self.obs_months[self.current_step + 1]
        start_date = pd.to_datetime(allocation_month + "-01")
        end_date = start_date + pd.offsets.MonthEnd(0)
        
        # Extract daily prices for the allocation month
        try:
            daily_prices = self.daily_prices.loc[start_date:end_date, self.tickers]
        except KeyError:
            print(f"Warning: Missing price data for {start_date} to {end_date}. Using zeros.")
            daily_prices = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns=self.tickers)
        
        # Compute portfolio performance
        daily_values = np.sum(daily_prices.values * weights, axis=1)
        
        # Calculate ROI
        roi = (daily_values[-1] / daily_values[0] - 1) if daily_values[0] > 0 else 0
        
        # Calculate daily returns and volatility
        daily_returns = daily_values[1:] / daily_values[:-1] - 1 if len(daily_values) > 1 else np.array([0])
        volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
        
        # Calculate Maximum Drawdown
        cummax = np.maximum.accumulate(daily_values)
        drawdown = (cummax - daily_values) / cummax if cummax[0] > 0 else np.zeros_like(daily_values)
        mdd = np.max(drawdown) if len(daily_values) > 0 else 0
        
        # Compute reward
        reward = 2 * roi - 0.7 * volatility - 0.5 * mdd
        
        # Advance step
        self.current_step += 1
        done = self.current_step >= self.total_steps
        truncated = False
        
        # Next observation
        next_obs = self.observations[self.current_step] if not done else np.zeros(len(TICKERS))
        
        info = {"allocation_month": allocation_month, "roi": roi, "volatility": volatility, "mdd": mdd}
        return next_obs, reward, done, truncated, info


class CustomPortfolioEnvNLPBacktest(gym.Env):
    """Backtesting environment for NLP agents (matching alpha.py)"""
    def __init__(self, price_dir="metrics_used_6assets/price", nlp_metrics_dir="observation/metrics"):
        super().__init__()
        self.tickers = TICKERS
        
        # Load daily price data
        price_file = os.path.join(price_dir, "clean_data.csv")
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price file {price_file} not found.")
        self.daily_prices = pd.read_csv(price_file, index_col="Date", parse_dates=True)
        
        # Load NLP observations into a dictionary
        months = pd.date_range(start=TRAIN_START, end=END_DATE, freq="ME")
        self.observations = {}
        
        for month in months:
            file_path = os.path.join(nlp_metrics_dir, f"{month.year}_{month.month:02d}_combined.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=None)
                    if len(df) == len(TICKERS):
                        self.observations[month.strftime("%Y-%m")] = df.iloc[:, 0].values.astype(np.float32)
                    else:
                        print(f"Warning: {file_path} has {len(df)} rows, expected {len(TICKERS)}. Skipping.")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        # Define backtest period
        self.backtest_months = pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq="ME").strftime("%Y-%m").tolist()
        for month in self.backtest_months:
            if month not in self.observations:
                print(f"Warning: Observation file for {month} not found. Using zeros.")
                self.observations[month] = np.zeros(len(TICKERS), dtype=np.float32)
        
        self.total_steps = len(self.backtest_months) - 1
        
        # Define spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(TICKERS),), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(TICKERS),), dtype=np.float32)
        
        self.current_step = 0
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        obs = self.observations[self.backtest_months[0]]
        return obs, {}
    
    def step(self, action):
        if self.current_step >= self.total_steps:
            raise ValueError("Episode has ended.")
        
        # Normalize weights
        action_sum = np.sum(action)
        weights = action / action_sum if action_sum > 0 else np.ones(len(TICKERS)) / len(TICKERS)
        
        # Get allocation month
        allocation_month = self.backtest_months[self.current_step + 1]
        start_date = pd.to_datetime(allocation_month + "-01")
        end_date = start_date + pd.offsets.MonthEnd(0)
        
        # Get daily prices
        try:
            daily_prices = self.daily_prices.loc[start_date:end_date, self.tickers]
        except KeyError:
            print(f"Warning: Missing price data for {start_date} to {end_date}. Using zeros.")
            daily_prices = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns=self.tickers)
        
        # Compute portfolio performance
        daily_values = np.sum(daily_prices.values * weights, axis=1)
        roi = (daily_values[-1] / daily_values[0] - 1) if daily_values[0] > 0 else 0
        daily_returns = daily_values[1:] / daily_values[:-1] - 1 if len(daily_values) > 1 else np.array([0])
        volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
        cummax = np.maximum.accumulate(daily_values)
        drawdown = (cummax - daily_values) / cummax if cummax[0] > 0 else np.zeros_like(daily_values)
        mdd = np.max(drawdown) if len(daily_values) > 0 else 0
        reward = 2 * roi - 0.7 * volatility - 0.5 * mdd
        
        # Advance
        self.current_step += 1
        done = self.current_step >= self.total_steps
        truncated = False
        next_obs = self.observations[self.backtest_months[self.current_step]] if not done else np.zeros(len(TICKERS))
        
        info = {
            "allocation_month": allocation_month,
            "roi": roi,
            "volatility": volatility,
            "mdd": mdd
        }
        
        return next_obs, reward, done, truncated, info


# --- Step 9: NLP Training and Backtesting (matching alpha.py) ---
def train_nlp_agents(price_dir="metrics_used_6assets/price",
                    nlp_metrics_dir="observation/metrics",
                    output_dir="results_nlp_6assets",
                    seeds=SEEDS,
                    total_timesteps=TOTAL_TIMESTEPS):
    """Train NLP-based RL agents (matching alpha.py NLP section)"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created/Verified output directory: {output_dir}")
    
    evaluation_results = []
    
    models = {
        "ppo": PPO,
        "sac": SAC,
        "ddpg": DDPG,
        "td3": TD3
    }
    
    def make_env():
        return CustomPortfolioEnvNLP(price_dir=price_dir, nlp_metrics_dir=nlp_metrics_dir)
    
    for seed in seeds:
        print(f"\nTraining NLP agents with seed {seed}")
        
        np.random.seed(seed)
        random.seed(seed)
        
        env = make_vec_env(make_env, n_envs=1, seed=seed)
        
        for model_name, model_class in models.items():
            print(f"Training NLP {model_name.upper()}...")
            
            try:
                model = model_class(
                    policy="MlpPolicy",
                    env=env,
                    verbose=0,
                    seed=seed
                )
                
                model.learn(total_timesteps=total_timesteps, progress_bar=True)
                
                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
                
                model_path = os.path.join(output_dir, f"{model_name}_seed_{seed}.zip")
                model.save(model_path)
                print(f"Saved NLP model: {model_path}")
                
                evaluation_results.append({
                    "model": model_name,
                    "seed": seed,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward
                })
                
            except Exception as e:
                print(f"Error training NLP {model_name} with seed {seed}: {e}")
        
        env.reset()
    
    eval_df = pd.DataFrame(evaluation_results)
    eval_path = os.path.join(output_dir, "evaluation.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved NLP evaluation results: {eval_path}")


def run_nlp_backtests(price_dir="metrics_used_6assets/price",
                     nlp_metrics_dir="observation/metrics",
                     results_dir="results_nlp_6assets",
                     output_dir="backtest_results_nlp_6assets",
                     seeds=SEEDS):
    """Run backtests for NLP agents (matching alpha.py)"""
    model_classes = {
        "ppo": PPO,
        "sac": SAC,
        "ddpg": DDPG,
        "td3": TD3
    }
    
    for model_type in model_classes:
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        for seed in seeds:
            model_path = os.path.join(results_dir, f"{model_type}_seed_{seed}.zip")
            if not os.path.exists(model_path):
                print(f"NLP Model {model_path} not found. Skipping.")
                continue
            
            try:
                model = model_classes[model_type].load(model_path)
                env = CustomPortfolioEnvNLPBacktest(
                    price_dir=price_dir,
                    nlp_metrics_dir=nlp_metrics_dir
                )
                output_file = os.path.join(model_dir, f"seed_{seed}.csv")
                backtest_model(model, env, output_file)
            except Exception as e:
                print(f"Error backtesting NLP {model_type} seed {seed}: {e}")


# --- Main Pipeline ---
def main():
    """Main pipeline matching alpha.py structure (both Metrics and NLP)"""
    print("=" * 60)
    print("STEP 1: Downloading price data...")
    print("=" * 60)
    data = fetch_data(TICKERS, START_DATE, END_DATE)
    
    if data.empty:
        raise ValueError("No data downloaded. Check tickers or date range.")
    
    print("\nSTEP 2: Cleaning price data...")
    clean_df = clean_stock_data(data)
    
    print("\nSTEP 3: Processing stock data (normalize & log evolution)...")
    normalized_df, log_evolution_df = process_stock_data(clean_df)
    
    print("\nSTEP 4: Calculating monthly metrics and correlations...")
    observations = calculate_monthly_metrics(normalized_df)
    print(f"Generated {len(observations)} monthly observations")
    
    print("\nSTEP 5: Saving organized data...")
    save_organized_data(clean_df, normalized_df, log_evolution_df, observations)
    
    """print("\n" + "=" * 60)
    print("STEP 6: Training METRICS RL agents...")
    print("=" * 60)
    train_rl_agents()
    
    print("\n" + "=" * 60)
    print("STEP 7: Running METRICS backtests...")
    print("=" * 60)
    run_backtests()"""
    
    print("\n" + "=" * 60)
    print("STEP 8: Training NLP RL agents...")
    print("=" * 60)
    train_nlp_agents()
    
    print("\n" + "=" * 60)
    print("STEP 9: Running NLP backtests...")
    print("=" * 60)
    run_nlp_backtests()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Both METRICS and NLP agents trained and backtested!")
    print("=" * 60)


if __name__ == "__main__":
    main()
