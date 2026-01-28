import os
import shutil
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# --- Configuration ---
TICKERS = ["^GSPC", "^IXIC", "^DJI", "GC=F", "SI=F", "CL=F"]

TRAIN_START = "2012-01-01"
TRAIN_END = "2021-12-31"

BACKTEST_START = "2022-01-01"
BACKTEST_END = "2024-11-01"

DATA_START = "2012-01-01"
DATA_END = "2024-12-31"


def create_super_folder():
    try:
        super_dir = "SUPER_6assets"
        os.makedirs(super_dir, exist_ok=True)
        print(f"Created/Verified folder: {super_dir}")
        print("SUPER_6assets folder created successfully.")

    except Exception as e:
        print(f"Error creating SUPER_6assets folder: {e}")


def copy_backtests_to_super():
    try:
        # Define source and destination directories
        nlp_source_dir = "Meta_6assets/NLP_obs"
        metrics_source_dir = "Meta_6assets/Metrics_obs"
        dest_dir = "SUPER_6assets"

        # Validate source directories
        if not os.path.exists(nlp_source_dir):
            print(f"Source directory {nlp_source_dir} does not exist.")
            return
        if not os.path.exists(metrics_source_dir):
            print(f"Source directory {metrics_source_dir} does not exist.")
            return

        # Validate destination directory
        if not os.path.exists(dest_dir):
            print(f"Destination directory {dest_dir} does not exist. Please create it first.")
            return

        # Copy backtest_meta_seed_1.csv from NLP_obs
        nlp_backtest = "backtest_meta_seed_1.csv"
        if os.path.exists(os.path.join(nlp_source_dir, nlp_backtest)):
            src_path = os.path.join(nlp_source_dir, nlp_backtest)
            dest_filename = f"nlp_{nlp_backtest}"
            dest_path = os.path.join(dest_dir, dest_filename)
            try:
                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
            except Exception as e:
                print(f"Error copying {src_path} to {dest_path}: {e}")
        else:
            print(f"NLP backtest file not found: {os.path.join(nlp_source_dir, nlp_backtest)}")

        # Copy backtest_meta_seed_1.csv from Metrics_obs
        metrics_backtest = "backtest_meta_seed_1.csv"
        if os.path.exists(os.path.join(metrics_source_dir, metrics_backtest)):
            src_path = os.path.join(metrics_source_dir, metrics_backtest)
            dest_filename = f"metrics_{metrics_backtest}"
            dest_path = os.path.join(dest_dir, dest_filename)
            try:
                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
            except Exception as e:
                print(f"Error copying {src_path} to {dest_path}: {e}")
        else:
            print(f"Metrics backtest file not found: {os.path.join(metrics_source_dir, metrics_backtest)}")

        print("Copy process completed successfully.")

    except Exception as e:
        print(f"Error during copy process: {e}")


def concatenate_super_backtests(super_dir="SUPER_6assets",
                                output_file="SUPER_6assets/super_weights_clean.csv"):
    try:
        # Define the input files
        nlp_file = os.path.join(super_dir, "nlp_backtest_meta_seed_1.csv")
        metrics_file = os.path.join(super_dir, "metrics_backtest_meta_seed_1.csv")

        # Validate input files
        if not os.path.exists(nlp_file):
            raise FileNotFoundError(f"NLP backtest file {nlp_file} not found.")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics backtest file {metrics_file} not found.")

        # Read the CSV files
        nlp_df = pd.read_csv(nlp_file)
        metrics_df = pd.read_csv(metrics_file)

        print(f"NLP backtest has {len(nlp_df)} rows ({nlp_df['month'].min()} to {nlp_df['month'].max()})")
        print(
            f"Metrics backtest has {len(metrics_df)} rows ({metrics_df['month'].min()} to {metrics_df['month'].max()})")

        # Keep only 'month' and weight columns
        weight_cols = [col for col in nlp_df.columns if col.startswith("weight_")]
        nlp_df = nlp_df[['month'] + weight_cols]
        metrics_df = metrics_df[['month'] + weight_cols]

        # Rename weight columns to include source identifier
        nlp_df = nlp_df.rename(columns={col: f"{col}_nlp" for col in weight_cols})
        metrics_df = metrics_df.rename(columns={col: f"{col}_metrics" for col in weight_cols})

        # Drop 'month' from metrics_df to avoid duplication
        metrics_df = metrics_df.drop(columns=['month'])

        # Concatenate side by side
        concatenated_df = pd.concat([nlp_df, metrics_df], axis=1)

        # Verify the number of columns (should be 1 + (6 weights × 2 sources) = 13)
        expected_columns = 1 + (6 * 2)
        if len(concatenated_df.columns) != expected_columns:
            print(
                f"Warning: Concatenated DataFrame has {len(concatenated_df.columns)} columns, expected {expected_columns}.")

        # Save the concatenated DataFrame
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        concatenated_df.to_csv(output_file, index=False)
        print(
            f"Concatenated and cleaned CSV saved to {output_file} with {len(concatenated_df)} rows and {len(concatenated_df.columns)} columns.")

        print("Concatenation process completed successfully.")

    except Exception as e:
        print(f"Error during concatenation process: {e}")


class CustomSuperMetaEnv(gym.Env):

    def __init__(self, csv_path="SUPER_6assets/super_weights_clean.csv",
                 price_path="metrics_used_6assets/price/clean_data.csv",
                 train_mode=True):
        super().__init__()

        # Load observation data (weights from both meta-agents)
        df = pd.read_csv(csv_path)

        # Filter to training period only if train_mode=True
        if train_mode:
            # Training: use full 2012-2024 period
            train_start_month = pd.to_datetime(TRAIN_START).strftime("%Y-%m")
            train_end_month = pd.to_datetime(TRAIN_END).strftime("%Y-%m")
            df = df[(df['month'] >= train_start_month) & (df['month'] <= train_end_month)].copy()
            print(f"Super Training: Using {len(df)} months ({train_start_month} to {train_end_month})")

            if len(df) == 0:
                raise ValueError(f"No training data found for period <= {train_end_month}")
        else:
            # Backtesting: use only 2020-2024 data
            backtest_start_month = pd.to_datetime(BACKTEST_START).strftime("%Y-%m")
            backtest_end_month = pd.to_datetime(BACKTEST_END).strftime("%Y-%m")
            df = df[(df['month'] >= backtest_start_month) & (df['month'] <= backtest_end_month)].copy()
            print(f"Super Backtest: Using {len(df)} months ({backtest_start_month} to {backtest_end_month})")

            if len(df) == 0:
                raise ValueError(f"No backtest data found for period {backtest_start_month} to {backtest_end_month}")

        self.months = df['month'].tolist()
        # Observations: 12 weight columns (6 weights from NLP + 6 weights from Metrics)
        self.observations = df.iloc[:, 1:].astype(np.float32).values

        # Load price data
        self.price_data = pd.read_csv(price_path, index_col="Date", parse_dates=True)
        self.tickers = TICKERS

        # Validate tickers
        for ticker in self.tickers:
            if ticker not in self.price_data.columns:
                raise ValueError(f"Ticker {ticker} not in price data")

        self.total_steps = len(self.months) - 1  # -1 because we need next month for returns

        # Define observation and action spaces
        obs_dim = self.observations.shape[1]  # Should be 12 (6 weights × 2 sources)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Define action space with finite bounds (6 assets)
        self.action_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

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

        # Normalize action to sum to 1 using softmax
        weights = np.exp(action) / np.sum(np.exp(action))

        # Get current month
        month = self.months[self.current_step]

        # Determine start and end dates
        start_date = pd.to_datetime(month + '-01')
        end_date = start_date + pd.offsets.MonthEnd(0)

        # Extract daily prices
        try:
            daily_prices = self.price_data.loc[start_date:end_date, self.tickers]
        except KeyError:
            print(f"Warning: Missing price data for {start_date} to {end_date}. Using zeros.")
            daily_prices = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns=self.tickers)

        # Compute daily portfolio values
        daily_values = (daily_prices * weights).sum(axis=1)

        # Calculate ROI
        roi = (daily_values.iloc[-1] / daily_values.iloc[0] - 1) if daily_values.iloc[0] > 0 else 0

        # Calculate daily returns and volatility
        daily_returns = daily_values.pct_change().dropna()
        volatility = daily_returns.std() if len(daily_returns) > 0 else 0

        # Calculate Maximum Drawdown (MDD)
        cummax = daily_values.cummax()
        drawdown = (cummax - daily_values) / cummax
        mdd = drawdown.max() if len(drawdown) > 0 else 0

        # Compute reward
        reward = 2 * roi - 0.7 * volatility - 0.5 * mdd

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.total_steps
        truncated = False

        # Next observation
        next_obs = self.observations[self.current_step] if not terminated else np.zeros(self.observations.shape[1])

        info = {"allocation_month": month, "roi": roi, "volatility": volatility, "mdd": mdd}
        return next_obs, reward, terminated, truncated, info

    def render(self):
        pass


def train_super_meta_agent():
    print("Training Super Meta-Agent...")
    # IMPORTANT: train_mode=True uses full 2012-2024 data
    env = make_vec_env(lambda: CustomSuperMetaEnv(train_mode=True), n_envs=1, seed=1)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={'net_arch': [256, 256, 256]},
        verbose=1,
        seed=1
    )
    # due to simpler observation space and pre-trained meta-agent knowledge
    model.learn(total_timesteps=1000, progress_bar=True)
    os.makedirs("super_results_6assets", exist_ok=True)
    model_path = "super_results_6assets/super_meta_agent_seed1"
    model.save(model_path)
    print(f"Super meta-agent trained and saved to {model_path}.zip")
    return model


def backtest_super_meta_agent(model, env, output_file="SUPER_6assets/backtest_super_meta_seed_1.csv"):
    print("Backtesting Super Meta-Agent...")
    obs, _ = env.reset()
    done = False
    records = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Normalize action to sum to 1 using softmax
        weights = np.exp(action) / np.sum(np.exp(action))
        record = {"month": env.months[env.current_step]}
        for i, ticker in enumerate(TICKERS):
            record[f"weight_{ticker}"] = weights[i]
        obs, reward, done, _, info = env.step(action)
        record.update({
            "roi": info["roi"],
            "volatility": info["volatility"],
            "mdd": info["mdd"],
            "reward": reward
        })
        records.append(record)

    # Save backtest results (env already filtered to backtest period)
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Backtest results saved to {output_file} with {len(df)} rows")


def main():
    print("=" * 80)
    print("SUPER META-AGENT TRAINING PIPELINE (6 ASSETS, 2012-2024)")
    print("=" * 80)
    print("\nThis agent learns from the decisions of both NLP and Metrics meta-agents")
    print("to make final portfolio allocation decisions.")
    print("\nTraining Range:")
    print("  - Super Training: 2012-2024 (from meta agent backtests)")
    print("  - Super Backtest: 2012-2024 (same period)")
    print("=" * 80)
    print()

    print("=" * 80)
    print("STEP 1: Creating SUPER folder structure")
    print("=" * 80)
    create_super_folder()

    print("\n" + "=" * 80)
    print("STEP 2: Copying meta-agent backtests to SUPER folder")
    print("=" * 80)
    copy_backtests_to_super()

    print("\n" + "=" * 80)
    print("STEP 3: Concatenating meta-agent backtests")
    print("=" * 80)
    concatenate_super_backtests()

    print("\n" + "=" * 80)
    print("STEP 4: Training Super Meta-Agent")
    print("=" * 80)
    model = train_super_meta_agent()

    print("\n" + "=" * 80)
    print("STEP 5: Backtesting Super Meta-Agent")
    print("=" * 80)
    # IMPORTANT: train_mode=False to use full dataset for backtesting
    env = CustomSuperMetaEnv(train_mode=False)
    backtest_super_meta_agent(model, env)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - SUPER_6assets/")
    print("  - super_results_6assets/")
    print("\nThe Super Meta-Agent combines:")
    print("  1. NLP Meta-Agent decisions (based on sentiment)")
    print("  2. Metrics Meta-Agent decisions (based on price metrics)")
    print("  → Final optimized portfolio allocation")
    print("=" * 80)


if __name__ == "__main__":
    main()
