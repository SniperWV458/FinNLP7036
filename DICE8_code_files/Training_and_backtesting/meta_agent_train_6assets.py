import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# --- Configuration ---
TICKERS = ["^GSPC", "^IXIC", "^DJI", "GC=F", "SI=F", "CL=F"]
SEEDS = [1, 2, 3, 4, 5]

METRICS_TRAIN_START = "2003-01-01"
NLP_TRAIN_START = "2012-01-01"
TRAIN_END = "2021-11-01"

BACKTEST_START = "2012-01-01"
BACKTEST_END = "2024-11-01"


def create_meta_folders():
    try:
        # Define the folder paths
        meta_dir = "Meta_6assets"
        nlp_obs_dir = os.path.join(meta_dir, "NLP_obs")
        metrics_obs_dir = os.path.join(meta_dir, "Metrics_obs")

        # Create the Meta folder
        os.makedirs(meta_dir, exist_ok=True)
        print(f"Created/Verified folder: {meta_dir}")

        # Create the NLP_obs subfolder
        os.makedirs(nlp_obs_dir, exist_ok=True)
        print(f"Created/Verified folder: {nlp_obs_dir}")

        # Create the Metrics_obs subfolder
        os.makedirs(metrics_obs_dir, exist_ok=True)
        print(f"Created/Verified folder: {metrics_obs_dir}")

        print("Folder structure created successfully.")

    except Exception as e:
        print(f"Error creating folders: {e}")


def merge_backtest_results_nlp(backtest_dir="backtest_results_nlp_6assets",
                               output_dir="Meta_6assets/NLP_obs",
                               seeds=SEEDS):
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/Verified output directory: {output_dir}")

        agents = ["ppo", "sac", "ddpg", "td3"]

        for agent in agents:
            agent_dir = os.path.join(backtest_dir, agent)
            if not os.path.exists(agent_dir):
                print(f"Agent directory {agent_dir} not found. Skipping.")
                continue

            merged_dfs = []

            for seed in seeds:
                csv_file = os.path.join(agent_dir, f"seed_{seed}.csv")
                if not os.path.exists(csv_file):
                    print(f"CSV file {csv_file} not found. Skipping seed {seed} for agent {agent}.")
                    continue

                try:
                    df = pd.read_csv(csv_file)

                    # Expected number of rows for 2020-2024 backtest period (~57-59 months)
                    if len(df) < 50:
                        print(f"Warning: {csv_file} has only {len(df)} rows. Continuing anyway.")

                    # Rename columns to include seed identifier
                    if seed == seeds[0]:
                        renamed_columns = {'month': 'month'}
                        for col in df.columns[1:]:
                            renamed_columns[col] = f"{col}_seed_{seed}"
                    else:
                        renamed_columns = {col: f"{col}_seed_{seed}" for col in df.columns if col != 'month'}
                        df = df.drop(columns=['month'])

                    df = df.rename(columns=renamed_columns)
                    merged_dfs.append(df)

                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    continue

            if not merged_dfs:
                print(f"No valid CSV files found for agent {agent}. Skipping.")
                continue

            # Merge DataFrames side by side
            merged_df = merged_dfs[0]
            for df in merged_dfs[1:]:
                merged_df = pd.concat([merged_df, df], axis=1)

            # Save the merged DataFrame
            output_file = os.path.join(output_dir, f"{agent}_merged.csv")
            merged_df.to_csv(output_file, index=False)
            print(
                f"Merged CSV for agent {agent} saved to {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns.")

        print("NLP merging process completed successfully.")

    except Exception as e:
        print(f"Error during NLP merging process: {e}")


def merge_nlp_obs_results(input_dir="Meta_6assets/NLP_obs", output_dir="Meta_6assets/NLP_obs"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/Verified output directory: {output_dir}")

        agents = ["ppo", "sac", "ddpg", "td3"]
        merged_dfs = []

        for agent in agents:
            csv_file = os.path.join(input_dir, f"{agent}_merged.csv")
            if not os.path.exists(csv_file):
                print(f"CSV file {csv_file} not found. Skipping agent {agent}.")
                continue

            try:
                df = pd.read_csv(csv_file)

                # Rename columns to include agent identifier
                if agent == agents[0]:
                    renamed_columns = {'month': 'month'}
                    for col in df.columns[1:]:
                        renamed_columns[col] = f"{col}_{agent}"
                else:
                    renamed_columns = {col: f"{col}_{agent}" for col in df.columns if col != 'month'}
                    df = df.drop(columns=['month'])

                df = df.rename(columns=renamed_columns)
                merged_dfs.append(df)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        if not merged_dfs:
            print("No valid CSV files found to merge.")
            return

        # Merge DataFrames side by side
        merged_df = merged_dfs[0]
        for df in merged_dfs[1:]:
            merged_df = pd.concat([merged_df, df], axis=1)

        # Save the final merged DataFrame
        output_file = os.path.join(output_dir, "nlp_obs_unclean.csv")
        merged_df.to_csv(output_file, index=False)
        print(
            f"Final merged CSV saved to {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns.")

        print("NLP obs merging process completed successfully.")

    except Exception as e:
        print(f"Error during NLP obs merging process: {e}")


def clean_nlp_obs_results(input_file="Meta_6assets/NLP_obs/nlp_obs_unclean.csv",
                          output_file="Meta_6assets/NLP_obs/nlp_obs_clean.csv"):
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found.")

        df = pd.read_csv(input_file)

        # Identify columns to keep: 'month' and all columns containing 'weight'
        columns_to_keep = ['month']
        for col in df.columns:
            if 'weight' in col:
                columns_to_keep.append(col)

        # Create the cleaned DataFrame
        cleaned_df = df[columns_to_keep]

        # Expected: 1 month + (6 weights × 5 seeds × 4 agents) = 121 columns
        expected_cleaned_columns = 1 + (6 * 5 * 4)
        if len(cleaned_df.columns) != expected_cleaned_columns:
            print(
                f"Warning: Cleaned DataFrame has {len(cleaned_df.columns)} columns, expected {expected_cleaned_columns}.")

        # Save the cleaned DataFrame
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned CSV saved to {output_file} with {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns.")

        print("NLP obs cleaning process completed successfully.")

    except Exception as e:
        print(f"Error during NLP obs cleaning process: {e}")


def merge_backtest_results_metrics(backtest_dir="backtest_results_6assets",
                                   output_dir="Meta_6assets/Metrics_obs",
                                   seeds=SEEDS):
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/Verified output directory: {output_dir}")

        agents = ["ppo", "sac", "ddpg", "td3"]

        for agent in agents:
            agent_dir = os.path.join(backtest_dir, agent)
            if not os.path.exists(agent_dir):
                print(f"Agent directory {agent_dir} not found. Skipping.")
                continue

            merged_dfs = []

            for seed in seeds:
                csv_file = os.path.join(agent_dir, f"seed_{seed}.csv")
                if not os.path.exists(csv_file):
                    print(f"CSV file {csv_file} not found. Skipping seed {seed} for agent {agent}.")
                    continue

                try:
                    df = pd.read_csv(csv_file)

                    # Expected number of rows for 2020-2024 backtest period
                    if len(df) < 50:
                        print(f"Warning: {csv_file} has only {len(df)} rows. Continuing anyway.")

                    # Rename columns to include seed identifier
                    if seed == seeds[0]:
                        renamed_columns = {'month': 'month'}
                        for col in df.columns[1:]:
                            renamed_columns[col] = f"{col}_seed_{seed}"
                    else:
                        renamed_columns = {col: f"{col}_seed_{seed}" for col in df.columns if col != 'month'}
                        df = df.drop(columns=['month'])

                    df = df.rename(columns=renamed_columns)
                    merged_dfs.append(df)

                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    continue

            if not merged_dfs:
                print(f"No valid CSV files found for agent {agent}. Skipping.")
                continue

            # Merge DataFrames side by side
            merged_df = merged_dfs[0]
            for df in merged_dfs[1:]:
                merged_df = pd.concat([merged_df, df], axis=1)

            # Save the merged DataFrame
            output_file = os.path.join(output_dir, f"{agent}_merged.csv")
            merged_df.to_csv(output_file, index=False)
            print(
                f"Merged CSV for agent {agent} saved to {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns.")

        print("Metrics merging process completed successfully.")

    except Exception as e:
        print(f"Error during Metrics merging process: {e}")


def merge_metrics_obs_results(input_dir="Meta_6assets/Metrics_obs", output_dir="Meta_6assets/Metrics_obs"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/Verified output directory: {output_dir}")

        agents = ["ppo", "sac", "ddpg", "td3"]
        merged_dfs = []

        for agent in agents:
            csv_file = os.path.join(input_dir, f"{agent}_merged.csv")
            if not os.path.exists(csv_file):
                print(f"CSV file {csv_file} not found. Skipping agent {agent}.")
                continue

            try:
                df = pd.read_csv(csv_file)

                # Rename columns to include agent identifier
                if agent == agents[0]:
                    renamed_columns = {'month': 'month'}
                    for col in df.columns[1:]:
                        renamed_columns[col] = f"{col}_{agent}"
                else:
                    renamed_columns = {col: f"{col}_{agent}" for col in df.columns if col != 'month'}
                    df = df.drop(columns=['month'])

                df = df.rename(columns=renamed_columns)
                merged_dfs.append(df)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        if not merged_dfs:
            print("No valid CSV files found to merge.")
            return

        # Merge DataFrames side by side
        merged_df = merged_dfs[0]
        for df in merged_dfs[1:]:
            merged_df = pd.concat([merged_df, df], axis=1)

        # Save the final merged DataFrame
        output_file = os.path.join(output_dir, "metrics_obs_unclean.csv")
        merged_df.to_csv(output_file, index=False)
        print(
            f"Final merged CSV saved to {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns.")

        print("Metrics obs merging process completed successfully.")

    except Exception as e:
        print(f"Error during Metrics obs merging process: {e}")


def clean_metrics_obs_results(input_file="Meta_6assets/Metrics_obs/metrics_obs_unclean.csv",
                              output_file="Meta_6assets/Metrics_obs/metrics_obs_clean.csv"):
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found.")

        df = pd.read_csv(input_file)

        # Identify columns to keep: 'month' and all columns containing 'weight'
        columns_to_keep = ['month']
        for col in df.columns:
            if 'weight' in col:
                columns_to_keep.append(col)

        # Create the cleaned DataFrame
        cleaned_df = df[columns_to_keep]

        # Expected: 1 month + (6 weights × 5 seeds × 4 agents) = 121 columns
        expected_cleaned_columns = 1 + (6 * 5 * 4)
        if len(cleaned_df.columns) != expected_cleaned_columns:
            print(
                f"Warning: Cleaned DataFrame has {len(cleaned_df.columns)} columns, expected {expected_cleaned_columns}.")

        # Save the cleaned DataFrame
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned CSV saved to {output_file} with {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns.")

        print("Metrics obs cleaning process completed successfully.")

    except Exception as e:
        print(f"Error during Metrics obs cleaning process: {e}")


class CustomMetaEnvMetrics(gym.Env):

    def __init__(self, csv_path="Meta_6assets/Metrics_obs/metrics_obs_clean.csv",
                 price_path="metrics_used_6assets/price/clean_data.csv",
                 train_mode=True):
        super().__init__()

        # Load observation data
        df = pd.read_csv(csv_path)

        # Filter data based on mode to prevent data leakage
        if train_mode:
            # Training: use only 2003-2019 data for metrics
            train_start_month = pd.to_datetime(METRICS_TRAIN_START).strftime("%Y-%m")
            train_end_month = pd.to_datetime(TRAIN_END).strftime("%Y-%m")
            df = df[(df['month'] >= train_start_month) & (df['month'] <= train_end_month)].copy()
            print(f"Meta Training (Metrics): Using {len(df)} months ({train_start_month} to {train_end_month})")
        else:
            # Backtesting: use only 2020-2024 data
            backtest_start_month = pd.to_datetime(BACKTEST_START).strftime("%Y-%m")
            backtest_end_month = pd.to_datetime(BACKTEST_END).strftime("%Y-%m")
            df = df[(df['month'] >= backtest_start_month) & (df['month'] <= backtest_end_month)].copy()
            print(f"Meta Backtest (Metrics): Using {len(df)} months ({backtest_start_month} to {backtest_end_month})")

        if len(df) == 0:
            raise ValueError(f"No data available for {'training' if train_mode else 'backtesting'} period")

        self.months = df['month'].tolist()
        # Observations: 120 weight columns (6 weights × 5 seeds × 4 agents)
        self.observations = df.iloc[:, 1:].astype(np.float32).values

        # Load price data
        self.price_data = pd.read_csv(price_path, index_col="Date", parse_dates=True)
        self.tickers = TICKERS

        # Validate tickers
        for ticker in self.tickers:
            if ticker not in self.price_data.columns:
                raise ValueError(f"Ticker {ticker} not in price data")

        self.total_steps = len(self.months)

        # Define observation and action spaces
        obs_dim = self.observations.shape[1]
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


class CustomMetaEnvNLP(gym.Env):

    def __init__(self, csv_path="Meta_6assets/NLP_obs/nlp_obs_clean.csv",
                 price_path="metrics_used_6assets/price/clean_data.csv",
                 train_mode=True):
        super().__init__()

        # Load observation data
        df = pd.read_csv(csv_path)

        # Filter data based on mode to prevent data leakage
        if train_mode:
            # Training: use only 2012-2019 data for NLP
            train_start_month = pd.to_datetime(NLP_TRAIN_START).strftime("%Y-%m")
            train_end_month = pd.to_datetime(TRAIN_END).strftime("%Y-%m")
            df = df[(df['month'] >= train_start_month) & (df['month'] <= train_end_month)].copy()
            print(f"Meta Training (NLP): Using {len(df)} months ({train_start_month} to {train_end_month})")
        else:
            # Backtesting: use only 2020-2024 data
            backtest_start_month = pd.to_datetime(BACKTEST_START).strftime("%Y-%m")
            backtest_end_month = pd.to_datetime(BACKTEST_END).strftime("%Y-%m")
            df = df[(df['month'] >= backtest_start_month) & (df['month'] <= backtest_end_month)].copy()
            print(f"Meta Backtest (NLP): Using {len(df)} months ({backtest_start_month} to {backtest_end_month})")

        if len(df) == 0:
            raise ValueError(f"No data available for {'training' if train_mode else 'backtesting'} period")

        self.months = df['month'].tolist()
        # Observations: 120 weight columns (6 weights × 5 seeds × 4 agents)
        self.observations = df.iloc[:, 1:].astype(np.float32).values

        # Load price data
        self.price_data = pd.read_csv(price_path, index_col="Date", parse_dates=True)
        self.tickers = TICKERS

        # Validate tickers
        for ticker in self.tickers:
            if ticker not in self.price_data.columns:
                raise ValueError(f"Ticker {ticker} not in price data")

        self.total_steps = len(self.months)

        # Define observation and action spaces
        obs_dim = self.observations.shape[1]
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


def train_meta_agent_metrics():
    print("Training Metrics Meta-Agent...")
    # train_mode=True ensures we only use 2012-2019 data
    env = make_vec_env(lambda: CustomMetaEnvMetrics(train_mode=True), n_envs=1, seed=1)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={'net_arch': [256, 256, 256]},
        verbose=1,
        seed=1
    )
    model.learn(total_timesteps=20000, progress_bar=True)
    os.makedirs("meta_results_6assets", exist_ok=True)
    model_path = "meta_results_6assets/meta_agent_metrics_seed1"
    model.save(model_path)
    print(f"Metrics meta-agent trained and saved to {model_path}.zip")
    return model


def train_meta_agent_nlp():
    print("Training NLP Meta-Agent...")
    # train_mode=True ensures we only use 2012-2019 data
    env = make_vec_env(lambda: CustomMetaEnvNLP(train_mode=True), n_envs=1, seed=1)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={'net_arch': [256, 256, 256]},
        verbose=1,
        seed=1
    )
    model.learn(total_timesteps=20000, progress_bar=True)
    os.makedirs("meta_results_6assets", exist_ok=True)
    model_path = "meta_results_6assets/meta_agent_nlp_seed1"
    model.save(model_path)
    print(f"NLP meta-agent trained and saved to {model_path}.zip")
    return model


def backtest_meta_agent(model, env, output_file, agent_type="Metrics"):
    print(f"Backtesting {agent_type} Meta-Agent...")
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

    # Save backtest results
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Backtest results saved to {output_file} with {len(df)} rows and {len(df.columns)} columns.")


def main():
    print("=" * 80)
    print("META-AGENT TRAINING PIPELINE (6 ASSETS)")
    print("=" * 80)
    print("\nTrain/Test Split:")
    print("  - Meta Training (Metrics): 2003-2019 (from base agent backtests)")
    print("  - Meta Training (NLP): 2012-2019 (from base agent backtests)")
    print("  - Meta Backtest: 2020-2024 (out-of-sample)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STEP 1: Creating folder structure")
    print("=" * 80)
    create_meta_folders()

    print("\n" + "=" * 80)
    print("STEP 2: Processing NLP observations")
    print("=" * 80)
    merge_backtest_results_nlp()
    merge_nlp_obs_results()
    clean_nlp_obs_results()

    print("\n" + "=" * 80)
    print("STEP 3: Processing Metrics observations")
    print("=" * 80)
    merge_backtest_results_metrics()
    merge_metrics_obs_results()
    clean_metrics_obs_results()

    print("\n" + "=" * 80)
    print("STEP 4: Training Metrics Meta-Agent")
    print("=" * 80)
    metrics_model = train_meta_agent_metrics()

    print("\n" + "=" * 80)
    print("STEP 5: Training NLP Meta-Agent")
    print("=" * 80)
    nlp_model = train_meta_agent_nlp()

    print("\n" + "=" * 80)
    print("STEP 6: Backtesting Metrics Meta-Agent")
    print("=" * 80)
    # train_mode=False ensures we only use 2020-2024 data for backtesting
    metrics_env = CustomMetaEnvMetrics(train_mode=False)
    backtest_meta_agent(
        metrics_model,
        metrics_env,
        "Meta_6assets/Metrics_obs/backtest_meta_seed_1.csv",
        "Metrics"
    )

    print("\n" + "=" * 80)
    print("STEP 7: Backtesting NLP Meta-Agent")
    print("=" * 80)
    # train_mode=False ensures we only use 2020-2024 data for backtesting
    nlp_env = CustomMetaEnvNLP(train_mode=False)
    backtest_meta_agent(
        nlp_model,
        nlp_env,
        "Meta_6assets/NLP_obs/backtest_meta_seed_1.csv",
        "NLP"
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - Meta_6assets/Metrics_obs/")
    print("  - Meta_6assets/NLP_obs/")
    print("  - meta_results_6assets/")
    print("=" * 80)


if __name__ == "__main__":
    main()
