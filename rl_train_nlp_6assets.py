import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


# --- Configuration ---
PRICE_DIR = os.path.join("metrics_used", "price")
NLP_METRICS_DIR = os.path.join("observation", "metrics")
PRICE_METRICS_DIR = os.path.join("observation", "metrics_price_6assets")
RESULTS_DIR_NLP = "results_nlp_6assets"
BACKTEST_DIR_NLP = "backtest_results_nlp_6assets"
RESULTS_DIR_PRICE = "results_price_6assets"
BACKTEST_DIR_PRICE = "backtest_results_price_6assets"

TICKERS = ["^GSPC", "^IXIC", "^DJI", "GC=F", "SI=F", "CL=F"]
START_DATE = "2012-01-01"
END_DATE = "2024-12-31"

TRAIN_START_MONTH = "2012-01"
TRAIN_END_MONTH = "2019-12"
BACKTEST_START_MONTH = "2020-01"
BACKTEST_END_MONTH = "2024-11"

SEEDS = [1, 2, 3, 4, 5]
TOTAL_TIMESTEPS = 10000


def fetch_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, actions=False)
    if data.empty:
        return data
    data = data["Close"] if len(tickers) > 1 else data[["Close"]]
    data = data.reindex(columns=tickers)
    return data


def clean_price_data(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df = df.bfill()
    df = df.interpolate(method="linear", limit_direction="both")
    return df


def save_price_data(df: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(PRICE_DIR, exist_ok=True)
    raw_path = os.path.join(PRICE_DIR, "stock_closing_prices.csv")
    clean_path = os.path.join(PRICE_DIR, "clean_data.csv")
    df.to_csv(raw_path)
    cleaned = clean_price_data(df)
    cleaned.to_csv(clean_path)
    print(f"Saved {raw_path} and {clean_path}")
    return cleaned


def create_price_observations(cleaned_prices: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    monthly_groups = cleaned_prices.groupby([cleaned_prices.index.year, cleaned_prices.index.month])
    for (year, month), group in monthly_groups:
        log_returns = np.log(group / group.shift(1)).dropna()
        if log_returns.empty:
            continue
        vol = log_returns.std() * np.sqrt(252)
        output_file = os.path.join(output_dir, f"{year}_{month:02d}_combined.csv")
        vol.reindex(TICKERS).to_frame().to_csv(
            output_file,
            index=False,
            header=False,
            encoding="utf-8",
        )


def get_obs_months(start_month: str, end_month: str) -> list[str]:
    start = pd.to_datetime(f"{start_month}-01")
    end = pd.to_datetime(f"{end_month}-01")
    return pd.date_range(start=start, end=end, freq="MS").strftime("%Y-%m").tolist()


def load_observations(metrics_dir: str, obs_months: list[str], expected_dim: int) -> dict[str, np.ndarray]:
    observations = {}
    for month in obs_months:
        year, month_num = month.split("-")
        file_path = os.path.join(metrics_dir, f"{year}_{month_num}_combined.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Observation file for {month} not found.")
            continue
        try:
            df = pd.read_csv(file_path, header=None)
            if len(df) != expected_dim:
                print(f"Warning: {file_path} has {len(df)} rows, expected {expected_dim}. Skipping.")
                continue
            observations[month] = df.iloc[:, 0].values.astype(np.float32)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return observations


class CustomPortfolioEnv(gym.Env):
    def __init__(self, price_dir: str, metrics_dir: str, obs_months: list[str]):
        super().__init__()
        self.tickers = TICKERS
        self.metrics_dir = metrics_dir
        self.obs_months = obs_months

        price_file = os.path.join(price_dir, "clean_data.csv")
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price file {price_file} not found.")
        self.daily_prices = pd.read_csv(price_file, index_col="Date", parse_dates=True)

        self.observations = load_observations(metrics_dir, obs_months, expected_dim=len(TICKERS))
        if not self.observations:
            raise ValueError("No valid observation files found.")

        self.total_steps = len(obs_months) - 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(TICKERS),), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(TICKERS),), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        obs = self.observations[self.obs_months[0]]
        return obs, {}

    def step(self, action):
        if self.current_step >= self.total_steps:
            raise ValueError("Episode has ended.")

        action_sum = np.sum(action)
        weights = action / action_sum if action_sum > 0 else np.ones(len(TICKERS)) / len(TICKERS)

        allocation_month = self.obs_months[self.current_step + 1]
        start_date = pd.to_datetime(allocation_month + "-01")
        end_date = start_date + pd.offsets.MonthEnd(0)

        try:
            daily_prices = self.daily_prices.loc[start_date:end_date, self.tickers]
        except KeyError:
            print(f"Warning: Missing price data for {start_date} to {end_date}. Using zeros.")
            daily_prices = pd.DataFrame(0, index=pd.date_range(start_date, end_date), columns=self.tickers)

        daily_values = np.sum(daily_prices.values * weights, axis=1)
        roi = (daily_values[-1] / daily_values[0] - 1) if daily_values[0] > 0 else 0
        daily_returns = daily_values[1:] / daily_values[:-1] - 1 if len(daily_values) > 1 else np.array([0])
        volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
        cummax = np.maximum.accumulate(daily_values)
        drawdown = (cummax - daily_values) / cummax if cummax[0] > 0 else np.zeros_like(daily_values)
        mdd = np.max(drawdown) if len(drawdown) > 0 else 0
        reward = 2 * roi - 0.7 * volatility - 0.5 * mdd

        self.current_step += 1
        done = self.current_step >= self.total_steps
        truncated = False
        next_obs = self.observations[self.obs_months[self.current_step]] if not done else np.zeros(len(TICKERS))
        info = {"allocation_month": allocation_month, "roi": roi, "volatility": volatility, "mdd": mdd}
        return next_obs, reward, done, truncated, info


def train_models(metrics_dir: str, results_dir: str):
    obs_months = get_obs_months(TRAIN_START_MONTH, TRAIN_END_MONTH)
    env = make_vec_env(lambda: CustomPortfolioEnv(PRICE_DIR, metrics_dir, obs_months), n_envs=1)
    os.makedirs(results_dir, exist_ok=True)

    models = {"ppo": PPO, "sac": SAC, "ddpg": DDPG, "td3": TD3}
    evaluation_results = []

    for seed in SEEDS:
        np.random.seed(seed)
        random.seed(seed)
        for model_name, model_class in models.items():
            model = model_class("MlpPolicy", env, verbose=0, seed=seed)
            model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
            model_path = os.path.join(results_dir, f"{model_name}_seed_{seed}.zip")
            model.save(model_path)
            evaluation_results.append(
                {"model": model_name, "seed": seed, "mean_reward": mean_reward, "std_reward": std_reward}
            )
            print(f"Trained and saved {model_name} seed {seed} to {model_path}")

    eval_df = pd.DataFrame(evaluation_results)
    eval_path = os.path.join(results_dir, "evaluation.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved evaluation results: {eval_path}")


def backtest_model(model, env, output_file):
    obs, _ = env.reset()
    done = False
    records = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        weights = action / np.sum(action) if np.sum(action) > 0 else np.ones(len(TICKERS)) / len(TICKERS)
        record = {"month": env.obs_months[env.current_step + 1]}
        for i, ticker in enumerate(TICKERS):
            record[f"weight_{ticker}"] = weights[i]
        obs, reward, done, _, info = env.step(action)
        record.update({"roi": info["roi"], "volatility": info["volatility"], "mdd": info["mdd"], "reward": reward})
        records.append(record)
    pd.DataFrame(records).to_csv(output_file, index=False)
    print(f"Backtest results saved to {output_file}")


def run_backtests(metrics_dir: str, results_dir: str, backtest_dir: str):
    obs_months = get_obs_months(BACKTEST_START_MONTH, BACKTEST_END_MONTH)
    env = CustomPortfolioEnv(PRICE_DIR, metrics_dir, obs_months)
    os.makedirs(backtest_dir, exist_ok=True)
    for model_type in ["ppo", "sac", "ddpg", "td3"]:
        model_dir = os.path.join(backtest_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        for seed in SEEDS:
            model_path = os.path.join(results_dir, f"{model_type}_seed_{seed}.zip")
            if not os.path.exists(model_path):
                print(f"Model {model_path} not found. Skipping.")
                continue
            try:
                model = globals()[model_type.upper()].load(model_path)
                output_file = os.path.join(model_dir, f"seed_{seed}.csv")
                backtest_model(model, env, output_file)
            except Exception as e:
                print(f"Error backtesting {model_type} seed {seed}: {e}")


def main():
    price_df = fetch_price_data(TICKERS, START_DATE, END_DATE)
    if price_df.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")
    cleaned_prices = save_price_data(price_df)
    create_price_observations(cleaned_prices, PRICE_METRICS_DIR)

    print("Starting price-metrics training...")
    train_models(PRICE_METRICS_DIR, RESULTS_DIR_PRICE)
    print("Starting price-metrics backtesting...")
    run_backtests(PRICE_METRICS_DIR, RESULTS_DIR_PRICE, BACKTEST_DIR_PRICE)

    print("Starting NLP training...")
    train_models(NLP_METRICS_DIR, RESULTS_DIR_NLP)
    print("Starting NLP backtesting...")
    run_backtests(NLP_METRICS_DIR, RESULTS_DIR_NLP, BACKTEST_DIR_NLP)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
