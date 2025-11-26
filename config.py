from database.entities.crypto import Crypto
# from database.network.coinapi.ohlcv import OHLCVDownloader

# --- Database ---
# supported_cryptos = { ... } # Removed as it was only used by the old downloader

ohlcv_dataset_period_id = '1h'
ohlcv_history_filepath = 'database/storage/downloads/ohlcv/{}.csv'
# gtrends_history_filepath = 'database/storage/downloads/gtrends/{}.csv' # Removed
dataset_save_filepath = 'database/storage/datasets/{}.csv'
# all_features = [...] # Removed as it is unused

regression_features = [
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns', 'hour',
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down', 'rsi', 'adx', 'cci',
    'close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down', 'adl_diffs', 'obv_diffs'
]

# --- Model ---
checkpoint_dir = 'database/storage/checkpoints/'

# --- Training Configuration ---
import torch
from agents.torch.ppo_agent import PPOAgent
from agents.torch.dqn_agent import DQNAgent
from environments.rewards.marketlimitorder import MarketLimitOrderRF
from environments.rewards.marketorder import MarketOrderRF

# Datasets to train on
datasets_dict = {'DOGEUSDT': 'DOGEUSDT'}

# Environment parameters
env_config = {
    'timeframe_size': 12,
    'target_horizon_len': 20,
    'num_eval_samples': 2250,
    'fees': 0.007,
    'fc_layers': [256, 256],
    'conv_layers': [(32, 3, 1)],
    'train_episode_steps': 100, # Steps per episode during training
    'eval_episodes': 1,         # Number of episodes to evaluate
    'save_best_only': True
}

# Agent parameters
agent_config = {
    'PPO': {
        'agent_class': PPOAgent,
        'learning_rate': 1e-4,
        'batch_size': 64,
        'train_iterations': 100000,
        'steps_per_eval': 1000,
        'steps_per_log': 1000,
        'steps_per_checkpoint': 1000,
        'device': 'cpu' # Force CPU to avoid SB3 warning for MlpPolicy
    },
    'DDQN': {
        'agent_class': DQNAgent,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'train_iterations': 100000,
        'steps_per_eval': 1000,
        'steps_per_log': 1000,
        'steps_per_checkpoint': 1000,
        'device': 'cpu'
    }
}

# Reward functions
reward_config = {
    'Market-Limit Orders': MarketLimitOrderRF,
    # 'Market-Orders': MarketOrderRF # Uncomment to enable
}
