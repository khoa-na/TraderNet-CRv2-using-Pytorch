#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import config
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from agents.torch.ppo_agent import PPOAgent
from agents.torch.dqn_agent import DQNAgent
from environments.environment import TradingEnvironment
from environments.rewards.marketlimitorder import MarketLimitOrderRF
from metrics.trading.pnl import CumulativeLogReturn
from metrics.trading.risk import InvestmentRisk
from metrics.trading.sharpe import SharpeRatio
from metrics.trading.sortino import SortinoRatio
from metrics.trading.drawdown import MaximumDrawdown

def read_dataset(
        dataset_filepath,
        timeframe_size,
        target_horizon_len,
        num_eval_samples,
        fees,
        reward_fn_instance,
        position_size=1.0,
        leverage=1.0
):
    # Reading dataset
    crypto_dataset_df = pd.read_csv(config.dataset_save_filepath.format(dataset_filepath))
    samples_df = crypto_dataset_df[config.regression_features]

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1.0))
    samples = samples_df.to_numpy(dtype=np.float32)

    num_train_scale_samples = samples.shape[0] - num_eval_samples - target_horizon_len - timeframe_size + 1
    samples[: num_train_scale_samples] = scaler.fit_transform(samples[: num_train_scale_samples])
    samples[num_train_scale_samples: ] = scaler.transform(samples[num_train_scale_samples: ])

    # Constructing timeframes for train, test
    inputs = np.float32([samples[i: i + timeframe_size] for i in range(samples.shape[0] - timeframe_size - target_horizon_len + 1)])

    # Splitting inputs to train-test data
    num_train_inputs = inputs.shape[0] - num_eval_samples
    x_eval = inputs[num_train_inputs:]

    # Computing reward functions for train, test data
    closes = crypto_dataset_df['close'].to_numpy(dtype=np.float32)
    highs = crypto_dataset_df['high'].to_numpy(dtype=np.float32)
    lows = crypto_dataset_df['low'].to_numpy(dtype=np.float32)

    eval_reward_fn = reward_fn_instance(
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        highs=highs[samples.shape[0] - num_eval_samples - timeframe_size - target_horizon_len + 1:],
        lows=lows[samples.shape[0] - num_eval_samples - timeframe_size - target_horizon_len + 1:],
        closes=closes[samples.shape[0] - num_eval_samples - timeframe_size - target_horizon_len + 1:],
        fees_percentage=fees,
        position_size=position_size,
        leverage=leverage
    )

    assert x_eval.shape[0] == eval_reward_fn.get_reward_fn_shape()[0], \
        f'AssertionError: DimensionMismatch: x_eval: {x_eval.shape}, eval_reward_fn: {eval_reward_fn.get_reward_fn_shape()}'

    return x_eval, eval_reward_fn

def build_eval_env(
        dataset_filepath,
        timeframe_size,
        target_horizon_len,
        num_eval_samples,
        fees,
        reward_fn_instance,
        position_size=1.0,
        leverage=1.0,
        **kwargs
):
    x_eval, eval_reward_fn = read_dataset(
        dataset_filepath=dataset_filepath,
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        num_eval_samples=num_eval_samples,
        fees=fees,
        reward_fn_instance=reward_fn_instance,
        position_size=position_size,
        leverage=leverage
    )

    def make_env():
        env = TradingEnvironment(env_config={
            'states': x_eval,
            'reward_fn': eval_reward_fn,
            'episode_steps': x_eval.shape[0] - 1,
            'metrics': [CumulativeLogReturn(), InvestmentRisk(), SharpeRatio(), SortinoRatio(), MaximumDrawdown()]
        })
        return Monitor(env)

    env = DummyVecEnv([make_env])
    return env

def load_agent(agent_class, checkpoint_filepath, env):
    # SB3 models are saved as zip files. We append 'model' because train.py saves as 'model.zip'
    model_path = os.path.join(checkpoint_filepath, "model")
    
    # Check if file exists (SB3 adds .zip automatically)
    if not os.path.exists(model_path + ".zip"):
        print(f"Warning: Model not found at {model_path}.zip")
        return None

    try:
        agent = agent_class.load(model_path, env=env)
        return agent
    except Exception as e:
        print(f"Error loading agent from {model_path}: {e}")
        return None

def eval_tradernet(agent, env):
    # Reset environment
    obs = env.reset()
    done = False
    cumulative_rewards = 0.0
    cumulative_pnls = 0.0
    pnls = []
    
    # We need to access the underlying environment to check for termination properly in a loop
    # or just trust the done flag from DummyVecEnv (which auto-resets)
    # For evaluation of a single continuous episode, we want to stop when the episode ends.
    
    while True:
        # SB3 predict returns (action, state)
        action, _ = agent.predict(obs, deterministic=True)
        
        # Gym step
        obs, reward, done, info = env.step(action)
        
        # Extract scalar reward from array (DummyVecEnv returns array)
        reward_val = reward[0]
        action_val = action[0]
        
        cumulative_rewards += reward_val
        
        # Logic: "if action != 2: cumulative_pnls += reward"
        # Action.HOLD is 2 in environments/actions.py (usually)
        # Let's verify: Action.LONG=0, Action.SHORT=1, Action.HOLD=2
        if action_val != 2: 
            cumulative_pnls += reward_val
        
        pnls.append(cumulative_pnls)
        
        # Check if the episode is done
        if done[0]:
            break
            
    return cumulative_rewards, pnls

if __name__ == "__main__":
    for agent_name, agent_params in config.agent_config.items():
        for dataset_name, dataset_filepath in config.datasets_dict.items():
            # Evaluation typically runs on one reward function type, or we loop through them?
            # The original code had a fixed reward_fn_name. Let's loop through config.reward_config
            # but maybe we only want to eval one. For now, let's loop to be consistent.
            for reward_fn_name, reward_fn_instance in config.reward_config.items():
                print(f"Evaluating {agent_name} on {dataset_name} with {reward_fn_name}...")
                
                # Build environment
                env = build_eval_env(
                    dataset_filepath=dataset_filepath,
                    reward_fn_instance=reward_fn_instance,
                    **config.env_config
                )
                
                # Load agent
                checkpoint_path = f'database/storage/checkpoints/experiments/tradernet/{agent_name}/{dataset_name}/{reward_fn_name}/'
                agent = load_agent(
                    agent_class=agent_params['agent_class'],
                    checkpoint_filepath=checkpoint_path,
                    env=env
                )
                
                if agent is None:
                    print(f"Skipping {agent_name} on {dataset_name} due to missing model.")
                    continue

                # Evaluate
                average_returns, pnls = eval_tradernet(
                    agent=agent.model, # Pass the SB3 model directly
                    env=env
                )
                
                # Get episode metrics from the environment
                # Access the first env in DummyVecEnv
                base_env = env.envs[0].unwrapped
                episode_metrics = base_env.get_metrics() if hasattr(base_env, 'get_metrics') else []
                
                metrics = {
                    'average_returns': [average_returns],
                    **{metric.name: [metric.result()] for metric in episode_metrics} # Use result() to get final value
                }
                results_df = pd.DataFrame(metrics)
                
                output_metrics_path = f'experiments/tradernet/{agent_name}/{dataset_name}_{reward_fn_name}_metrics.csv'
                os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
                results_df.to_csv(output_metrics_path, index=False)

                print(results_df, '\n')

                episode_pnls_df = pd.DataFrame(pnls, columns=['cumulative_pnl'])
                output_pnls_path = f'experiments/tradernet/{agent_name}/{dataset_name}_{reward_fn_name}_eval_cumul_pnls.csv'
                episode_pnls_df.to_csv(output_pnls_path, index=False)

                print(episode_pnls_df.tail(5))
