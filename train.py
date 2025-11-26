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
from stable_baselines3.common.evaluation import evaluate_policy
from agents.torch.ppo_agent import PPOAgent
from agents.torch.dqn_agent import DQNAgent
from environments.environment import TradingEnvironment
from environments.rewards.marketorder import MarketOrderRF
from environments.rewards.marketlimitorder import MarketLimitOrderRF
from metrics.trading.pnl import CumulativeLogReturn
from metrics.trading.risk import InvestmentRisk
from metrics.trading.sharpe import SharpeRatio
from metrics.trading.sortino import SortinoRatio
from metrics.trading.drawdown import MaximumDrawdown
from rules.nconsecutive import NConsecutive

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
    x_train = inputs[: num_train_inputs]
    x_eval = inputs[num_train_inputs:]

    # Computing reward functions for train, test data
    closes = crypto_dataset_df['close'].to_numpy(dtype=np.float32)
    highs = crypto_dataset_df['high'].to_numpy(dtype=np.float32)
    lows = crypto_dataset_df['low'].to_numpy(dtype=np.float32)

    train_reward_fn = reward_fn_instance(
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        highs=highs[: samples.shape[0] - num_eval_samples],
        lows=lows[: samples.shape[0] - num_eval_samples],
        closes=closes[: samples.shape[0] - num_eval_samples],
        fees_percentage=fees,
        position_size=position_size,
        leverage=leverage
    )

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

    assert x_train.shape[0] == train_reward_fn.get_reward_fn_shape()[0], \
        f'AssertionError: DimensionMismatch: x_train: {x_train.shape}, train_reward_fn: {train_reward_fn.get_reward_fn_shape()}'
    assert x_eval.shape[0] == eval_reward_fn.get_reward_fn_shape()[0], \
        f'AssertionError: DimensionMismatch: x_eval: {x_eval.shape}, eval_reward_fn: {eval_reward_fn.get_reward_fn_shape()}'

    return x_train, train_reward_fn, x_eval, eval_reward_fn

def train(
        dataset_filepath,
        timeframe_size,
        target_horizon_len,
        num_eval_samples,
        fees,
        reward_fn_instance,
        agent_class,
        checkpoint_filepath,
        fc_layers,
        conv_layers,
        train_episode_steps,
        train_iterations,
        eval_episodes,
        steps_per_eval,
        steps_per_log,
        steps_per_checkpoint,
        save_best_only,
        position_size=1.0,
        leverage=1.0,
        **kwargs
):
    x_train, train_reward_fn, x_eval, eval_reward_fn = read_dataset(
        dataset_filepath=dataset_filepath,
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        num_eval_samples=num_eval_samples,
        fees=fees,
        reward_fn_instance=reward_fn_instance,
        position_size=position_size,
        leverage=leverage
    )

    # Create training environment wrapped with Monitor and DummyVecEnv for SB3
    def make_train_env():
        env = TradingEnvironment(env_config={
            'states': x_train,
            'reward_fn': train_reward_fn,
            'episode_steps': train_episode_steps,
            'metrics': [CumulativeLogReturn(), InvestmentRisk(), SharpeRatio(), SortinoRatio(), MaximumDrawdown()],
            'rules': [NConsecutive(window_size=3)]
        })
        return Monitor(env)

    def make_eval_env():
        env = TradingEnvironment(env_config={
            'states': x_eval,
            'reward_fn': eval_reward_fn,
            'episode_steps': x_eval.shape[0] - 1,
            'metrics': [CumulativeLogReturn(), InvestmentRisk(), SharpeRatio(), SortinoRatio(), MaximumDrawdown()],
            'rules': [NConsecutive(window_size=3)]
        })
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    # Initialize agent with SB3 wrapper
    # Initialize agent with SB3 wrapper
    agent = agent_class(
        env=train_env,
        tensorboard_log=checkpoint_filepath,
        verbose=0,
        **kwargs
    )

    # Train the agent
    agent.train(total_timesteps=train_iterations, progress_bar=True)

    # Save the model
    agent.save(checkpoint_filepath + 'model')

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        agent.model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )

    # Get metrics from the underlying environment
    eval_base_env = eval_env.envs[0].unwrapped
    eval_metrics = eval_base_env.get_metrics() if hasattr(eval_base_env, 'get_metrics') else []

    return [mean_reward], eval_metrics

if __name__ == "__main__":
    # Configuration
    # Results container
    results = {
        agent_name: {dataset_name: {} for dataset_name in config.datasets_dict.keys()}
        for agent_name in config.agent_config.keys()
    }

    # Ensure directories exist
    os.makedirs('experiments/tradernet', exist_ok=True)

    # Main training loop
    for agent_name, agent_params in config.agent_config.items():
        for dataset_name, dataset_filepath in config.datasets_dict.items():
            for reward_fn_name, reward_fn_instance in config.reward_config.items():
                print(f"Training {agent_name} on {dataset_name} with {reward_fn_name}...")
                torch.manual_seed(0)
                np.random.seed(0)

                train_params = {
                    'dataset_filepath': dataset_filepath,
                    'reward_fn_instance': reward_fn_instance,
                    'checkpoint_filepath': f'database/storage/checkpoints/experiments/tradernet/{agent_name}/{dataset_name}/{reward_fn_name}/',
                    **config.env_config,
                    **agent_params
                }
                
                # Ensure checkpoint directory exists
                os.makedirs(train_params['checkpoint_filepath'], exist_ok=True)
                
                eval_avg_returns, eval_metrics = train(**train_params)

                results[agent_name][dataset_name][reward_fn_name] = (eval_avg_returns, eval_metrics)

            for reward_fn_name, reward_fn_results in results[agent_name][dataset_name].items():
                eval_avg_returns, eval_metrics = reward_fn_results

                metrics_dict = {
                    'steps': [10000*i for i in range(len(eval_avg_returns))],
                    'average_returns': eval_avg_returns,
                    **{metric.name: metric.episode_metrics for metric in eval_metrics}
                }
                metrics_df = pd.DataFrame(metrics_dict)
                output_csv_path = f'experiments/tradernet/{agent_name}/{dataset_name}_{reward_fn_name}.csv'
                os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
                metrics_df.to_csv(output_csv_path, index=False)
                print(f"Saved results to {output_csv_path}")
