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
        reward_fn_instance
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
        fees_percentage=fees
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
        **kwargs
):
    x_eval, eval_reward_fn = read_dataset(
        dataset_filepath=dataset_filepath,
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        num_eval_samples=num_eval_samples,
        fees=fees,
        reward_fn_instance=reward_fn_instance
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
    model_path = os.path.join(checkpoint_filepath, "model")
    if not os.path.exists(model_path + ".zip"):
        return None
    try:
        agent = agent_class.load(model_path, env=env)
        return agent
    except Exception as e:
        print(f"Error loading agent from {model_path}: {e}")
        return None

def eval_tradernet_smurf(tradernet_agent, smurf_agent, env):
    obs = env.reset()
    done = False
    cumulative_rewards = 0.0
    cumulative_pnls = 0.0
    pnls = []
    
    # Action.HOLD is 2
    HOLD_ACTION = 2 
    
    while True:
        # 1. Ask Smurf Agent ("Should I HOLD?")
        # Smurf agent is trained to prefer HOLD (action 2) if it's safe/rewarding.
        smurf_action, _ = smurf_agent.predict(obs, deterministic=True)
        smurf_action_val = smurf_action[0]
        
        # 2. Decide Action
        if smurf_action_val == HOLD_ACTION:
            final_action = smurf_action
        else:
            # If Smurf says "Not HOLD" (Buy/Sell), ask TraderNet for the specific trade action
            tradernet_action, _ = tradernet_agent.predict(obs, deterministic=True)
            final_action = tradernet_action
        
        # 3. Step Environment
        obs, reward, done, info = env.step(final_action)
        
        reward_val = reward[0]
        final_action_val = final_action[0]
        
        cumulative_rewards += reward_val
        
        if final_action_val != HOLD_ACTION: 
            cumulative_pnls += reward_val
        
        pnls.append(cumulative_pnls)
        
        if done[0]:
            break
            
    return cumulative_rewards, pnls

if __name__ == "__main__":
    # Example: TraderNet=PPO, Smurf=DDQN (as seen in original notebook)
    # We can infer this from config.agent_config if we want, or define pairs here.
    # For now, let's keep the explicit pair definition but use classes from config.
    agent_pairs = [
        {
            'name': 'PPO_TraderNet_DDQN_Smurf',
            'tradernet': {'class': config.agent_config['PPO']['agent_class'], 'name': 'PPO'},
            'smurf': {'class': config.agent_config['DDQN']['agent_class'], 'name': 'DDQN'}
        },
    ]

    for pair in agent_pairs:
        tradernet_conf = pair['tradernet']
        smurf_conf = pair['smurf']
        
        for dataset_name, dataset_filepath in config.datasets_dict.items():
            for reward_fn_name, reward_fn_instance in config.reward_config.items():
                print(f"Evaluating Integrated {pair['name']} on {dataset_name} with {reward_fn_name}...")
                
                # Build environment
                env = build_eval_env(
                    dataset_filepath=dataset_filepath,
                    reward_fn_instance=reward_fn_instance,
                    **config.env_config
                )
                
                # Load TraderNet Agent
                tradernet_path = f'database/storage/checkpoints/experiments/tradernet/{tradernet_conf["name"]}/{dataset_name}/{reward_fn_name}/'
                tradernet_agent = load_agent(tradernet_conf['class'], tradernet_path, env)
                
                # Load Smurf Agent
                smurf_path = f'database/storage/checkpoints/experiments/smurf/{smurf_conf["name"]}/{dataset_name}/{reward_fn_name}/'
                smurf_agent = load_agent(smurf_conf['class'], smurf_path, env)
                
                if tradernet_agent is None or smurf_agent is None:
                    print(f"Skipping {pair['name']} due to missing models.")
                    if tradernet_agent is None: print(f"Missing TraderNet at {tradernet_path}")
                    if smurf_agent is None: print(f"Missing Smurf at {smurf_path}")
                    continue

                # Evaluate Hybrid
                average_returns, pnls = eval_tradernet_smurf(
                    tradernet_agent=tradernet_agent.model,
                    smurf_agent=smurf_agent.model,
                    env=env
                )
                
                # Get episode metrics
                base_env = env.envs[0].unwrapped
                episode_metrics = base_env.get_metrics() if hasattr(base_env, 'get_metrics') else []
                
                metrics = {
                    'average_returns': [average_returns],
                    **{metric.name: [metric.result()] for metric in episode_metrics}
                }
                results_df = pd.DataFrame(metrics)
                
                output_dir = f'experiments/integrated/{tradernet_conf["name"]}' 
                
                output_metrics_path = f'{output_dir}/{dataset_name}_{reward_fn_name}_metrics.csv'
                os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
                results_df.to_csv(output_metrics_path, index=False)

                print(results_df, '\n')

                episode_pnls_df = pd.DataFrame(pnls, columns=['cumulative_pnl'])
                output_pnls_path = f'{output_dir}/{dataset_name}_{reward_fn_name}_eval_cumul_pnls.csv'
                episode_pnls_df.to_csv(output_pnls_path, index=False)

                print(episode_pnls_df.tail(5))
