# Task: Refactor `integrated.ipynb` to `integrated.py`

## Objective
Convert the hybrid strategy notebook `integrated.ipynb` into a clean, executable Python script `integrated.py`. The new script must use **PyTorch** and **Stable Baselines 3** (SB3) models.

## Context
- **Source File**: `integrated.ipynb` (currently uses TF-Agents).
- **Target File**: `integrated.py` (new file).
- **Dependencies**: `agents.torch.ppo_agent`, `agents.torch.dqn_agent`, `stable_baselines3`, `environments.environment.TradingEnvironment`.

## Instructions

### 1. Imports
- Remove all `tensorflow` and `tf_agents` imports.
- Import necessary PyTorch and SB3 libraries:
    ```python
    import os
    import torch
    import numpy as np
    import pandas as pd
    import config
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from agents.torch.ppo_agent import PPOAgent
    from agents.torch.dqn_agent import DQNAgent
    from environments.environment import TradingEnvironment
    from environments.rewards.marketlimitorder import MarketLimitOrderRF
    # Import metrics...
    ```

### 2. `build_eval_env`
- Reuse the `build_eval_env` logic from `eval.py`.
- Ensure it supports `n_consecutive_size` if needed (though the integrated strategy logic might handle safety differently, the original notebook passed it to `TFRuleTradingEnvironment`. In our new design, `TradingEnvironment` handles rules directly).

### 3. `load_agent`
- Reuse `load_agent` from `eval.py`.

### 4. `eval_tradernet_smurf` (Hybrid Logic)
- Rewrite the hybrid evaluation loop:
    ```python
    def eval_tradernet_smurf(tradernet_agent, smurf_agent, env):
        # Reset environment
        obs = env.reset()
        done = False
        cumulative_rewards = 0.0
        cumulative_pnls = 0.0
        pnls = []
        
        # Action.HOLD is usually 2. Verify in environments/actions.py
        HOLD_ACTION = 2 
        
        while True:
            # 1. Ask Smurf Agent
            smurf_action, _ = smurf_agent.predict(obs, deterministic=True)
            smurf_action_val = smurf_action[0]
            
            # 2. Decide Action
            if smurf_action_val == HOLD_ACTION:
                final_action = smurf_action
            else:
                # Ask TraderNet Agent
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
    ```

### 5. Main Execution Block
- Define `datasets_dict`, `agent_dict` (for both TraderNet and Smurf), `env_dict`.
- Iterate and run evaluation.
- Save results to `experiments/integrated/...`.

## Output
- The script should be runnable via `python integrated.py`.
