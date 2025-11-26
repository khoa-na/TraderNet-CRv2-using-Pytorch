# Task: Refactor `tradernet_eval.ipynb` to `eval.py`

## Objective
Convert the existing evaluation notebook `tradernet_eval.ipynb` into a clean, executable Python script `eval.py`. The new script must use **PyTorch** and **Stable Baselines 3** (SB3) instead of TensorFlow and TF-Agents.

## Context
- **Source File**: `tradernet_eval.ipynb` (currently uses TF-Agents).
- **Target File**: `eval.py` (new file).
- **Dependencies**: Use `agents.torch.ppo_agent`, `agents.torch.dqn_agent`, `stable_baselines3`, and `environments.environment.TradingEnvironment`.

## Instructions

### 1. Imports
- Remove all `tensorflow` and `tf_agents` imports.
- Import necessary PyTorch and SB3 libraries:
    ```python
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

### 2. `read_dataset` / `build_eval_environments`
- Reuse the `read_dataset` logic from `train.py` (or import it if you refactor `train.py` to be a module, but copying/adapting is fine for now to keep it standalone).
- The function should return `x_eval` and `eval_reward_fn`.
- Wrap the `TradingEnvironment` with `Monitor` and `DummyVecEnv` for SB3 compatibility.

### 3. `load_agent`
- Instead of `build_agent` which initializes a new agent, create a `load_agent` function.
- Use `PPOAgent.load(path, env)` or `DQNAgent.load(path, env)`.
- **Important**: The checkpoint path in `tradernet_eval.ipynb` points to `database/storage/checkpoints/...`. Ensure the path structure matches where `train.py` saves models (e.g., appending `/model.zip` if SB3 adds it, or just the base path).

### 4. `eval_tradernet` (Evaluation Loop)
- Rewrite the evaluation loop to use standard Gym API and SB3 model:
    ```python
    def eval_tradernet(agent, env):
        # Reset environment
        obs = env.reset()
        done = False
        cumulative_rewards = 0.0
        cumulative_pnls = 0.0
        pnls = []
        
        while not done:
            # SB3 predict returns (action, state)
            action, _ = agent.predict(obs, deterministic=True)
            
            # Gym step
            obs, reward, done, info = env.step(action)
            
            cumulative_rewards += reward
            
            # Logic from original notebook: "if action != 2: cumulative_pnls += reward"
            # Check Action enum value for HOLD (usually 2 or 0, check environments/actions.py)
            # Assuming Action.HOLD.value is 2 based on original code, but verify.
            if action != 2: 
                cumulative_pnls += reward
            
            pnls.append(cumulative_pnls)
            
        return cumulative_rewards, pnls
    ```
- *Note*: `DummyVecEnv` automatically resets when done, so `env.step` might return the first observation of the next episode. For evaluation of a single episode, you might want to use `env.envs[0]` (the unwrapped env) or handle the `done` flag carefully. SB3's `evaluate_policy` is easier but might not give the detailed `pnls` list per step. A custom loop using `env.envs[0].step(action)` is often clearer for single-episode analysis.

### 5. Main Execution Block
- Define `datasets_dict`, `agent_dict`, `env_dict` similar to the notebook.
- Iterate through agents and datasets.
- Load the agent.
- Run evaluation.
- Save results to CSV (metrics and cumulative PnLs).

## Output
- The script should be runnable via `python eval.py`.
