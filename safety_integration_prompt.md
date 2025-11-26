# Task: Integrate Safety Rules into Trading Environment

## Objective
Integrate the existing safety rules (specifically `NConsecutive`) into the `TradingEnvironment` to ensure the Agent's actions are filtered for safety and stability before execution.

## Context
- **Existing Rule**: `rules/nconsecutive.py` defines the `NConsecutive` class. This rule requires an action to be repeated `N` times consecutively before it is actually executed (otherwise `HOLD` is returned). This helps prevent "flickering" or spamming orders.
- **Target Environment**: `environments/environment.py` currently executes actions directly from the agent without any filtering.
- **Training Script**: `train.py` initializes the environment but currently does not configure any rules.

## Instructions

### 1. Modify `environments/environment.py`
Update the `TradingEnvironment` class to support a list of rules.
- **`__init__`**: Add a `rules` key to the `env_config` (optional, default to empty list). Store these rules in `self._rules`.
- **`step`**: Before processing the `action` (calculating reward and updating state), pass the `action` through all registered rules.
    - Iterate through `self._rules`.
    - Update `action = rule.filter(action)`.
    - Use the filtered action for the rest of the step logic.

### 2. Modify `train.py`
Update the training script to initialize the rule and pass it to the environment.
- Import `NConsecutive` from `rules.nconsecutive`.
- In the `train` function (or where `env_config` is created), initialize `NConsecutive` (e.g., with `window_size=3` or a configurable parameter).
- Add the initialized rule to the `metrics` list or a new `rules` list in `env_config`. *Note: The current environment seems to mix metrics and config, ensure rules are passed correctly.*

### 3. Verification
- Run a short training session (using `train.py` with low iterations) to ensure the code runs without errors.
- Verify that the rule is actually affecting actions (e.g., by adding a print statement in the rule or environment temporarily).
