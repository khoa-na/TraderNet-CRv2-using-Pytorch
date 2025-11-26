import numpy as np
from metrics.metric import Metric


class SharpeRatio(Metric):
    def __init__(self):
        super().__init__(name='Sharpe')
        self._episode_log_pnls = []

    def reset(self):
        self._episode_log_pnls = []

    def update(self, log_pnl: float):
        self._episode_log_pnls.append(log_pnl)

    def result(self) -> float:
        episode_log_returns = np.float64(self._episode_log_pnls)
        if len(episode_log_returns) < 2:
            return 0.0
        average_returns = episode_log_returns.mean()
        std_returns = episode_log_returns.std(ddof=1)
        if std_returns == 0:
            return 0.0
        return np.exp(average_returns/std_returns)
